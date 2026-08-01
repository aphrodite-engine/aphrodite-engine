#include <cuda_fp16.h>
#include "exl3_gemv_int8.cuh"
#include "exl3_gemv_int8_kernel.cuh"
#include "comp_units/exl3_gemv_int8_instances.cuh"
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_devctx.cuh"
#include "hadamard_inner.cuh"
#include <cooperative_groups.h>
#include <cstdlib>
#include <set>
#include <map>

// Mode 0: disabled; 1: int8 + error-feedback residual pass (~15-16 bit
// effective activation precision, KL at parity with fp16 or better); 2: plain
// int8 (cheaper, ~0.9% output RMS deviation).
static int _exl3_gemv_int8_mode = 0;
bool _exl3_gemv_int8_mode_chk = false;

static int exl3_gemv_int8_mode() {
  if (_exl3_gemv_int8_mode_chk) return _exl3_gemv_int8_mode;
  const char* e = getenv("EXL3_INT8_GEMV");
  _exl3_gemv_int8_mode = e ? atoi(e) : 2;
  return _exl3_gemv_int8_mode;
}

bool exl3_gemv_int8_enabled() { return exl3_gemv_int8_mode() != 0; }

// Highest K the int8 path accepts; above it the regular kernel wins. The fp16
// pipeline must be compute/latency-limited for the reduced per-weight work to
// matter, and where that ends is per-arch. Ampere is DRAM-bound from K = 6 up
// (3090: int8 -29/-22/-9/-6% at K=2/3/4/5, then -11..-26% on wide shapes at
// K=6); Ada is marginal at K=6 (4090: -0..+7%, residual mode loses) and keeps
// the conservative gate. Hopper's fp16 kernel is per-SM INT-throughput-bound at
// K = 6 (H200, issue #242: +26/+57% per call, +16% e2e), and Blackwell measures
// the same way (5090: +7..+19% at K=6 across shapes, fp16 kernel at only
// ~65-78% of DRAM peak; K=7/8 are flat). EXL3_INT8_GEMV_MAX_K overrides the
// per-arch default for testing on unmeasured parts (kernel instances exist up
// to K = 8; at m == 1, K = 7..8 fall through to the cooperative kernel)
int exl3_gemv_int8_max_k(int device) {
  static const int env_max_k = [] {
    const char* e = getenv("EXL3_INT8_GEMV_MAX_K");
    return e ? atoi(e) : 0;
  }();
  if (env_max_k) return MIN(env_max_k, 8);
  int cc = DevCtx::instance().get_cc(device);
  return (cc == CC_HOPPER || cc == CC_BLACKWELL) ? 6 : 5;
}

struct GemvInt8Workspace {
  int* ws = nullptr;
  size_t ws_ints = 0;
};

static GemvInt8Workspace gemv_ws[MAX_DEVICES];
static std::set<void*> gemv_attr_set[MAX_DEVICES];
static std::map<std::pair<void*, size_t>, int> gemv_occ_cache[MAX_DEVICES];

typedef void (*gemv_int8_coop_fn)(const half*, const uint16_t*, void*, int, int,
                                  int, int*, const half*, half*, const half*);

static void* select_gemv_int8_kernel(int K, bool c_fp32, bool residual) {
  switch (K) {
    case 1:
      return exl3_gemv_int8_coop_sel_k1(c_fp32, residual);
    case 2:
      return exl3_gemv_int8_coop_sel_k2(c_fp32, residual);
    case 3:
      return exl3_gemv_int8_coop_sel_k3(c_fp32, residual);
    case 4:
      return exl3_gemv_int8_coop_sel_k4(c_fp32, residual);
    case 5:
      return exl3_gemv_int8_coop_sel_k5(c_fp32, residual);
    case 6:
      return exl3_gemv_int8_coop_sel_k6(c_fp32, residual);
    case 7:
      return exl3_gemv_int8_coop_sel_k7(c_fp32, residual);
    case 8:
      return exl3_gemv_int8_coop_sel_k8(c_fp32, residual);
  }
  return nullptr;
}

static void* select_gemv_int8_sq_kernel(int K, int M, bool c_fp32,
                                        bool residual) {
  switch (K) {
    case 1:
      return exl3_gemv_int8_sq_sel_k1(M, c_fp32, residual);
    case 2:
      return exl3_gemv_int8_sq_sel_k2(M, c_fp32, residual);
    case 3:
      return exl3_gemv_int8_sq_sel_k3(M, c_fp32, residual);
    case 4:
      return exl3_gemv_int8_sq_sel_k4(M, c_fp32, residual);
    case 5:
      return exl3_gemv_int8_sq_sel_k5(M, c_fp32, residual);
    case 6:
      return exl3_gemv_int8_sq_sel_k6(M, c_fp32, residual);
  }
  return nullptr;
}

// Fixed-size per-device workspace shared by the sq and coop paths, allocated
// once and never reallocated: the pointer is baked as a kernel argument into
// captured CUDA graphs, so growing the buffer would leave every previously
// captured graph with a dangling workspace pointer (and let a reallocation
// clobber the self-resetting completion counters at the start of the buffer).
// Zeroed at allocation so the counters begin at zero. Callers must reject work
// that exceeds the fixed size (returns nullptr) and fall through to a
// non-workspace path.
#define GEMV_INT8_WS_INTS (WORKSPACE_SIZE / sizeof(int))  // 16 MB
static int* gemv_int8_get_ws(int device, size_t ws_ints) {
  if (ws_ints > GEMV_INT8_WS_INTS) return nullptr;
  GemvInt8Workspace& ws = gemv_ws[device];
  if (!ws.ws) {
    cuda_check(cudaMalloc(&ws.ws, GEMV_INT8_WS_INTS * sizeof(int)));
    cuda_check(cudaMemset(ws.ws, 0, GEMV_INT8_WS_INTS * sizeof(int)));
    ws.ws_ints = GEMV_INT8_WS_INTS;
  }
  return ws.ws;
}

// m == 1 fast path: per-slice-scale kernel, regular launch. Returns false to
// fall through to the cooperative kernel (and from there to the regular fp16
// kernel).
static bool exl3_gemv_int8_sq(const half* A_ptr, const uint16_t* B_ptr,
                              void* C_ptr, int size_m, int size_k, int size_n,
                              int K, bool c_fp32, bool residual,
                              const half* suh_ptr, half* A_had_ptr,
                              const half* svh_ptr, int device, int num_sms,
                              cudaStream_t stream, Graph* graph) {
  if (size_m > 2) return false;
  int M = size_m;
  void* fn = select_gemv_int8_sq_kernel(K, M, c_fp32, residual);
  if (!fn) return false;

  int rows_max = gemv_int8_sq_rows_max(M, residual);

  // Mirror of the kernel's work decomposition (single-wave rule with a
  // half-wave floor)
  auto decomp = [&](int grid_, int& ksplit, int& rows_per) {
    int rows_total = size_k / 16;
    int nb256 = size_n / 256;
    int r = CEIL_DIVIDE(rows_total * nb256, grid_);
    rows_per = (MAX(r, MIN(2 * r, 32)) + 7) & ~7;
    rows_per = MAX(rows_per, SQ_MINROWS);
    rows_per = MIN(rows_per, rows_max);
    rows_per = MIN(rows_per, (rows_total + 7) & ~7);
    ksplit = CEIL_DIVIDE(rows_total, rows_per);
  };
  auto smem_for = [&](int rows_per) -> size_t {
    size_t stage =
        gemv_int8_stage_smem(K) ? (size_t)8 * GEMV_STAGE_D * 16 * K * 4 : 0;
    return (size_t)rows_per * 16 * 2 +
           (size_t)rows_per * 16 * 4 * M * (residual ? 2 : 1) + stage +
           (size_t)2 * M * 128 * 4;
  };

  if (gemv_attr_set[device].find(fn) == gemv_attr_set[device].end()) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_for(rows_max));
    // Match the tensor-core kernels' shared-memory carveout: these kernels
    // interleave with them (hundreds of launches per decoded token), and a
    // smaller carveout would make the GPU drain and reconfigure the SMs on
    // every transition - measured at ~4 us per launch in graph replay
    cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    gemv_attr_set[device].insert(fn);
    cuda_check(cudaPeekAtLastError());
  }

  int ksplit, rows_per;
  decomp(6 * num_sms, ksplit, rows_per);
  size_t smem_guess = smem_for(rows_per);
  int maxb;
  auto occ_key = std::make_pair(fn, smem_guess);
  auto occ_it = gemv_occ_cache[device].find(occ_key);
  if (occ_it != gemv_occ_cache[device].end())
    maxb = occ_it->second;
  else {
    maxb = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, NUM_THREADS,
                                                  smem_guess);
    gemv_occ_cache[device][occ_key] = maxb;
  }
  int grid = MIN(MAX(maxb, 1) * num_sms, 1024);
  decomp(grid, ksplit, rows_per);
  size_t smem = smem_for(rows_per);
  if (ksplit > SQ_KSPLIT_CAP) return false;
  if (size_n / 256 > SQ_COUNTERS_CAP) return false;

  int pstride = size_n * (residual ? 2 : 1);
  int* ws_ptr =
      gemv_int8_get_ws(device, SQ_WS_RESERVED + (size_t)ksplit * M * pstride);
  if (!ws_ptr) return false;

  void* kernelArgs[] = {(void*)&A_ptr,  (void*)&B_ptr,   (void*)&C_ptr,
                        (void*)&size_m, (void*)&size_k,  (void*)&size_n,
                        (void*)&ws_ptr, (void*)&suh_ptr, (void*)&A_had_ptr,
                        (void*)&svh_ptr};

  cudaError_t err = cudaLaunchKernel(fn, dim3(grid), dim3(NUM_THREADS),
                                     kernelArgs, smem, stream);
  if (graph) {
    graph->record_param(fn, GP_gemm_A, 0);
    graph->record_param(fn, GP_gemm_B_trellis, 1);
    graph->record_param(fn, GP_gemm_C, 2);
    graph->record_param(fn, GP_gemm_B_suh, 7);
    graph->record_param(fn, GP_gemm_A_had, 8);
    graph->record_param(fn, GP_gemm_B_svh, 9);
    graph->record_param(fn, GP_end, 0);
  }
  if (err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
}

bool exl3_gemv_int8(const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
                    const c10::optional<at::Tensor>& suh,
                    const c10::optional<at::Tensor>& A_had,
                    const c10::optional<at::Tensor>& svh, cudaStream_t stream,
                    Graph* graph) {
  if (!suh.has_value() || !A_had.has_value() || !svh.has_value()) return false;

  int K = B.size(2) / 16;
  int size_k = A.size(-1);
  int size_n = B.size(1) * 16;
  int size_m = A.numel() / size_k;
  if (size_n % 256) return false;
  if (size_k % 128) return false;

  int device;
  cudaGetDevice(&device);
  if (K < 1 || K > exl3_gemv_int8_max_k(device)) return false;
  int num_sms = DevCtx::instance().get_num_sms(device);
  bool c_fp32 = C.scalar_type() == at::kFloat;
  bool residual = exl3_gemv_int8_mode() == 1;

  // Per-slice-scale kernel: m == 1, plus m == 2 in plain int8 mode (rows share
  // the decoded weights and the B stream; measured on 3090, larger m and
  // batched residual lose to the fp16 tensor-core kernel). Falls through to the
  // cooperative kernel on a constraint miss at m == 1; batched rows beyond the
  // gate go straight to the regular kernel.
  if (size_m <= (residual ? 1 : 2) &&
      exl3_gemv_int8_sq(
          (const half*)A.data_ptr(), (const uint16_t*)B.data_ptr(),
          C.data_ptr(), size_m, size_k, size_n, K, c_fp32, residual,
          (const half*)suh->data_ptr(), (half*)A_had->data_ptr(),
          (const half*)svh->data_ptr(), device, num_sms, stream, graph))
    return true;
  if (size_m > 1) return false;

  void* fn = select_gemv_int8_kernel(K, c_fp32, residual);
  if (!fn) return false;

  // Mirror the kernel's work decomposition for the shared memory size; grid =
  // max co-resident blocks (natural register allocation measures faster than
  // forcing higher occupancy)
  auto smem_for_grid = [&](int grid_) -> size_t {
    int rows_total = size_k / 16;
    int nb256 = size_n / 256;
    int smem_rows_max = residual ? 384 : 768;
    int ksplit = CEIL_DIVIDE(4 * grid_, nb256);
    ksplit = MAX(ksplit, CEIL_DIVIDE(rows_total, smem_rows_max));
    ksplit = MIN(ksplit, rows_total);
    int rows_per = CEIL_DIVIDE(rows_total, ksplit);
    size_t stage =
        gemv_int8_stage_smem(K) ? (size_t)8 * GEMV_STAGE_D * 16 * K * 4 : 0;
    return MAX((size_t)rows_per * 16 * 4 * (residual ? 2 : 1) + stage,
               (size_t)8 * 128 * 4);
  };

  if (gemv_attr_set[device].find(fn) == gemv_attr_set[device].end()) {
    // Upper bound over all shapes: smem_rows_max * 64 B
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         768 * 16 * 4 + GEMV_STAGE_MAX_BYTES);
    // Match the tensor-core kernels' shared-memory carveout: these kernels
    // interleave with them (hundreds of launches per decoded token), and a
    // smaller carveout would make the GPU drain and reconfigure the SMs on
    // every transition - measured at ~4 us per launch in graph replay
    cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    gemv_attr_set[device].insert(fn);
    cuda_check(cudaPeekAtLastError());
  }

  size_t smem_guess = smem_for_grid(6 * num_sms);
  int maxb;
  auto occ_key = std::make_pair(fn, smem_guess);
  auto occ_it = gemv_occ_cache[device].find(occ_key);
  if (occ_it != gemv_occ_cache[device].end())
    maxb = occ_it->second;
  else {
    maxb = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, NUM_THREADS,
                                                  smem_guess);
    gemv_occ_cache[device][occ_key] = maxb;
  }
  int grid = MIN(MAX(maxb, 1) * num_sms, 1024);
  size_t smem = smem_for_grid(grid);

  // Coop region beyond the sq-reserved prefix: [2n accs][4m qsums][grid partial
  // maxes]
  size_t ws_ints = SQ_WS_RESERVED + (size_t)2 * size_n + 4 * size_m + 1024;
  int* ws_base = gemv_int8_get_ws(device, ws_ints);
  if (!ws_base) return false;
  int* ws_ptr = ws_base + SQ_WS_RESERVED;

  const half* A_ptr = (const half*)A.data_ptr();
  const uint16_t* B_ptr = (const uint16_t*)B.data_ptr();
  void* C_ptr = C.data_ptr();
  const half* suh_ptr = (const half*)suh->data_ptr();
  half* A_had_ptr = (half*)A_had->data_ptr();  // scratch; used through a raw
                                               // half* like the regular kernel
  const half* svh_ptr = (const half*)svh->data_ptr();

  void* kernelArgs[] = {(void*)&A_ptr,  (void*)&B_ptr,   (void*)&C_ptr,
                        (void*)&size_m, (void*)&size_k,  (void*)&size_n,
                        (void*)&ws_ptr, (void*)&suh_ptr, (void*)&A_had_ptr,
                        (void*)&svh_ptr};

  auto add_graph_args = [&](void* kernel_ptr) {
    if (graph) {
      graph->record_param(kernel_ptr, GP_gemm_A, 0);
      graph->record_param(kernel_ptr, GP_gemm_B_trellis, 1);
      graph->record_param(kernel_ptr, GP_gemm_C, 2);
      graph->record_param(kernel_ptr, GP_gemm_B_suh, 7);
      graph->record_param(kernel_ptr, GP_gemm_A_had, 8);
      graph->record_param(kernel_ptr, GP_gemm_B_svh, 9);
      graph->record_param(kernel_ptr, GP_end, 0);
    }
  };

  cudaError_t err = cudaLaunchCooperativeKernel(fn, grid, NUM_THREADS,
                                                kernelArgs, smem, stream);
  add_graph_args((void*)fn);

  if (err != cudaSuccess) {
    // e.g. cooperative launch unsupported or co-residency violated: fall back
    // to the regular kernel
    cudaGetLastError();
    return false;
  }
  return true;
}
