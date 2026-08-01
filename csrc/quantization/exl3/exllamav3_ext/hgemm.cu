#include <cuda_fp16.h>
#include "hgemm.cuh"
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include <limits>

/*

Row-major matmul using cuBLAS, a @ b -> c
- if c is float16, operation is float16 @ float16 -> float16 (float16
accumulate)
- if c is float32, operation is float16 @ float16 -> float32 (float32
accumulate)
*/

using bfloat16 = __nv_bfloat16;

static void hgemm_gemmex_impl(at::Tensor a, at::Tensor b, at::Tensor c,
                              cudaStream_t stream) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());

  bool output_fp32 = c.scalar_type() == at::kFloat;
  bool output_fp16 = c.scalar_type() == at::kHalf;

  TORCH_CHECK(output_fp32 || output_fp16, "c must be float32 or float16");

  // Check shapes of a,b,c are compatible
  TORCH_CHECK_DTYPE(a, kHalf);
  TORCH_CHECK_DTYPE(b, kHalf);
  TORCH_CHECK_DIM(b, 2);
  TORCH_CHECK(c.dim() >= 2, "c must have at least 2 dimensions");
  TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
  TORCH_CHECK_SHAPES(b, 1, c, -1, 1);
  TORCH_CHECK(c.stride(-1) == 1, "c must have contiguous columns");

  const half* a_ptr = (const half*)a.data_ptr();
  const half* b_ptr = (const half*)b.data_ptr();

  int size_k = a.size(-1);
  int size_m = a.numel() / size_k;
  int size_n = b.size(-1);
  int64_t c_stride_m = c.stride(-2);
  TORCH_CHECK(c_stride_m >= size_n, "c row stride is too small");
  TORCH_CHECK(c_stride_m <= std::numeric_limits<int>::max(),
              "c row stride is too large");

  // Set cuBLAS modes and workspace
  cublasHandle_t cublas_handle = get_current_cuda_blas_handle();
  cublasSetStream(cublas_handle, stream);
  cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
  int device;
  cudaGetDevice(&device);
  void* ws = DevCtx::instance().get_ws(device);
  cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

  float alpha_ = 1.0f;
  float beta_ = 0.0f;
  cudaDataType_t c_type = output_fp32 ? CUDA_R_32F : CUDA_R_16F;
  auto r = cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size_n, size_m,
                        size_k, &alpha_, b_ptr, CUDA_R_16F, size_n, a_ptr,
                        CUDA_R_16F, size_k, &beta_, c.data_ptr(), c_type,
                        (int)c_stride_m, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cublas_check(r);
  cuda_check(cudaPeekAtLastError());
}

void hgemm_gr(at::Tensor a, at::Tensor b, at::Tensor c, Graph* graph) {
  cudaStream_t stream = graph ? graph->capture_stream
                              : get_current_cuda_stream(a.get_device_index());
  hgemm_gemmex_impl(a, b, c, stream);

  if (graph) graph->need_cublas = true;
}

void hgemm(at::Tensor a, at::Tensor b, at::Tensor c) {
  hgemm_gr(a, b, c, nullptr);
}
