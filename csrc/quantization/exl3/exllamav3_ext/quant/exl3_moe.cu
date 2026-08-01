#include <cuda_fp16.h>
#include "exl3_gemm.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "comp_units/exl3_moe_instances.cuh"
#include "exl3_devctx.cuh"
#include <set>

int exl3_moe_max_concurrency(int device) {
  int num_sms = DevCtx::instance().get_num_sms(device);
  return num_sms / MOE_SMS_PER_EXPERT;
}

std::set<void*> moe_kernel_attr_set[MAX_DEVICES] = {};

fp_exl3_moe_kernel exl3_moe_kernel_instances[] = {
    // [K][cb - 1][N_off]: K = 0 switches Kg/Ku/Kd at runtime, K > 0 =
    // compile-time Kg = Ku = Kd
    exl3_moe_kernel_k0_n128_cb1(), exl3_moe_kernel_k0_n256_cb1(),
    exl3_moe_kernel_k0_n128_cb2(), exl3_moe_kernel_k0_n256_cb2(),
    exl3_moe_kernel_k1_n128_cb1(), exl3_moe_kernel_k1_n256_cb1(),
    exl3_moe_kernel_k1_n128_cb2(), exl3_moe_kernel_k1_n256_cb2(),
    exl3_moe_kernel_k2_n128_cb1(), exl3_moe_kernel_k2_n256_cb1(),
    exl3_moe_kernel_k2_n128_cb2(), exl3_moe_kernel_k2_n256_cb2(),
    exl3_moe_kernel_k3_n128_cb1(), exl3_moe_kernel_k3_n256_cb1(),
    exl3_moe_kernel_k3_n128_cb2(), exl3_moe_kernel_k3_n256_cb2(),
    exl3_moe_kernel_k4_n128_cb1(), exl3_moe_kernel_k4_n256_cb1(),
    exl3_moe_kernel_k4_n128_cb2(), exl3_moe_kernel_k4_n256_cb2(),
    exl3_moe_kernel_k5_n128_cb1(), exl3_moe_kernel_k5_n256_cb1(),
    exl3_moe_kernel_k5_n128_cb2(), exl3_moe_kernel_k5_n256_cb2(),
    exl3_moe_kernel_k6_n128_cb1(), exl3_moe_kernel_k6_n256_cb1(),
    exl3_moe_kernel_k6_n128_cb2(), exl3_moe_kernel_k6_n256_cb2(),
    exl3_moe_kernel_k7_n128_cb1(), exl3_moe_kernel_k7_n256_cb1(),
    exl3_moe_kernel_k7_n128_cb2(), exl3_moe_kernel_k7_n256_cb2(),
    exl3_moe_kernel_k8_n128_cb1(), exl3_moe_kernel_k8_n256_cb1(),
    exl3_moe_kernel_k8_n128_cb2(), exl3_moe_kernel_k8_n256_cb2()};

/*
Fused mixture-of-experts MLP operation for EXL3 weights

inputs:
    hidden_state:
        input hidden state - shape (bsz, hidden_dim) - fp16

    output_state:
        output hidden state - shape (bsz, hidden_dim) - fp32
        zero-initialized

    expert_count:
        bincount of expert indices across all tokens in batch - shape
(num_experts + 1,) - int64 last item is ignored, used for the case where some
tokens may activate less than num_experts_per_token experts (specifically in
expert split mode)

    token_sorted:
        token indices, sorted by expert - shape (bsz * num_experts_per_tok,)  -
int64

    weight_sorted:
        routing weight per token, sorted by expert - shape (bsz *
num_experts_per_tok,) - fp16

    temp_state_g:
    temp_state_u:
        temp state storage - shape (concurrency, max_tokens_per_expert,
hidden_dim), fp16

    temp_intermediate_g
    temp_intermediate_u:
        temp intermediate storage - shape (concurrency, max_tokens_per_expert,
intermediate_dim), fp16

    act_function:
        int, see exl3_moe.cuh

    K_gate
    K_up
    K_down:
        int, bitrates for gate, up, down tensors

    gate_ptrs_trellis
    gate_ptrs_suh
    gate_ptrs_svh
    up_ptrs_trellis
    up_ptrs_suh
    up_ptrs_svh
    down_ptrs_trellis
    down_ptrs_suh
    down_ptrs_svh:
        tensors of data_ptrs to quantized tensor data - each shape
(num_experts,) - void*

    gate_mcg
    gate_mul1
    up_mcg
    up_mul1
    down_mcg
    down_mul1:
        bool, codebook flags

    num_active:
        number of experts with 0 < token count <= max_tokens_per_expert, i.e.
the number of experts this kernel will process. Used to size the launch: fewer,
wider expert groups when few experts are active. Pass -1 if unknown (defaults to
MOE_SMS_PER_EXPERT-wide groups at max concurrency)
*/

void exl3_moe(const at::Tensor& hidden_state, const at::Tensor& output_state,
              const at::Tensor& expert_count, const at::Tensor& token_sorted,
              const at::Tensor& weight_sorted,

              const at::Tensor& temp_state_g, const at::Tensor& temp_state_u,
              const at::Tensor& temp_intermediate_g,
              const at::Tensor& temp_intermediate_u,

              const int act_function,

              const int K_gate, const int K_up, const int K_down,

              const at::Tensor& gate_ptrs_trellis,
              const at::Tensor& gate_ptrs_suh, const at::Tensor& gate_ptrs_svh,
              const at::Tensor& up_ptrs_trellis, const at::Tensor& up_ptrs_suh,
              const at::Tensor& up_ptrs_svh,
              const at::Tensor& down_ptrs_trellis,
              const at::Tensor& down_ptrs_suh, const at::Tensor& down_ptrs_svh,

              const bool gate_mcg, const bool gate_mul1, const bool up_mcg,
              const bool up_mul1, const bool down_mcg, const bool down_mul1,

              const float act_limit, const int num_active) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      hidden_state.get_device_index());
  cudaStream_t stream =
      get_current_cuda_stream(hidden_state.get_device_index());

  // Nothing for the fused kernel to do
  if (num_active == 0) return;

  // Validate args
  TORCH_CHECK_DTYPE(hidden_state, kHalf);
  TORCH_CHECK_DIM(hidden_state, 2);
  size_t bsz = hidden_state.size(0);
  size_t hidden_dim = hidden_state.size(1);

  TORCH_CHECK_DTYPE(output_state, kFloat);
  TORCH_CHECK_SHAPES_FULL(output_state, hidden_state);

  TORCH_CHECK_DTYPE(expert_count, kLong);
  TORCH_CHECK_DIM(expert_count, 1);
  size_t num_experts = expert_count.size(0) - 1;

  TORCH_CHECK_DTYPE(token_sorted, kLong);
  TORCH_CHECK_DIM(token_sorted, 1);
  TORCH_CHECK_SHAPES_FULL(token_sorted, weight_sorted);
  size_t num_experts_per_tok = token_sorted.size(0) / bsz;

  TORCH_CHECK_DTYPE(temp_state_g, kHalf);
  TORCH_CHECK_DTYPE(temp_state_u, kHalf);
  TORCH_CHECK_DIM(temp_state_g, 3);
  TORCH_CHECK_SHAPES(temp_state_g, 2, hidden_state, 1, 1);
  TORCH_CHECK_SHAPES_FULL(temp_state_g, temp_state_u);
  size_t max_tokens_per_expert = temp_state_g.size(1);
  size_t concurrency = temp_state_g.size(0);

  TORCH_CHECK_DTYPE(temp_intermediate_g, kHalf);
  TORCH_CHECK_DTYPE(temp_intermediate_u, kHalf);
  TORCH_CHECK_DIM(temp_intermediate_g, 3);
  TORCH_CHECK_DIM(temp_intermediate_u, 3);
  TORCH_CHECK_SHAPES_FULL(temp_intermediate_g, temp_intermediate_u);
  TORCH_CHECK_SHAPES(temp_intermediate_g, 1, temp_state_g, 1, 1);
  size_t intermediate_dim = temp_intermediate_g.size(2);

  // TORCH_CHECK(!(gate_mcg && gate_mul1), "Specified both mcg and mul1
  // (gate)"); TORCH_CHECK(!(up_mcg && up_mul1), "Specified both mcg and mul1
  // (up)"); TORCH_CHECK(!(down_mcg && down_mul1), "Specified both mcg and mul1
  // (down)");
  TORCH_CHECK(gate_mcg == up_mcg && up_mcg == down_mcg &&
                  gate_mul1 == up_mul1 && up_mul1 == down_mul1,
              "MoE kernel: gate/up/down must share the same codebook");
  TORCH_CHECK(gate_mcg != gate_mul1,
              "MoE kernel: Only mcg and mul1 codebooks are supported");
  const int cb_idx = gate_mul1 ? 1 : 0;

  // TORCH_CHECK(act_function == MOE_ACT_SILU, "MoE kernel: Only SiLU is
  // currently supported");

  int K = 0;
  if (K_gate == K_up && K_up == K_down) K = K_gate;

  TORCH_CHECK_DIM(gate_ptrs_trellis, 1);
  TORCH_CHECK(gate_ptrs_trellis.size(0) == num_experts,
              "Number of gate tensors doesn't match num_experts");
  TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, gate_ptrs_suh);
  TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, gate_ptrs_svh);
  TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, up_ptrs_trellis);
  TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, up_ptrs_suh);
  TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, up_ptrs_svh);
  TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, down_ptrs_trellis);
  TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, down_ptrs_suh);
  TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, down_ptrs_svh);

  // Device properties
  int device;
  cudaGetDevice(&device);
  int num_sms = DevCtx::instance().get_num_sms(device);
  int cc = DevCtx::instance().get_cc(device);
  int* locks = DevCtx::instance().get_locks(device);

  // Launch. All blocks of the grid must be co-resident for the group barriers,
  // so groups * width <= num_sms. With a known number of active experts, launch
  // only as many groups as there are experts and widen them to use the freed
  // SMs, up to MOE_MAX_SMS_PER_EXPERT
  int block_dim = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;
  TORCH_CHECK(concurrency * MOE_SMS_PER_EXPERT <= num_sms,
              "Concurrency too high for device num_sms");
  int num_groups = MIN((int)concurrency, MOE_MAX_GROUPS);
  int group_size = MOE_SMS_PER_EXPERT;
  if (num_active > 0) {
    num_groups = MIN(num_groups, num_active);
    group_size = MIN(num_sms / num_groups, MOE_MAX_SMS_PER_EXPERT);
  }
  dim3 grid_dim(group_size, 1, num_groups);

  int N_off = 0;
  if (hidden_dim % 256 == 0 && intermediate_dim % 256 == 0) N_off = 1;
  fp_exl3_moe_kernel kernel =
      exl3_moe_kernel_instances[4 * K + 2 * cb_idx + N_off];

  if (moe_kernel_attr_set[device].find((void*)kernel) ==
      moe_kernel_attr_set[device].end()) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM_MAX);
    moe_kernel_attr_set[device].insert((void*)kernel);
    cuda_check(cudaPeekAtLastError());
  }

  void* _hidden_state = hidden_state.data_ptr();
  void* _temp_state_g = temp_state_g.data_ptr();
  void* _temp_state_u = temp_state_u.data_ptr();
  void* _temp_intermediate_g = temp_intermediate_g.data_ptr();
  void* _temp_intermediate_u = temp_intermediate_u.data_ptr();
  void* _output_state = output_state.data_ptr();

  void* _gate_ptrs_trellis = gate_ptrs_trellis.data_ptr();
  void* _gate_ptrs_suh = gate_ptrs_suh.data_ptr();
  void* _gate_ptrs_svh = gate_ptrs_svh.data_ptr();
  void* _up_ptrs_trellis = up_ptrs_trellis.data_ptr();
  void* _up_ptrs_suh = up_ptrs_suh.data_ptr();
  void* _up_ptrs_svh = up_ptrs_svh.data_ptr();
  void* _down_ptrs_trellis = down_ptrs_trellis.data_ptr();
  void* _down_ptrs_suh = down_ptrs_suh.data_ptr();
  void* _down_ptrs_svh = down_ptrs_svh.data_ptr();

  void* _expert_count = expert_count.data_ptr();
  void* _token_sorted = token_sorted.data_ptr();
  void* _weight_sorted = weight_sorted.data_ptr();

  void* kernelArgs[] = {&_hidden_state,
                        &_temp_state_g,
                        &_temp_state_u,
                        &_temp_intermediate_g,
                        &_temp_intermediate_u,
                        &_output_state,
                        &_gate_ptrs_trellis,
                        &_gate_ptrs_suh,
                        &_gate_ptrs_svh,
                        &_up_ptrs_trellis,
                        &_up_ptrs_suh,
                        &_up_ptrs_svh,
                        &_down_ptrs_trellis,
                        &_down_ptrs_suh,
                        &_down_ptrs_svh,
                        &_expert_count,
                        &_token_sorted,
                        &_weight_sorted,
                        (void*)&hidden_dim,
                        (void*)&intermediate_dim,
                        (void*)&num_experts,
                        (void*)&num_experts_per_tok,
                        (void*)&max_tokens_per_expert,
                        (void*)&num_groups,
                        (void*)&act_limit,
                        (void*)&act_function,
                        (void*)&K_gate,
                        (void*)&K_up,
                        (void*)&K_down,
                        (void*)&locks};

  cudaLaunchKernel((void*)kernel, grid_dim, block_dim, kernelArgs, SMEM_MAX,
                   stream);

  cuda_check(cudaPeekAtLastError());
}
