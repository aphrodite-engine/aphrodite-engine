#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"

// Dynamic MMQ optimization
struct mmq_args {
    const void * x;
    const void * y;
    void * dst;
    int64_t nrows_x;
    int64_t ncols_x;
    int64_t nrows_y;
    int64_t ncols_y;
    int64_t nrows_dst;
};

template<int type>
static size_t get_mmq_nbytes_shared(int mmq_x, int mmq_y, int cc) {
    // More conservative and accurate shared memory calculation
    if (type == 12) { // Q4_K
        // The actual shared memory usage is complex and depends on the specific kernel implementation
        // For safety, use a conservative estimate based on the original tile sizes
        // TODO: This is a guess, we should use the actual shared memory usage of the kernel
        return mmq_x * mmq_y * sizeof(float) * 4;  // Conservative estimate
    }
    // Fallback for other types
    return mmq_x * mmq_y * sizeof(float) * 4;
}

// Dynamic kernel launcher template
template<typename scalar_t, int qtype, int mmq_x>
static void launch_mmq_kernel_optimized(const mmq_args& args, cudaStream_t stream) {
    const int mmq_y = MMQ_Y_Q4_K;  // consistent MMQ_Y 
    const int nwarps = NWARPS_Q4_K;  // consistent NWARPS

    const int block_num_x = (args.nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (args.ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(32, nwarps, 1);  // WARP_SIZE_GGUF = 32

    if (qtype == 12) { // Q4_K only
        if (args.nrows_x % mmq_y == 0) {
            mul_mat_q4_K_dynamic<scalar_t, false, mmq_x><<<block_nums, block_dims, 0, stream>>>(
                args.x, args.y, (scalar_t*)args.dst, args.ncols_x, args.nrows_x, 
                args.ncols_y, args.nrows_y, args.nrows_dst);
        } else {
            mul_mat_q4_K_dynamic<scalar_t, true, mmq_x><<<block_nums, block_dims, 0, stream>>>(
                args.x, args.y, (scalar_t*)args.dst, args.ncols_x, args.nrows_x,
                args.ncols_y, args.nrows_y, args.nrows_dst);
        }
    }
    // Note: Q5_K and Q6_K not supported in this launcher - use original functions
}

// Optimized Q4_K kernel dispatch
template<typename scalar_t>
static void ggml_mul_mat_q4_K_q8_1_cuda_optimized(
    const void * vx, const void * vy, scalar_t * dst, 
    const int ncols_x, const int nrows_x, const int ncols_y, 
    const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    mmq_args args = {vx, vy, dst, nrows_x, ncols_x, nrows_y, ncols_y, nrows_dst};

    // Get device compute capability
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int cc = prop.major * 100 + prop.minor * 10;

    // CONSERVATIVE optimization
    // otherwise, we get NaN outputs
    // TODO: figure out what's happening here
    const int mmq_y = MMQ_Y_Q4_K;  // consistent MMQ_Y

    int mmq_x_best = MMQ_X_Q4_K;  // consistent MMQ_X

    // Only try modest improvements on modern GPUs for large batches
    const bool large_batch = ncols_y >= 1024; // conservative threshold
    if (cc >= 800 && large_batch) {  // only Ampere+ and large batches
        // test only 8, 16 as safe options
        for (int mmq_x : {8, 16}) {
            // Check shared memory constraints very conservatively
                         const size_t shmem_needed = get_mmq_nbytes_shared<12>(mmq_x, mmq_y, cc);
             const size_t shmem_limit = (size_t)(prop.sharedMemPerBlock * 0.5); // Use only 50% of available
             if (shmem_needed <= shmem_limit) {
                 const int ntiles_x = (ncols_y + mmq_x - 1) / mmq_x;
                 const int ntiles_x_orig = (ncols_y + MMQ_X_Q4_K - 1) / MMQ_X_Q4_K;

                 // Only use if it actually reduces the number of tiles
                 if (ntiles_x < ntiles_x_orig) {
                     mmq_x_best = mmq_x;
                }
            }
        }
    }

    switch (mmq_x_best) {
        case 8:  launch_mmq_kernel_optimized<scalar_t, 12, 8>(args, stream); break;
        case 16: launch_mmq_kernel_optimized<scalar_t, 12, 16>(args, stream); break;
        default:
            // fallback to original implementation for safety
            ggml_mul_mat_q4_K_q8_1_cuda(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, stream);
            break;
    }
}

// Optimized Q5_K kernel dispatch
template<typename scalar_t>
static void ggml_mul_mat_q5_K_q8_1_cuda_optimized(
    const void * vx, const void * vy, scalar_t * dst, 
    const int ncols_x, const int nrows_x, const int ncols_y, 
    const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    // For now, disable optimization for Q5_K to ensure stability
    // TODO: Enable conservative optimization after Q4_K is fully validated
    ggml_mul_mat_q5_K_q8_1_cuda(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, stream);
}

// Optimized Q6_K kernel dispatch
template<typename scalar_t>
static void ggml_mul_mat_q6_K_q8_1_cuda_optimized(
    const void * vx, const void * vy, scalar_t * dst, 
    const int ncols_x, const int nrows_x, const int ncols_y, 
    const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    // For now, disable optimization for Q6_K to ensure stability
    // TODO: Enable conservative optimization after Q4_K is fully validated
    ggml_mul_mat_q6_K_q8_1_cuda(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, stream);
}

// Q8 gemv
template <typename scalar_t>
static __global__ void quantize_q8_1(const scalar_t* __restrict__ x,
                                     void* __restrict__ vy, const int kx,
                                     const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, APHRODITE_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
    sum += APHRODITE_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx,
                                   const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[off * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

torch::Tensor ggml_dequantize(torch::Tensor W,  // quant weight
                              int64_t type, int64_t m, int64_t n,
                              std::optional<at::ScalarType> const& dtype) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  auto dtype_ = dtype.value_or(torch::kFloat16);
  auto options = torch::TensorOptions().dtype(dtype_).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  APHRODITE_DISPATCH_FLOATING_TYPES(DW.scalar_type(), "ggml_dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    to_cuda((void*)W.data_ptr(), (scalar_t*)DW.data_ptr(), m * n, stream);
  });

  return DW;
}

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W,  // quant weight
                                  torch::Tensor X,  // input
                                  int64_t type, int64_t row) {
  int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({1, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({1, padded / 32 * 9}, options);
  APHRODITE_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_mul_mat_vec_a8", [&] {
        quantize_row_q8_1_cuda<scalar_t>(
            (scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, 1, stream);
        switch (type) {
          case 2:
            mul_mat_vec_q4_0_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 3:
            mul_mat_vec_q4_1_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 6:
            mul_mat_vec_q5_0_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 7:
            mul_mat_vec_q5_1_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 8:
            mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 10:
            mul_mat_vec_q2_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 11:
            mul_mat_vec_q3_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 12:
            mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 13:
            mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 14:
            mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 16:
            mul_mat_vec_iq2_xxs_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 17:
            mul_mat_vec_iq2_xs_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 18:
            mul_mat_vec_iq3_xxs_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 19:
            mul_mat_vec_iq1_s_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 20:
            mul_mat_vec_iq4_nl_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 21:
            mul_mat_vec_iq3_s_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 22:
            mul_mat_vec_iq2_s_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 23:
            mul_mat_vec_iq4_xs_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
          case 29:
            mul_mat_vec_iq1_m_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, stream);
            break;
        }
      });
  return Y;
}

torch::Tensor ggml_mul_mat_a8(torch::Tensor W,  // quant weight
                              torch::Tensor X,  // input
                              int64_t type, int64_t row) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  int batch = X.sizes()[0];
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({batch, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({batch, padded / 32 * 9}, options);
  APHRODITE_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, batch, stream);

    switch (type) {
      case 2:
        ggml_mul_mat_q4_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 3:
        ggml_mul_mat_q4_1_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 6:
        ggml_mul_mat_q5_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 7:
        ggml_mul_mat_q5_1_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 8:
        ggml_mul_mat_q8_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 10:
        ggml_mul_mat_q2_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 11:
        ggml_mul_mat_q3_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 12:
        ggml_mul_mat_q4_K_q8_1_cuda_optimized(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 13:
        ggml_mul_mat_q5_K_q8_1_cuda_optimized(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 14:
        ggml_mul_mat_q6_K_q8_1_cuda_optimized(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
    }
  });
  return Y;
}

torch::Tensor ggml_moe_a8(torch::Tensor X,  // input
                          torch::Tensor W,  // expert weights
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t type,
                          int64_t row, int64_t top_k, int64_t tokens) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  APHRODITE_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, tokens, stream);
    switch (type) {
      case 2:
        ggml_moe_q4_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 3:
        ggml_moe_q4_1_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 6:
        ggml_moe_q5_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 7:
        ggml_moe_q5_1_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 8:
        ggml_moe_q8_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 10:
        ggml_moe_q2_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 11:
        ggml_moe_q3_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 12:
        ggml_moe_q4_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 13:
        ggml_moe_q5_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 14:
        ggml_moe_q6_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
    }
  });
  return Y;
}

int64_t ggml_moe_get_block_size(int64_t type) {
  switch (type) {
    case 2:
      return MOE_X_Q4_0;
    case 3:
      return MOE_X_Q4_1;
    case 6:
      return MOE_X_Q5_0;
    case 7:
      return MOE_X_Q5_1;
    case 8:
      return MOE_X_Q8_0;
    case 10:
      return MOE_X_Q2_K;
    case 11:
      return MOE_X_Q3_K;
    case 12:
      return MOE_X_Q4_K;
    case 13:
      return MOE_X_Q5_K;
    case 14:
      return MOE_X_Q6_K;
  }
  return 0;
}
