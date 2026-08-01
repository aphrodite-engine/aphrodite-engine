#include <cuda_fp16.h>
#include "util.cuh"
#include "../util.h"
#include "../util.cuh"

#define NUM_THREADS 1024
#define BLOCK_SIZE 32768

#define uint64_cu unsigned long long int

__device__ inline uint64_cu warp_reduce_sum(uint64_cu v) {
  for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
    uint64_cu other_v = __shfl_down_sync(0xffffffff, v, offset);
    v += other_v;
  }
  return v;
}

__device__ inline uint64_cu block_reduce_sum(uint64_cu v) {
  __shared__ uint64_cu shared[NUM_THREADS / 32];

  int lane_id = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  v = warp_reduce_sum(v);

  if (lane_id == 0) shared[warp_id] = v;
  __syncthreads();

  int max_warp_id = NUM_THREADS / 32;
  if (warp_id == 0) {
    v = lane_id < max_warp_id ? shared[lane_id] : 0;
    v = warp_reduce_sum(v);
  }
  __syncthreads();
  return v;
}

__device__ inline bool isinf(half v) { return isinf(__half2float(v)); }

__device__ inline bool isnan(half v) { return isnan(__half2float(v)); }

template <typename T>
__global__ __launch_bounds__(NUM_THREADS) void count_inf_nan_kernel(
    const T* __restrict__ x, uint64_cu* __restrict__ y, uint64_cu numel) {
  uint64_cu idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  uint64_cu max_idx = MIN(blockIdx.x * BLOCK_SIZE + BLOCK_SIZE, numel);
  uint64_cu thread_inf = 0;
  uint64_cu thread_nan = 0;
  for (; idx < max_idx; idx += NUM_THREADS) {
    T val = x[idx];
    if (isinf(val)) thread_inf++;
    if (isnan(val)) thread_nan++;
  }

  thread_inf = block_reduce_sum(thread_inf);
  thread_nan = block_reduce_sum(thread_nan);

  if (threadIdx.x == 0) {
    atomicAdd(y + 0, thread_inf);
    atomicAdd(y + 1, thread_nan);
  }
}

/*
Count number of inf and NaN values in tensor

x: Tensor to test
y: Output, dtype kLong, shape (2,)
*/

void count_inf_nan(at::Tensor x, at::Tensor y) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      x.get_device_index());
  cudaStream_t stream = get_current_cuda_stream(x.get_device_index());
  TORCH_CHECK_DTYPE(y, kLong);

  uint64_cu numel = x.numel();
  uint64_cu num_blocks = CEIL_DIVIDE(numel, BLOCK_SIZE);

  if (x.scalar_type() == at::kHalf)
    count_inf_nan_kernel<half><<<num_blocks, NUM_THREADS, 0, stream>>>(
        (const half*)x.data_ptr(), (uint64_cu*)y.data_ptr(), numel);
  else if (x.scalar_type() == at::kFloat)
    count_inf_nan_kernel<float><<<num_blocks, NUM_THREADS, 0, stream>>>(
        (const float*)x.data_ptr(), (uint64_cu*)y.data_ptr(), numel);
  else
    TORCH_CHECK(false, "Unsupported dtype");
}

__global__ void make_gate_up_indices_kernel(int64_t* out,
                                            const int64_t* indices,
                                            int64_t count, int64_t offset) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  out[index] = indices[index];
  out[index + count] = indices[index] + offset;
}

void exl3_make_gate_up_indices(at::Tensor out, at::Tensor indices,
                               int64_t offset) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      indices.get_device_index());
  cudaStream_t stream = get_current_cuda_stream(indices.get_device_index());
  TORCH_CHECK_DTYPE(out, kLong);
  TORCH_CHECK_DTYPE(indices, kLong);
  TORCH_CHECK(out.numel() == indices.numel() * 2,
              "out must have twice as many elements as indices");
  const int64_t count = indices.numel();
  make_gate_up_indices_kernel<<<CEIL_DIVIDE(count, 256), 256, 0, stream>>>(
      static_cast<int64_t*>(out.data_ptr()),
      static_cast<const int64_t*>(indices.data_ptr()), count, offset);
}

__global__ void silu_mul_kernel(half* out, const half* gate, const half* up,
                                int64_t count) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float gate_f = __half2float(gate[index]);
  out[index] = __float2half_rn((gate_f / (1.0f + __expf(-gate_f))) *
                               __half2float(up[index]));
}

void exl3_silu_mul(at::Tensor out, at::Tensor gate, at::Tensor up) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      gate.get_device_index());
  cudaStream_t stream = get_current_cuda_stream(gate.get_device_index());
  TORCH_CHECK_DTYPE(out, kHalf);
  TORCH_CHECK_DTYPE(gate, kHalf);
  TORCH_CHECK_DTYPE(up, kHalf);
  TORCH_CHECK_NUMEL(out, gate);
  TORCH_CHECK_NUMEL(out, up);
  const int64_t count = out.numel();
  silu_mul_kernel<<<CEIL_DIVIDE(count, 256), 256, 0, stream>>>(
      static_cast<half*>(out.data_ptr()),
      static_cast<const half*>(gate.data_ptr()),
      static_cast<const half*>(up.data_ptr()), count);
}
