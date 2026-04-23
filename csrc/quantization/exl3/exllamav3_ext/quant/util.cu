#include <cuda_fp16.h>
#include "util.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"

#define NUM_THREADS 1024
#define BLOCK_SIZE 32768

#define uint64_cu unsigned long long int

__device__ inline uint64_cu warp_reduce_sum(uint64_cu v)
{
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        uint64_cu other_v = __shfl_down_sync(0xffffffff, v, offset);
        v += other_v;
    }
    return v;
}

__device__ inline uint64_cu block_reduce_sum(uint64_cu v)
{
    __shared__ uint64_cu shared[NUM_THREADS / 32];

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    v = warp_reduce_sum(v);

    if (lane_id == 0) shared[warp_id] = v;
    __syncthreads();

    int max_warp_id = NUM_THREADS / 32;
    if (warp_id == 0)
    {
        v = lane_id < max_warp_id ? shared[lane_id] : 0;
        v = warp_reduce_sum(v);
    }
    __syncthreads();
    return v;
}

__device__ inline bool isinf(half v)
{
    return isinf(__half2float(v));
}

__device__ inline bool isnan(half v)
{
    return isnan(__half2float(v));
}

template <typename T>
__global__ __launch_bounds__(NUM_THREADS)
void count_inf_nan_kernel
(
    const T* __restrict__ x,
    uint64_cu* __restrict__ y,
    uint64_cu numel
)
{
    uint64_cu idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    uint64_cu max_idx = MIN(blockIdx.x * BLOCK_SIZE + BLOCK_SIZE, numel);
    uint64_cu thread_inf = 0;
    uint64_cu thread_nan = 0;
    for (; idx < max_idx; idx += NUM_THREADS)
    {
        T val = x[idx];
        if (isinf(val)) thread_inf++;
        if (isnan(val)) thread_nan++;
    }

    thread_inf = block_reduce_sum(thread_inf);
    thread_nan = block_reduce_sum(thread_nan);

    if (threadIdx.x == 0)
    {
        atomicAdd(y + 0, thread_inf);
        atomicAdd(y + 1, thread_nan);
    }
}

/*
Count number of inf and NaN values in tensor

x: Tensor to test
y: Output, dtype kLong, shape (2,)
*/

void count_inf_nan
(
    at::Tensor x,
    at::Tensor y
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(y, kLong);

    uint64_cu numel = x.numel();
    uint64_cu num_blocks = CEIL_DIVIDE(numel, BLOCK_SIZE);

    if (x.dtype() == at::kHalf)
        count_inf_nan_kernel<half><<<num_blocks, NUM_THREADS, 0, stream>>>
        (
            (const half*) x.data_ptr(),
            (uint64_cu*) y.data_ptr(),
            numel
        );
    else if (x.dtype() == at::kFloat)
        count_inf_nan_kernel<float><<<num_blocks, NUM_THREADS, 0, stream>>>
        (
            (const float*) x.data_ptr(),
            (uint64_cu*) y.data_ptr(),
            numel
        );
    else
        TORCH_CHECK(false, "Unsupported dtype");
}