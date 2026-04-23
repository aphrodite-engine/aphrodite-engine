#include <cuda_fp16.h>
#include "reconstruct.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"

// TODO: Benchmark, profile, unit test

template <int K, int cb>
__global__ __launch_bounds__(256)
void reconstruct_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed
)
{
    constexpr int packed_size = 256 * K / 16;  // in uint16s

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int k = blockIdx.y;
    int n = blockIdx.x * 8;
    int tiles_n = gridDim.x;
    int blocks_n = tiles_n * 8;

    // Load packed 16*128 tile
    __shared__ uint32_t s_packed[8][packed_size / 2];
    g_packed += (k * blocks_n + n) * packed_size;
    for (int s = t; s < packed_size * 8 / 8; s += 256)
        ((int4*) s_packed)[t] = ((int4*) g_packed)[t];
    __syncthreads();

    // Dequant
    register FragB frag[2];
    dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    // Shuffle from tensor core layout to row major tile
//    __shared__ half tile[16 * 8 * 16];
    __shared__ half2 tile[16][8][8];

    half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
    half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
    half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
    half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);
    __syncwarp();

    if (!(lane_id & 4))
    {
        half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
        half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
        half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
        half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
        half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
        half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
        half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
        half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
        int r0 = (lane_id % 4) * 2;
        int r1 = r0 + 1;
        int r2 = r0 + 8;
        int r3 = r0 + 9;
        int c0 = lane_id / 8;
        int c1 = c0 + 4;
        tile[r0][warp_id][c0] = m0;
        tile[r1][warp_id][c0] = m1;
        tile[r2][warp_id][c0] = m2;
        tile[r3][warp_id][c0] = m3;
        tile[r0][warp_id][c1] = m4;
        tile[r1][warp_id][c1] = m5;
        tile[r2][warp_id][c1] = m6;
        tile[r3][warp_id][c1] = m7;
    }
    __syncthreads();

    // Store unpacked tile
    int r = t / 16;
    int c = t % 16;
    int4* tile_int4 = (reinterpret_cast<int4*> (tile));
    int4* out_int4 = ((int4*) g_unpacked) + (k * 16 + r) * 2 * blocks_n + n * 2 + c;
    *out_int4 = tile_int4[t];
}

#define __(i, cb) reconstruct_kernel<i, cb>
constexpr auto reconstruct_kernel_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __

/*
Reconstruct encoded+packed tensor
*/
void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(unpacked, 0, packed, 0, 16);
    TORCH_CHECK_SHAPES(unpacked, 1, packed, 1, 16);
    TORCH_CHECK_SIZE(packed, 2, 256 * K / 16);
    TORCH_CHECK_DTYPE(unpacked, kHalf);

    int rows = packed.size(0);
    int cols = packed.size(1);

    dim3 blockDim(256);
    dim3 gridDim(cols / 8, rows);

    int cbi = K - 1;
    if (mcg) cbi += 8;
    else if (mul1) cbi += 16;

    reconstruct_kernel_instances[cbi]<<<gridDim, blockDim, 0, stream>>>
    (
        (half*) unpacked.data_ptr(),
        (const uint16_t*) packed.data_ptr()
    );
    cuda_check(cudaPeekAtLastError());
}
