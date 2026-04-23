#pragma once

#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner.cuh"

template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_gemm_kernel(EXL3_GEMM_ARGS)
{
    auto grid = cg::this_grid();

    if (suh)
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                nullptr,
                0.088388347648f  // 1/sqrt(128)
            );

        grid.sync();
        A = A_had;
    }

    int size_m_ = size_m;
    const half* A_ = A;
    void* C_ = C;

    while (size_m_ > 0)
    {
        exl3_gemm_kernel_inner
        <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES>
        (A_, B, C_, size_m_, size_k, size_n, locks);

        A_ += 16 * size_k;
        if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * size_n);
        else                  C_ = (void*) (((half*) C_) + 16 * size_n);
        size_m_ -= 16;

        if (size_m_ > 0 || svh)
            grid.sync();
    }

    if (svh)
    {
        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            if constexpr (c_fp32)
                had_ff_r_128_inner
                (
                    ((const float*) C) + this_warp * 128,
                    ((float*) C) + this_warp * 128,
                    nullptr,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
            else
                had_hf_r_128_inner
                (
                    ((const half*) C) + this_warp * 128,
                    ((half*) C) + this_warp * 128,
                    nullptr,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
        }
    }
}

#define MAX_INDICES 128

__device__ int64_t v_indices[128];
__device__ half v_weights[128];
__device__ int bszm_sync;

template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_mgemm_kernel(EXL3_MGEMM_ARGS)
{
    int bszm = MAX(bszm_in, bszm_out);
    auto grid = cg::this_grid();

    // Pack indices within min_index <= idx < max_index

    if (min_index >= 0)
    {
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        {
            int j = 0;
            for (int i = 0; i < bszm; ++i)
            {
                int idx = B_indices[i];
                if (idx >= min_index && idx < max_index)
                {
                    v_indices[j] = idx - min_index;
                    if (B_weights) v_weights[j] = B_weights[i];
                    j++;
                }
            }
            bszm_sync = j;
            for (; j < bszm; ++j)
            {
                v_indices[j] = -1;
            }
        }
        __threadfence();
        grid.sync();
        B_indices = v_indices;
        if (B_weights) B_weights = v_weights;
        bszm = bszm_sync;
    }

    for (int i = 0; i < bszm; i += gridDim.z)
    {
        int j = i + blockIdx.z;
        int mat_index = -1;
        const uint16_t* B = nullptr;
        if (j >= bszm) j = -1;
        else
        {
            mat_index = B_indices ? (int) B_indices[j] : j;
            if (mat_index >= 0)
            {
                B = B_list[mat_index];
            }
        }

        // Had and input scales

        if (B)
        {
            int total_warps = size_m * size_k / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            const half* suh = suh_list[mat_index];
            const half* A_ = bszm_in == 1 ? A : A + j * size_m * size_k;
            half* A_had_ = A_had + j * size_m * size_k;

            for(; this_warp < total_warps; this_warp += warps_grid)
                had_hf_r_128_inner
                (
                    A_ + this_warp * 128,
                    A_had_ + this_warp * 128,
                    suh + (this_warp * 128) % size_k,
                    nullptr,
                    0.088388347648f  // 1/sqrt(128)
                );
        }
        grid.sync();

        // Matmul

        int size_m_ = size_m;
        half* A_ = A_had + j * size_m * size_k;
        void* C_;
        if constexpr (c_fp32) C_ = (void*) (((float*) C) + j * size_m * size_n);
        else                  C_ = (void*) (((half*) C) + j * size_m * size_n);

        while (size_m_ > 0)
        {
            if (B)
            {
                int lock_offs = blockIdx.z * size_n / 128;

                exl3_gemm_kernel_inner
                <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES>
                (A_, B, C_, size_m_, size_k, size_n, locks + lock_offs);
             }

            A_ += 16 * size_k;
            if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * size_n);
            else                  C_ = (void*) (((half*) C_) + 16 * size_n);
            size_m_ -= 16;
            grid.sync();
        }

        // Had and output scales

        if (B)
        {
            int total_warps = size_m * size_n / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            const half* svh = svh_list[mat_index];
            float scale = 0.088388347648f;  // 1/sqrt(128)
            if (B_weights) scale *= __half2float(B_weights[j]);

            if constexpr (c_fp32) C_ = (void*) (((float*) C) + j * size_m * size_n);
            else                  C_ = (void*) (((half*) C) + j * size_m * size_n);

            for(; this_warp < total_warps; this_warp += warps_grid)
            {
                if constexpr (c_fp32)
                    had_ff_r_128_inner
                    (
                        ((const float*) C_) + this_warp * 128,
                        ((float*) C_) + this_warp * 128,
                        nullptr,
                        svh + (this_warp * 128) % size_n,
                        scale
                    );
                else
                    had_hf_r_128_inner
                    (
                        ((const half*) C_) + this_warp * 128,
                        ((half*) C_) + this_warp * 128,
                        nullptr,
                        svh + (this_warp * 128) % size_n,
                        scale
                    );
            }
        }
    }

    if (B_weights)
        grid.sync();

    // Final reduction
    if (B_weights && blockIdx.z == 0)
    {
        int total_warps = size_m * size_n / 32;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        int this_lane = threadIdx.x % 32;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            if constexpr (c_fp32)
            {
                float* C__ = ((float*) C) + this_warp * 32 + this_lane;
                float* C___ = C__;
                float sum = 0.0f;
                for (int j = 0; j < bszm; ++j)
                {
                    sum += *C___;
                    C___ += size_m * size_n;
                }
                *C__ = sum;
            }
            else
            {
                half* C__ = ((half*) C) + this_warp * 32 + this_lane;
                half* C___ = C__;
                half sum = {};
                for (int j = 0; j < bszm; ++j)
                {
                    sum = __hadd(sum, *C___);
                    C___ += size_m * size_n;
                }
                *C__ = sum;
            }
        }
    }
}