#pragma once

#include "../ptx.cuh"

// Constants
#define EXL3_GEMM_BASE_THREADS 256
#define SMEM_MAX (90 * 1024)  // max shared memory on compute capability 8.6

#include "exl3_dq.cuh"

template<EXL3_GEMM_T_ARGS>
inline __device__
void exl3_gemm_kernel_inner
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks
)
{
    const int TILEBLOCKS_M = TILESIZE_M / 16;
    const int TILEBLOCKS_K = TILESIZE_K / 16;
    const int TILEBLOCKS_N = TILESIZE_N / 16;
    const int FRAGS_M = TILEBLOCKS_M;
    const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);

    const int sh_a_stage_size = TILESIZE_M * TILESIZE_K;                         // in halfs
    const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;   // in uint16s
    const int sh_c_size = 4 * EXL3_GEMM_BASE_THREADS;                            // in floats

    // XOR-swizzle constants for bank-conflict-free A fragment loads
    // col_swizzled = col ^ ((row >> SHIFT) & MASK)
    const int A_COLS = TILESIZE_K / 8;                                            // int4 columns per row
    const int A_SWIZZLE_MASK = A_COLS - 1;
    const int A_SWIZZLE_SHIFT = (A_COLS <= 2) ? 2 : 1;

    // Sanity checks
    static_assert(EXL3_GEMM_BASE_THREADS == 256);
    static_assert(TILESIZE_M % 16 == 0, "Invalid kernel params");
    static_assert(TILESIZE_K % 16 == 0, "Invalid kernel params");
    static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
    static_assert
    (
        SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
        "Invalid kernel params (insufficient shared memory for shape)"
    );

    // Shared memory
    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage_size);
    float* sh_c = (float*) (sh_b + sh_b_stage_size * SH_STAGES);

    // Thread index
    int t = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    int sub_k = threadIdx.x / EXL3_GEMM_BASE_THREADS;
    int warp_id = t / 32;
    int lane_id = t % 32;

    // Dimensions
    //int tiles_m = CEIL_DIVIDE(size_m, TILESIZE_M);
    int tiles_k = size_k / TILESIZE_K;
    int tiles_n = size_n / TILESIZE_N;
    //int blocks_m = 1;
    //int blocks_k = tiles_k * TILEBLOCKS_K;
    int blocks_n = tiles_n * TILEBLOCKS_N;

    // Start and end index of current slice, must span at least one tile
    int num_slices = gridDim.x;
    int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;
    int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
    int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    auto index_m = [&] (int slice_i) { return 0; }; //blockIdx.y; };
    auto index_k = [&] (int slice_i) { return (slice_i % tiles_k); };
    auto index_n = [&] (int slice_i) { return (slice_i / tiles_k); };

    // Batch dimension
    int slice_m = index_m(slice_beg);
    int max_m = MIN(size_m - slice_m * TILESIZE_M, TILESIZE_M);

    // Pipe 0, global A, B tile and shared A, B tile
    int slice0_k = index_k(slice_beg);
    int slice0_n = index_n(slice_beg);
    int slice0_iters = slice_len;

    int gl_a_stride_m = TILESIZE_M * size_k;
    const int gl_a_stride_k = TILESIZE_K;
    const int sh0_a_stride_m = TILESIZE_M * TILESIZE_K;
    const half* gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
    half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

    const int load_a_iters = CEIL_DIVIDE(sh0_a_stride_m / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_a_gl[load_a_iters];
    int load_a_gl[load_a_iters];
    int load_a_sh[load_a_iters];
    for (int i = 0; i < load_a_iters; ++i)
    {
        int k = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_a_stride_k / 8);
        int m = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_a_stride_k / 8);
        load_a_gl[i] = m * size_k / 8 + k;
        load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
        pred_a_gl[i] = m < max_m;
    }

    int gl_b_stride_k = blocks_n * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    const int load_b_iters = CEIL_DIVIDE(sh0_b_stride_k / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_b_gl[load_b_iters];
    int load_b_gl[load_b_iters];
    for (int i = 0; i < load_b_iters; ++i)
    {
        int n = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n * 256 / 16 * bits / 8) + n;
        pred_b_gl[i] = i * EXL3_GEMM_BASE_THREADS + t < sh0_b_stride_k / 8;
    }

    auto advance0 = [&] ()
    {
        slice0_k++;
        slice0_iters--;

        int stage = slice0_iters % SH_STAGES;
        sh0_a_ptr = sh_a + stage * sh_a_stage_size;
        sh0_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice0_k >= tiles_k)
        {
            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
        }
        else
        {
            gl_a_ptr += gl_a_stride_k;
            gl_b_ptr += gl_b_stride_k;
        }
    };

    // Pipe 1, shared A, B tile and registers
    int slice1_k = slice0_k;
    int slice1_n = slice0_n;
    int slice1_iters = slice0_iters;

    half* sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t* sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

    auto advance1 = [&] ()
    {
        slice1_k++;
        slice1_iters--;

        int stage = slice1_iters % SH_STAGES;
        sh1_a_ptr = sh_a + stage * sh_a_stage_size;
        sh1_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice1_k >= tiles_k)
        {
            slice1_k = 0;
            slice1_n++;
        }
    };

    // Pipe 2
    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;

    int gl_c_stride_n = TILESIZE_N;
    int gl_c_stride_m = TILESIZE_M * size_n;

    half* gl_c_ptr_16 = ((half*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;
    float* gl_c_ptr_32 = ((float*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;

    register FragA frag_a[FRAG_STAGES][FRAGS_M];
    register FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    register FragC frag_c[FRAGS_M][FRAGS_N_PER_WARP];

    auto advance2 = [&] ()
    {
        slice2_k++;
        slice2_iters--;

        if (slice2_k >= tiles_k)
        {
            slice2_k = 0;
            slice2_k0 = 0;
            slice2_n++;
            if constexpr (c_fp32)
                gl_c_ptr_32 += gl_c_stride_n;
            else
                gl_c_ptr_16 += gl_c_stride_n;
        }
    };

    // Schedule load of the next A, B tiles to shared memory and advance the pipeline
    auto async_load_gl = [&] ()
    {
        if (sub_k)
        {
            cp_async_fence();
            return;
        }

        if (slice0_iters)
        {
            // Copy tile from row-major A matrix (XOR-swizzled for bank-conflict-free ldmatrix)
            {
                const int4* gl = (const int4*) gl_a_ptr;
                int4* sh = (int4*) sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i)
                {
                    if (pred_a_gl[i]) cp_async(sh + load_a_sh[i], gl + load_a_gl[i]);
                }
            }

            // Copy tile of 256-element blocks from quantized B matrix
            {
                const int4* gl = (const int4*) gl_b_ptr;
                int4* sh = (int4*) sh0_b_ptr;
                #pragma unroll
                for (int i = 0; i < load_b_iters; ++i)
                {
                    // cp_async_pred(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i], pred_b_gl[i]);
                    if (pred_b_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i]);
                }
            }
            advance0();
        }

        // Sync and advance
        cp_async_fence();
    };

    // Load fragments
    // Ref. for fragment layout:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    auto load_frags = [&] (int buf)
    {
        if (!slice1_iters) return;

        // A fragments (XOR-swizzled shared memory layout)
        {
            int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
            int base_c = lane_id / 16 + sub_k * 2;
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
            {
                int R = r + m * 16;
                int c_swizzled = base_c ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                ldsm4(frag_a[buf][m], (int4*) sh1_a_ptr + R * A_COLS + c_swizzled);
            }
        }

        // B fragments
        #pragma unroll
        for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2)
        {
            int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
            const uint32_t* shb = (const uint32_t*) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);

            dq_dispatch<bits, cb>(shb, lane_id << 3, frag_b[buf][n2], frag_b[buf][n2 + 1]);
        }

        __syncthreads();
        advance1();
    };

    // Clear C fragments
    auto clear_frag_c = [&] ()
    {
        #pragma unroll
        for (int m = 0; m < FRAGS_M; ++m)
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                frag_c[m][n] = {};
    };

    // Threadblock reduction
    auto threadblock_reduce = [&] ()
    {
        auto store = [&] (int i, int m, int n)
        {
            // TODO: Shuffle to avoid bank conflicts here? Doesn't seem to be a bottleneck
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                if (size_m <= 8)
                {
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) *sh_red++ = frag_c[m][n][i];
                }
                else
                {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) *sh_red++ = frag_c[m][n][i];
                }
            }
            __syncthreads();
        };

        auto add = [&] (int i, int m, int n)
        {
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                if (size_m <= 8)
                {
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) frag_c[m][n][i] += *sh_red++;
                }
                else
                {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) frag_c[m][n][i] += *sh_red++;
                }
            }
            __syncthreads();
        };

        #pragma unroll
        for (int m = 0; m < FRAGS_M; ++m)
        {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                if constexpr (TILEBLOCKS_K == 2)
                {
                    store(1, m, n);
                    add(0, m, n);
                }
                if constexpr (TILEBLOCKS_K == 3)
                {
                    store(1, m, n);
                    add(0, m, n);
                    store(2, m, n);
                    add(0, m, n);
                }
                if constexpr (TILEBLOCKS_K == 4)
                {
                    store(3, m, n);
                    add(2, m, n);
                    store(1, m, n);
                    add(0, m, n);
                    store(2, m, n);
                    add(0, m, n);
                }
            }
        }
    };

    // Output reduction
    auto reduce = [&] ()
    {
        // First reduce all partial sums along k for the current slice
        threadblock_reduce();

        // Process (partial) slices within column in reverse order so the threadblock doing the bottom slice is
        // free to proceed to the next column right away
        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice_m * blocks_n + slice2_n];

        barrier_acquire(lock, lock_i);

        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles_k;

        int n0 = warp_id * FRAGS_N_PER_WARP;

        // Second and subsequent threadblocks in column read back the intermediate sum from global memory
        if (!sub_k && !first)
        {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                #pragma unroll
                for (int m = 0; m < FRAGS_M; ++m)
                {
                    int r0 = lane_id / 4 + 16 * m;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                            frag_c[m][n][0] += *c_ptr++;
                            frag_c[m][n][1] += *c_ptr++;
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                            float2 interm = __half22float2(*c_ptr);
                            frag_c[m][n][0] += interm.x;
                            frag_c[m][n][1] += interm.y;
                        }
                    }
                    if (r1 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                            frag_c[m][n][2] += *c_ptr++;
                            frag_c[m][n][3] += *c_ptr++;
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                            float2 interm = __half22float2(*c_ptr);
                            frag_c[m][n][2] += interm.x;
                            frag_c[m][n][3] += interm.y;
                        }
                    }
                }
            }
        }

        // All but last threadblock in column write the intermediate result to global memory
        if (!sub_k && !last)
        {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                #pragma unroll
                for (int m = 0; m < FRAGS_M; ++m)
                {
                    int r0 = lane_id / 4 + 16 * m;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][0];
                            *c_ptr++ = frag_c[m][n][1];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
                            *c_ptr = sum;
                        }
                    }
                    if (r1 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][2];
                            *c_ptr++ = frag_c[m][n][3];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
                            *c_ptr = sum;
                        }
                    }
                }
            }
        }

        // Last block writes in row-major format
        if (!sub_k && last)
        {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                #pragma unroll
                for (int m = 0; m < FRAGS_M; ++m)
                {
                    int r0 = lane_id / 4 + 16 * m;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][0];
                            *c_ptr++ = frag_c[m][n][1];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
                            *c_ptr = sum;
                        }
                    }
                    if (r1 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][2];
                            *c_ptr++ = frag_c[m][n][3];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
                            *c_ptr = sum;
                        }
                    }
                }
            }
        }

        barrier_release(lock, lock_d, last);

        clear_frag_c();
    };

    // Wait until there are at most SH_STAGES - 2 async copies pending, i.e. at least one stage has finished loading
    auto wait_stage = [&] ()
    {
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
    };

    // Perform tensor core matmul on current tile
    auto matmul = [&] (int buf)
    {
        #pragma unroll
        for (int m = 0; m < FRAGS_M; ++m)
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                ptx_mma_m16n8k16(frag_a[buf][m], frag_b[buf][n], frag_c[m][n]);
    };

    // Start global to shared pipeline
    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i)
        async_load_gl();
    wait_stage();

    // Start shared to register pipeline.
    clear_frag_c();
    if constexpr (FRAG_STAGES > 1)
        load_frags(0);

    // Main loop. Fragments are double buffered to allow more interleaving. This is especially important to hide the
    // dequantization overhead, but we need two different iterations of the main loop to avoid confusing the compiler
    // and making it (sometimes) place the fragment arrays in local memory

    #define FSTAGE(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        load_frags(_load); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \

    if constexpr (FRAG_STAGES == 1)
    {
        while (true)
        {
            FSTAGE(0, 0);
        }
    }

    if constexpr (FRAG_STAGES == 2)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(0, 1);
        }
    }

    if constexpr (FRAG_STAGES == 3)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(0, 2);
        }
    }

    if constexpr (FRAG_STAGES == 4)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(0, 3);
        }
    }

    if constexpr (FRAG_STAGES == 5)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(4, 3);
            FSTAGE(0, 4);
        }
    }
}
