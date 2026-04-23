#pragma once

#define TILESIZE_M 8
#define TILESIZE_K 512
#define TILESIZE_N 32

#include "../ptx.cuh"
#include "exl3_dq.cuh"

using FragA_m8 = Vec<half2, 2>;  // {a0, a1, a2, a3} aliased to {a0, a0, a1, a1}
using FragC_m8 = Vec<float, 3>;  // {c0, c1, c2, c3} aliased to {c0, c1, c2}

// frag_c (FP32, m16n16) += frag_a (FP16, m8k16) @ frag_b (FP16, k16n16)
// FP16 @ FP16 + FP32 -> FP32
__device__ inline void ptx_mma_m8n8k16
(
    const FragA_m8& frag_a,
    const FragB& frag_b,
    FragC_m8& frag_c
)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);

    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%2}, {%3,%3,%4,%4}, {%5,%6}, {%0,%1,%2,%3};\n"

        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2])   // Todo: try to alias into m8 fragment
        :  "r"(a[0]), "r"(a[1]),
           "r"(b[0]), "r"(b[1])
    );
}

__device__ inline float2 shfl_float2(float2 v, int src, int width)
{
    return make_float2(__shfl_sync(0xffffffff, v.x, src, width), __shfl_sync(0xffffffff, v.y, src, width));
}

template <int bits, bool c_fp32, int cb, int split_k>
__global__
__launch_bounds__(32 * TILESIZE_K / 16 * 1)
void exl3_gemv_kernel
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int __grid_constant__ size_m,
    const int __grid_constant__ size_k,
    const int __grid_constant__ size_n
//    const half* __restrict__ suh,
//    half* __restrict__ A_had,
//    const half* __restrict__ svh,
)
{
    const int num_stages = 2;

    // TB covers TILESIZE_M/K/N elements (M == 8, K and N are multiples of 16.)
    // block: (32, TILESIZE_N / 16, TILESIZE_K / 16)
    // grid: (size_k / TILESIZE_K, size_n / TILESIZE_N)
    const int ts = 32 * TILESIZE_K / 16 * TILESIZE_N / 16;
    int t = threadIdx.x + (threadIdx.y + threadIdx.z * blockDim.y) * blockDim.x;
    int lane_id = threadIdx.x; // & 31;

    //int k_slice = blockIdx.z;

    const int blocks_n = TILESIZE_N / 16;
    const int blocks_k = TILESIZE_K / 16;
    const int tiles_n = size_n / TILESIZE_N;
    const int tiles_k = size_k / TILESIZE_K;
    const int tiles_k_slice = tiles_k / split_k;

    const size_t str_A_gl_row = size_k * 2;                                 // A global: row stride, M
    const size_t str_A_gl_block_k = 32;                                     // A global: block stride, K
    const size_t str_A_gl_tile_k = TILESIZE_K * 2;                          // A global: tile stride, K

    const size_t str_B_gl_block_n = 256 * bits / 8;                         // B global: block stride, N
    const size_t str_B_gl_block_k = str_B_gl_block_n * (size_n / 16);       // B global: block stride, K
    const size_t str_B_gl_tile_n = str_B_gl_block_n * blocks_n;             // B global: tile stride, N
    const size_t str_B_gl_tile_k = str_B_gl_block_k * blocks_k;             // B global: tile stride, K

    const size_t str_B_sh_block_n = str_B_gl_block_n;                       // B shared: block stride, N
    const size_t str_B_sh_block_k = str_B_sh_block_n * blocks_n;            // B shared: block stride, K
    const size_t str_B_sh_tile_n = str_B_sh_block_k;                        // B shared: tile stride, N
    const size_t str_B_sh_tile = str_B_sh_block_k * blocks_k;               // B shared: tile size

    constexpr int c_esize = c_fp32 ? 4 : 2;
    const size_t str_C_gl_row = size_n * c_esize;                           // C global: row stride, M
    const size_t str_C_gl_block_n = 16 * c_esize;                           // C global: block stride, N
    const size_t str_C_gl_tile_n = TILESIZE_N * c_esize;                    // C global: tile stride, N

    //const int block_n = threadIdx.y;
    const int block_k = threadIdx.z;
    //const int tile_k = blockIdx.x;
    const int tile_n = blockIdx.y;

    uint8_t* A_gl_tile;
    uint8_t* B_gl_tile;
    uint8_t* C_gl_tile;
    uint8_t* A_gl_block;
    uint8_t* C_gl_block;
    uint8_t* B_sh_block;
    uint8_t* B_gl_block;

    // Tile buffer
    constexpr int bsh0 = num_stages ? num_stages : 1;
    constexpr int bsh1 = num_stages ? str_B_sh_tile : 32;
    __shared__ uint8_t B_sh_tile[bsh0][bsh1];
    __shared__ float C_sh_red[MAX(blocks_k / 2, 1)][blocks_n][128];

    // Fragments
    register FragA_m8 frag_a;
    register FragB frag_b[blocks_n][2];
    register FragC_m8 frag_c[blocks_n][2];
    #pragma unroll
    for (int block_n = 0; block_n < blocks_n; ++block_n)
    {
        frag_c[block_n][0] = {};
        frag_c[block_n][1] = {};
    }

    // Load A fragment from global (warp)
    auto load_gl2fr_a = [&]()
    {
        int frag_row = lane_id >> 2;
        int frag_col = lane_id & 3;
        if (frag_row < size_m)
        {
            half2* lda = (half2*)(A_gl_block + str_A_gl_row * frag_row);
            frag_a[0] = lda[frag_col];
            frag_a[1] = lda[frag_col + 4];
        }
    };

    // Load B shared tile from global (across warps)
    auto load_gl2sh_b = [&](int buffer, bool pred)
    {
        if (pred)
        {
            uint4* src = (uint4*) B_gl_tile;
            uint4* dst = (uint4*) B_sh_tile[buffer];
            #pragma unroll
            for (int i = t; i < str_B_sh_tile / 16; i += ts)
            {
                int x = i % (str_B_sh_tile_n / 16);
                int y = i / (str_B_sh_tile_n / 16);
                cp_async(dst + i, src + y * (str_B_gl_tile_n / 16 * tiles_n) + x);
            }
        }

        // Fence here to ensure number of fences matches number of waits
        cp_async_fence();
    };

    // Load B fragment from shared tile in warp
    auto load_sh2fr_b = [&](int block_n)
    {
        dq_dispatch<bits, cb>((uint32_t*) B_sh_block, lane_id << 3, frag_b[block_n][0], frag_b[block_n][1]);
    };

    // Load B fragment from gmem tile in warp
    auto load_gl2fr_b = [&](int block_n)
    {
        dq_dispatch<bits, cb>((uint32_t*) B_gl_block, lane_id << 3, frag_b[block_n][0], frag_b[block_n][1]);
    };

    // Perform matmul in current warp
    auto matmul = [&](int block_n)
    {
        ptx_mma_m8n8k16(frag_a, frag_b[block_n][0], frag_c[block_n][0]);
        ptx_mma_m8n8k16(frag_a, frag_b[block_n][1], frag_c[block_n][1]);
    };

    // Reduce C fragments across threadblock
    auto reduce_fr_c = [&]()
    {
        int frag_row = lane_id >> 2;
        //int frag_col = lane_id & 3;

        #pragma unroll
        for (int r_k = blocks_k / 2; r_k > 0; r_k >>= 1)
        {
            float* red;

            // High blocks store
            if (block_k < r_k * 2 && block_k >= r_k && frag_row < size_m)
            {
                #pragma unroll
                for (int block_n = 0; block_n < blocks_n; ++block_n)
                {
                    red = &C_sh_red[block_k & (r_k - 1)][block_n][lane_id * 4];
                    red[0] = frag_c[block_n][0][0];
                    red[1] = frag_c[block_n][0][1];
                    red[2] = frag_c[block_n][1][0];
                    red[3] = frag_c[block_n][1][1];
                }
            }
            __syncthreads();

            // Low blocks load
            if (block_k < r_k && frag_row < size_m)
            {
                #pragma unroll
                for (int block_n = 0; block_n < blocks_n; ++block_n)
                {
                    red = &C_sh_red[block_k & (r_k - 1)][block_n][lane_id * 4];
                    frag_c[block_n][0][0] += red[0];
                    frag_c[block_n][0][1] += red[1];
                    frag_c[block_n][1][0] += red[2];
                    frag_c[block_n][1][1] += red[3];
                }
            }

            if (r_k) __syncthreads();
        }
    };

    // Store fragment
    auto store_fr2gl_c = [&](int block_n)
    {
        int frag_row = lane_id >> 2;
        int frag_col = lane_id & 3;
        if (frag_row < size_m)
        {
            if (c_fp32)
            {
                float2* stc = (float2*)(C_gl_block + str_C_gl_row * frag_row);
                float2 c0 = make_float2(frag_c[block_n][0][0], frag_c[block_n][0][1]);
                float2 c1 = make_float2(frag_c[block_n][1][0], frag_c[block_n][1][1]);

                // Thread has c[i] and c[i+4], shuffle so threads hold c[i] and c[i+1] in groups of four
                int base = lane_id & (~3);
                int src0 = base + ((2 * frag_col) & 3);
                int src1 = base + ((2 * frag_col + 1) & 3);
                float2 c00 = shfl_float2(c0, src0, 4);
                float2 c01 = shfl_float2(c1, src0, 4);
                float2 c10 = shfl_float2(c0, src1, 4);
                float2 c11 = shfl_float2(c1, src1, 4);
                bool low = frag_col < 2;
                float2 s0 = low ? c00 : c01;
                float2 s1 = low ? c10 : c11;

                // Store 4x4 bytes per thread, coalesced
                float4 s01 = make_float4(s0.x, s0.y, s1.x, s1.y);
                ((float4*) stc)[frag_col] = s01;
            }
            else
            {
                half2* stc = (half2*)(C_gl_block + str_C_gl_row * frag_row);
                half2 c0 = __floats2half2_rn(frag_c[block_n][0][0], frag_c[block_n][0][1]);
                half2 c1 = __floats2half2_rn(frag_c[block_n][1][0], frag_c[block_n][1][1]);

                // Thread has c[i] and c[i+4], shuffle so threads hold c[i] and c[i+1] in groups of four
                int base = lane_id & (~3);
                int src0 = base + ((2 * frag_col) & 3);
                int src1 = base + ((2 * frag_col + 1) & 3);
                half2 c00 = __shfl_sync(0xffffffff, c0, src0, 4);
                half2 c01 = __shfl_sync(0xffffffff, c1, src0, 4);
                half2 c10 = __shfl_sync(0xffffffff, c0, src1, 4);
                half2 c11 = __shfl_sync(0xffffffff, c1, src1, 4);
                bool low = frag_col < 2;
                half2 s0 = low ? c00 : c01;
                half2 s1 = low ? c10 : c11;

                // Store 4x2 bytes per thread, coalesced
                half4 s01(s0, s1);
                ((half4*) stc)[frag_col] = s01;
            }
        }
    };

    // Direct
    if constexpr (num_stages == 0)
    {
        for (int tile_k = 0; tile_k < tiles_k_slice; ++tile_k)
        {
            // Load A frag
            A_gl_tile = ((uint8_t*) A) + str_A_gl_tile_k * tile_k;
            A_gl_block = A_gl_tile + str_A_gl_block_k * block_k;
            load_gl2fr_a();

            // Load B frag
            B_gl_tile = ((uint8_t*) B) + str_B_gl_tile_k * tile_k + str_B_gl_tile_n * tile_n;
            B_gl_block = B_gl_tile + str_B_gl_block_k * block_k;
            #pragma unroll
            for (int block_n = 0; block_n < blocks_n; ++block_n)
            {
                load_gl2fr_b(block_n);
                B_gl_block += str_B_gl_block_n;
            }

            // Mul
            for (int block_n = 0; block_n < blocks_n; ++block_n)
            {
                matmul(block_n);
            }
        }
    }

    // Staged
    else
    {
        // Load first B chunk
        B_gl_tile = ((uint8_t*) B) + str_B_gl_tile_n * tile_n;
        load_gl2sh_b(0, true);

        #pragma unroll
        for (int tile_k = 0; tile_k < tiles_k_slice; ++tile_k)
        {
            // Load next B chunk
            B_gl_tile = ((uint8_t*) B) + str_B_gl_tile_k * (tile_k + 1) + str_B_gl_tile_n * tile_n;
            load_gl2sh_b((tile_k + 1) % num_stages, tile_k < tiles_k_slice - 1);

            // Load A frag
            A_gl_tile = ((uint8_t*) A) + str_A_gl_tile_k * tile_k;
            A_gl_block = A_gl_tile + str_A_gl_block_k * block_k;
            load_gl2fr_a();

            // Wait until at most one stage is pending
            cp_async_wait<1>();
            __syncthreads();

            // Load B frag
            B_sh_block = B_sh_tile[tile_k % num_stages] + str_B_sh_block_k * block_k;
            #pragma unroll
            for (int block_n = 0; block_n < blocks_n; ++block_n)
            {
                load_sh2fr_b(block_n);
                B_sh_block += str_B_sh_block_n;
            }

            // Mul
            #pragma unroll
            for (int block_n = 0; block_n < blocks_n; ++block_n)
            {
                matmul(block_n);
            }
        }
    }

    // Reduce
    C_gl_tile = ((uint8_t*) C) + str_C_gl_tile_n * tile_n;
    //C_gl_block = C_gl_tile + str_C_gl_block_n * block_n;
    reduce_fr_c();

    // Store
    if (block_k == 0)
    {
        for (int block_n = 0; block_n < blocks_n; ++block_n)
        {
            C_gl_block = C_gl_tile + str_C_gl_block_n * block_n;
            store_fr2gl_c(block_n);
        }
    }
}