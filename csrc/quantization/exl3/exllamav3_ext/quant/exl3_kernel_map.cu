#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include <tuple>
#include <mutex>
#include <map>
#include "exl3_kernel_map.cuh"
#include "exl3_devctx.cuh"
#include "comp_units/exl3_comp_unit_1.cuh"
#include "comp_units/exl3_comp_unit_2.cuh"
#include "comp_units/exl3_comp_unit_3.cuh"
#include "comp_units/exl3_comp_unit_4.cuh"
#include "comp_units/exl3_comp_unit_5.cuh"
#include "comp_units/exl3_comp_unit_6.cuh"
#include "comp_units/exl3_comp_unit_7.cuh"
#include "comp_units/exl3_comp_unit_8.cuh"

#include "exl3_kernel_map_samples.cuh"
std::map<uint64_t, TResult> _tuning_cache = {};

int select_gemm_shape(int cc, int size_m, int size_k, int size_n, int K, bool multi, int bszm_in, int bszm_out)
{
    bool mod_256 = (size_n % 256 == 0);
    bool mod_512 = (size_n % 512 == 0);

    size_k *= bszm_in;
    size_n *= bszm_out;

    switch(cc)
    {
        case CC_OLD:
        case CC_AMPERE:
            if (mod_256 && K <= 4)
            {
                if (size_n <= 2048 || size_k <= 2048) return 2;
                return 3;
            }
            if (mod_256 && size_n < 4096) return size_k > 8192 ? 3 : 2;
            if (mod_512 && (size_n * size_k) > (4096 * 4096) && K <= 6) return 4;
            if (mod_256) return 3;
            return 2;

        case CC_ADA:
            if (mod_256 && K <= 3)
            {
                if (size_k <= 2048 && !multi) return 2;
                if (size_n < 4096 && size_k <= 12288) return 2;
                return 3;
            }
            if (size_n <= 16384) return 2;
            if (mod_512 && size_n >= 32768) return 4;
            if (mod_256) return 3;
            return 2;

        // case CC_HOPPER:
        case CC_BLACKWELL:
            if ((K == 4 || K == 2) && !multi)
            {
                if (size_k <= 2048) return 1;
            }
            if (K >= 7)
            {
                if (mod_256 && size_n <= 8192) return size_k > 32768 ? 3 : 2;
                if (mod_512 && size_n > 32768) return 4;
                return 2;
            }
            if (mod_256 && size_n <= 4096) return size_k > 8192 && K >= 3 ? 3 : 2;
            if (mod_512 && size_n > 16384) return 4;
            if (mod_256) return 3;
            return 2;
    }
    return 0;
}

int exl3_gemm_num_kernel_shapes()
{
    return EXL3_GEMM_NUM_SHAPES;
}

int exl3_gemm_tilesize_k[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim[] = {EXL3_GEMM_BLOCKDIM};

bool exl3_gemm_shape_compat(int shape_idx, int size_m, int size_k, int size_n, int K)
{
    int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
    int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
    return (size_k % tilesize_k == 0) && (size_n % tilesize_n == 0);
}

fp_exl3_gemm_kernel select_exl3_gemm_kernel
(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* num_sms,
    int cb
)
{
    int shape_idx = force_shape_idx <= 0 ? select_gemm_shape(cc, size_m, size_k, size_n, K, false, 1, 1) : force_shape_idx;

    TORCH_CHECK(shape_idx > 0, "exl3_gemm: no compatible kernel");
    if (out_shape_idx) *out_shape_idx = shape_idx;
    if (out_block_dim) *out_block_dim = exl3_gemm_blockdim[shape_idx];

    // Avoid empty blocks
    if (num_sms)
    {
        int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
        int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
        int max_slices = size_k / tilesize_k * size_n / tilesize_n;
        *num_sms = MAX(MIN(max_slices, *num_sms), 1);
    }

    int kernel_idx = shape_idx + (EXL3_GEMM_NUM_SHAPES + 1) * cb;

    if (c_fp32)
    {
        switch (K)
        {
            case 1: return tfp_exl3_gemm_kernel_fp32_b1[kernel_idx];
            case 2: return tfp_exl3_gemm_kernel_fp32_b2[kernel_idx];
            case 3: return tfp_exl3_gemm_kernel_fp32_b3[kernel_idx];
            case 4: return tfp_exl3_gemm_kernel_fp32_b4[kernel_idx];
            case 5: return tfp_exl3_gemm_kernel_fp32_b5[kernel_idx];
            case 6: return tfp_exl3_gemm_kernel_fp32_b6[kernel_idx];
            case 7: return tfp_exl3_gemm_kernel_fp32_b7[kernel_idx];
            case 8: return tfp_exl3_gemm_kernel_fp32_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }
    else
    {
        switch (K)
        {
            case 1: return tfp_exl3_gemm_kernel_fp16_b1[kernel_idx];
            case 2: return tfp_exl3_gemm_kernel_fp16_b2[kernel_idx];
            case 3: return tfp_exl3_gemm_kernel_fp16_b3[kernel_idx];
            case 4: return tfp_exl3_gemm_kernel_fp16_b4[kernel_idx];
            case 5: return tfp_exl3_gemm_kernel_fp16_b5[kernel_idx];
            case 6: return tfp_exl3_gemm_kernel_fp16_b6[kernel_idx];
            case 7: return tfp_exl3_gemm_kernel_fp16_b7[kernel_idx];
            case 8: return tfp_exl3_gemm_kernel_fp16_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }
}

fp_exl3_mgemm_kernel select_exl3_mgemm_kernel
(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* num_sms,
    int cb,
    int bszm_in,
    int bszm_out
)
{
    int shape_idx = force_shape_idx <= 0 ? select_gemm_shape(cc, size_m, size_k, size_n, K, true, bszm_in, bszm_out) : force_shape_idx;
    TORCH_CHECK(shape_idx > 0, "exl3_mgemm: no compatible kernel");
    if (out_shape_idx) *out_shape_idx = shape_idx;
    if (out_block_dim) *out_block_dim = exl3_gemm_blockdim[shape_idx];

    // Avoid empty blocks
    if (num_sms)
    {
        int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
        int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
        int max_slices = size_k / tilesize_k * size_n / tilesize_n / (*num_sms > 128 ? 20 : 24);
        *num_sms = MIN(max_slices, *num_sms);
    }

    int kernel_idx = shape_idx + (EXL3_GEMM_NUM_SHAPES + 1) * cb;

    if (c_fp32)
    {
        switch (K)
        {
            case 1: return tfp_exl3_mgemm_kernel_fp32_b1[kernel_idx];
            case 2: return tfp_exl3_mgemm_kernel_fp32_b2[kernel_idx];
            case 3: return tfp_exl3_mgemm_kernel_fp32_b3[kernel_idx];
            case 4: return tfp_exl3_mgemm_kernel_fp32_b4[kernel_idx];
            case 5: return tfp_exl3_mgemm_kernel_fp32_b5[kernel_idx];
            case 6: return tfp_exl3_mgemm_kernel_fp32_b6[kernel_idx];
            case 7: return tfp_exl3_mgemm_kernel_fp32_b7[kernel_idx];
            case 8: return tfp_exl3_mgemm_kernel_fp32_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }
    else
    {
        switch (K)
        {
            case 1: return tfp_exl3_mgemm_kernel_fp16_b1[kernel_idx];
            case 2: return tfp_exl3_mgemm_kernel_fp16_b2[kernel_idx];
            case 3: return tfp_exl3_mgemm_kernel_fp16_b3[kernel_idx];
            case 4: return tfp_exl3_mgemm_kernel_fp16_b4[kernel_idx];
            case 5: return tfp_exl3_mgemm_kernel_fp16_b5[kernel_idx];
            case 6: return tfp_exl3_mgemm_kernel_fp16_b6[kernel_idx];
            case 7: return tfp_exl3_mgemm_kernel_fp16_b7[kernel_idx];
            case 8: return tfp_exl3_mgemm_kernel_fp16_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }
}


fp_exl3_gemm_kernel get_gemm_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb)
{
    int kernel_idx = shape_idx + (EXL3_GEMM_NUM_SHAPES + 1) * cb;

    if (c_fp32)
    {
        switch (K)
        {
            case 1: return tfp_exl3_gemm_kernel_fp32_b1[kernel_idx];
            case 2: return tfp_exl3_gemm_kernel_fp32_b2[kernel_idx];
            case 3: return tfp_exl3_gemm_kernel_fp32_b3[kernel_idx];
            case 4: return tfp_exl3_gemm_kernel_fp32_b4[kernel_idx];
            case 5: return tfp_exl3_gemm_kernel_fp32_b5[kernel_idx];
            case 6: return tfp_exl3_gemm_kernel_fp32_b6[kernel_idx];
            case 7: return tfp_exl3_gemm_kernel_fp32_b7[kernel_idx];
            case 8: return tfp_exl3_gemm_kernel_fp32_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }
    else
    {
        switch (K)
        {
            case 1: return tfp_exl3_gemm_kernel_fp16_b1[kernel_idx];
            case 2: return tfp_exl3_gemm_kernel_fp16_b2[kernel_idx];
            case 3: return tfp_exl3_gemm_kernel_fp16_b3[kernel_idx];
            case 4: return tfp_exl3_gemm_kernel_fp16_b4[kernel_idx];
            case 5: return tfp_exl3_gemm_kernel_fp16_b5[kernel_idx];
            case 6: return tfp_exl3_gemm_kernel_fp16_b6[kernel_idx];
            case 7: return tfp_exl3_gemm_kernel_fp16_b7[kernel_idx];
            case 8: return tfp_exl3_gemm_kernel_fp16_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }
}


fp_exl3_mgemm_kernel get_mgemm_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb)
{
    int kernel_idx = shape_idx + (EXL3_GEMM_NUM_SHAPES + 1) * cb;

    if (c_fp32)
    {
        switch (K)
        {
            case 1: return tfp_exl3_mgemm_kernel_fp32_b1[kernel_idx];
            case 2: return tfp_exl3_mgemm_kernel_fp32_b2[kernel_idx];
            case 3: return tfp_exl3_mgemm_kernel_fp32_b3[kernel_idx];
            case 4: return tfp_exl3_mgemm_kernel_fp32_b4[kernel_idx];
            case 5: return tfp_exl3_mgemm_kernel_fp32_b5[kernel_idx];
            case 6: return tfp_exl3_mgemm_kernel_fp32_b6[kernel_idx];
            case 7: return tfp_exl3_mgemm_kernel_fp32_b7[kernel_idx];
            case 8: return tfp_exl3_mgemm_kernel_fp32_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }
    else
    {
        switch (K)
        {
            case 1: return tfp_exl3_mgemm_kernel_fp16_b1[kernel_idx];
            case 2: return tfp_exl3_mgemm_kernel_fp16_b2[kernel_idx];
            case 3: return tfp_exl3_mgemm_kernel_fp16_b3[kernel_idx];
            case 4: return tfp_exl3_mgemm_kernel_fp16_b4[kernel_idx];
            case 5: return tfp_exl3_mgemm_kernel_fp16_b5[kernel_idx];
            case 6: return tfp_exl3_mgemm_kernel_fp16_b6[kernel_idx];
            case 7: return tfp_exl3_mgemm_kernel_fp16_b7[kernel_idx];
            case 8: return tfp_exl3_mgemm_kernel_fp16_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }
}


TResult f_tr;

TResult* select_exl3_gemm_mgemm_kernel_new
(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int force_shape_idx,
    int force_num_sms,
    int cb
)
{
    // Force parameters for tuning/benchmarking
    if (force_shape_idx > 0)
    {
        TORCH_CHECK(force_num_sms, "Must supply force_shape_idx and force_num_sms together");
        f_tr.kernel = get_gemm_kernel_ptr(K, force_shape_idx, c_fp32, cb);
        f_tr.mkernel = get_mgemm_kernel_ptr(K, force_shape_idx, c_fp32, cb);
        f_tr.shape_idx = force_shape_idx;
        f_tr.num_sms = force_num_sms;
        f_tr.block_dim = exl3_gemm_blockdim[force_shape_idx];
        return &f_tr;
    };
    TORCH_CHECK(!force_num_sms, "Must supply force_shape_idx and force_num_sms together.");

    // Cache parameters
    uint64_t key = (((uint64_t) size_k) << 40) |
                   (((uint64_t) size_n) << 16) |
                   (((uint64_t) cc)     <<  8) |
                   (((uint64_t) K)      <<  4) |
                   (c_fp32 ? 0x01ull : 0x00ull);

    auto lookup = _tuning_cache.find(key);
    if (lookup == _tuning_cache.end())
    {
        // Find closest kernel in map
        bool mod512 = (size_n % 512 == 0);
        bool mod256 = (size_n % 256 == 0);
        bool mod128 = (size_n % 128 == 0);
        TORCH_CHECK(mod128, "size_n must be a multiple of 128");
        TSample* cand = mod512 ? samples_512 : (mod256 ? samples_256 : samples_128);
        TSample* best = nullptr;
        int64_t best_dist = 1ll<<62;

        for (; cand->K; cand++)
        {
            if (cand->K != K) continue;
            if (cand->cc != cc) continue;

            int64_t distk = (int64_t) (size_k - cand->k);
            int64_t distn = (int64_t) (size_n - cand->n);
            int64_t dist = distk * distk + distn * distn;
            if (dist < best_dist) { best_dist = dist; best = cand; }
        }
        TORCH_CHECK(best, "Failed to find valid kernel for shape");

        // Avoid empty blocks
        int tilesize_k = exl3_gemm_tilesize_k[best->shape_idx];
        int tilesize_n = exl3_gemm_tilesize_n[best->shape_idx];
        int max_slices = size_k / tilesize_k * size_n / tilesize_n;
        int num_sms = MAX(MIN(max_slices, best->num_sms), 1);

        // Results
        TResult tr = {
            get_gemm_kernel_ptr(K, best->shape_idx, c_fp32, cb),
            get_mgemm_kernel_ptr(K, best->shape_idx, c_fp32, cb),
            best->shape_idx,
            num_sms,
            exl3_gemm_blockdim[best->shape_idx]
        };

        _tuning_cache[key] = tr;
    }

    lookup = _tuning_cache.find(key);
    return &(lookup->second);
}