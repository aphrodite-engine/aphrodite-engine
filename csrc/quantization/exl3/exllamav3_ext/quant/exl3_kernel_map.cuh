#pragma once

int select_gemm_shape(int cc, int size_m, int size_k, int size_n, int bits, bool multi);
int exl3_gemm_num_kernel_shapes();
bool exl3_gemm_shape_compat(int shape_idx, int size_m, int size_k, int size_n, int bits);

#define EXL3_GEMM_T_ARGS \
    const int bits, \
    const bool c_fp32, \
    const int cb, \
    const int TILESIZE_M, \
    const int TILESIZE_K, \
    const int TILESIZE_N, \
    const int SH_STAGES, \
    const int FRAG_STAGES

#define EXL3_GEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t* __restrict__ B, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half* __restrict__ suh, \
    half* __restrict__ A_had, \
    const half* __restrict__ svh

#define EXL3_MGEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t** __restrict__ B_list, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half** __restrict__ suh_list, \
    half* __restrict__ A_had, \
    const half** __restrict__ svh_list, \
    int64_t* B_indices, \
    half* B_weights, \
    const int bszm_in, \
    const int bszm_out, \
    const int min_index, \
    const int max_index

typedef void (*fp_exl3_gemm_kernel) (EXL3_GEMM_ARGS);
typedef void (*fp_exl3_mgemm_kernel) (EXL3_MGEMM_ARGS);

#define EXL3_GEMM_SHAPE_1     16,     16,    128,     6,     5
#define EXL3_GEMM_SHAPE_2     16,     32,    128,     4,     3
#define EXL3_GEMM_SHAPE_3     16,     32,    256,     4,     3
#define EXL3_GEMM_SHAPE_4     16,     16,    512,     4,     3

#define EXL3_GEMM_TILESIZE_K  0, 16, 32, 32, 16
#define EXL3_GEMM_TILESIZE_N  0, 128, 128, 256, 512
#define EXL3_GEMM_BLOCKDIM  0, 256, 512, 512, 256

#define EXL3_GEMM_NUM_SHAPES 4

// Shape 1 not currently used anywhere
#define EXL3_GEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb) \
    nullptr, \
    exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_1>, \
    exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_2>, \
    exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_3>, \
    exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_4>

#define EXL3_MGEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb) \
    nullptr, \
    exl3_mgemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_1>, \
    exl3_mgemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_2>, \
    exl3_mgemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_3>, \
    exl3_mgemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_4>

#define EXL3_GEMM_BASE_THREADS 256

#define ALL_EXL3_KERNEL_EXTERNS(K) \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b##K[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b##K[]; \

#define ALL_EXL3_KERNEL_INSTANCES(K) \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, true, 0), \
        EXL3_GEMM_KERNEL_INSTANCES(K, true, 1), \
        EXL3_GEMM_KERNEL_INSTANCES(K, true, 2) \
    }; \
    \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, 0), \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, 1), \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, 2) \
    }; \
    \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b##K[] = { \
        EXL3_MGEMM_KERNEL_INSTANCES(K, true, 0), \
        EXL3_MGEMM_KERNEL_INSTANCES(K, true, 1), \
        EXL3_MGEMM_KERNEL_INSTANCES(K, true, 2) \
    }; \
    \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b##K[] = { \
        EXL3_MGEMM_KERNEL_INSTANCES(K, false, 0), \
        EXL3_MGEMM_KERNEL_INSTANCES(K, false, 1), \
        EXL3_MGEMM_KERNEL_INSTANCES(K, false, 2) \
    };

fp_exl3_gemm_kernel select_exl3_gemm_kernel
(
    const int cc,
    const int size_m,
    const int size_k,
    const int size_n,
    const int bits,
    const bool c_fp32,
    const int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* out_num_sms,
    const int cb
);

fp_exl3_mgemm_kernel select_exl3_mgemm_kernel
(
    const int cc,
    const int size_m,
    const int size_k,
    const int size_n,
    const int K,
    const bool c_fp32,
    const int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* out_num_sms,
    const int cb,
    const int bszm_in,
    const int bszm_out
);

struct TSample {
    int cc;
    int K;
    int m;
    int k;
    int n;
    int shape_idx;
    int num_sms;
};

struct TMSample {
    int cc;
    int K;
    int m;
    int k;
    int n;
    int shape_idx;
    int num_sms;
    int bszm_in;
    int bszm_out;
};

struct TResult
{
    fp_exl3_gemm_kernel kernel;
    fp_exl3_mgemm_kernel mkernel;
    int shape_idx;
    int num_sms;
    int block_dim;
};

TResult* select_exl3_gemm_mgemm_kernel_new
(
    const int cc,
    const int size_m,
    const int size_k,
    const int size_n,
    const int K,
    const bool c_fp32,
    const int force_shape_idx,
    const int force_num_sms,
    const int cb
);
