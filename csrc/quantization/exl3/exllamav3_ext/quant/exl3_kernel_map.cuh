#pragma once

int select_gemm_shape(int cc, int size_m, int size_k, int size_n, int bits,
                      bool multi);
int exl3_gemm_num_kernel_shapes();
bool exl3_gemm_shape_compat(int shape_idx, int size_m, int size_k, int size_n,
                            int bits);

#define EXL3_GEMM_T_ARGS                                                 \
  const int bits, const bool c_fp32, const int cb, const int TILESIZE_M, \
      const int TILESIZE_K, const int TILESIZE_N, const int SH_STAGES,   \
      const int FRAG_STAGES

#define EXL3_GEMM_ARGS                                                         \
  const half *__restrict__ A, const uint16_t *__restrict__ B,                  \
      void *__restrict__ C, const int size_m, const int size_k,                \
      const int size_n, int *__restrict__ locks, const half *__restrict__ suh, \
      half *__restrict__ A_had, const half *__restrict__ svh

#define EXL3_MGEMM_ARGS                                                        \
  const half *__restrict__ A, const uint16_t **__restrict__ B_list,            \
      void *__restrict__ C, const int size_m, const int size_k,                \
      const int size_n, int *__restrict__ locks,                               \
      const half **__restrict__ suh_list, half *__restrict__ A_had,            \
      const half **__restrict__ svh_list, int64_t *B_indices, half *B_weights, \
      const int bszm_in, const int bszm_out, const int min_index,              \
      const int max_index, const int num_tokens

typedef void (*fp_exl3_gemm_kernel)(EXL3_GEMM_ARGS);
typedef void (*fp_exl3_mgemm_kernel)(EXL3_MGEMM_ARGS);

#define EXL3_GEMM_SHAPE_1 16, 16, 128, 6, 5
#define EXL3_GEMM_SHAPE_2 16, 32, 128, 4, 3
#define EXL3_GEMM_SHAPE_3 16, 32, 256, 4, 3
#define EXL3_GEMM_SHAPE_4 16, 16, 512, 4, 3

#define EXL3_GEMM_TILESIZE_K 0, 16, 32, 32, 16
#define EXL3_GEMM_TILESIZE_N 0, 128, 128, 256, 512
#define EXL3_GEMM_BLOCKDIM 0, 256, 512, 512, 256

#define EXL3_GEMM_NUM_SHAPES 4

// Shape 1 not currently used anywhere
#define EXL3_GEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb)              \
  nullptr, exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_1>, \
      exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_2>,      \
      exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_3>,      \
      exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_4>

#define EXL3_MGEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb)              \
  nullptr, exl3_mgemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_1>, \
      exl3_mgemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_2>,      \
      exl3_mgemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_3>,      \
      exl3_mgemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_4>

#define EXL3_GEMM_BASE_THREADS 256

// Instance arrays are indexed by shape and defined per (K, cb) so each codebook
// compiles as a separate translation unit (see
// comp_units/exl3_comp_unit_K_cbX.cu)

#define EXL3_KERNEL_EXTERNS_CB(K, cb)                                     \
  extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K##_cb##cb[];   \
  extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K##_cb##cb[];   \
  extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b##K##_cb##cb[]; \
  extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b##K##_cb##cb[];

#define ALL_EXL3_KERNEL_EXTERNS(K) \
  EXL3_KERNEL_EXTERNS_CB(K, 0)     \
  EXL3_KERNEL_EXTERNS_CB(K, 1)     \
  EXL3_KERNEL_EXTERNS_CB(K, 2)

#define EXL3_KERNEL_INSTANCES_CB(K, cb)                               \
  fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K##_cb##cb[] = {   \
      EXL3_GEMM_KERNEL_INSTANCES(K, true, cb)};                       \
                                                                      \
  fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K##_cb##cb[] = {   \
      EXL3_GEMM_KERNEL_INSTANCES(K, false, cb)};                      \
                                                                      \
  fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b##K##_cb##cb[] = { \
      EXL3_MGEMM_KERNEL_INSTANCES(K, true, cb)};                      \
                                                                      \
  fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b##K##_cb##cb[] = { \
      EXL3_MGEMM_KERNEL_INSTANCES(K, false, cb)};

fp_exl3_gemm_kernel select_exl3_gemm_kernel(
    const int cc, const int size_m, const int size_k, const int size_n,
    const int bits, const bool c_fp32, const int force_shape_idx,
    int* out_block_dim, int* out_shape_idx, int* out_num_sms, const int cb);

fp_exl3_mgemm_kernel select_exl3_mgemm_kernel(
    const int cc, const int size_m, const int size_k, const int size_n,
    const int K, const bool c_fp32, const int force_shape_idx,
    int* out_block_dim, int* out_shape_idx, int* out_num_sms, const int cb,
    const int bszm_in, const int bszm_out);

fp_exl3_gemm_kernel get_gemm_kernel_ptr(int K, int shape_idx, bool c_fp32,
                                        int cb);
fp_exl3_mgemm_kernel get_mgemm_kernel_ptr(int K, int shape_idx, bool c_fp32,
                                          int cb);
