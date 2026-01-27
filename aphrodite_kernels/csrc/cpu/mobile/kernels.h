#ifndef APHRODITE_MOBILE_KERNELS_H
#define APHRODITE_MOBILE_KERNELS_H

#include <cstddef>
#include <cstdint>

namespace aphrodite::mobile {

// Quantization kernels
void int8_to_fp32(const int8_t* src, float* dst, size_t count, float scale);
void fp32_to_int8(const float* src, int8_t* dst, size_t count, float scale);
void dynamic_quantize_fp32_to_int8(const float* src, int8_t* dst, size_t count,
                                   float* computed_scale);
void fp16_to_fp32(const __fp16* src, float* dst, size_t count);
void fp32_to_fp16(const float* src, __fp16* dst, size_t count);
void int8_to_fp16(const int8_t* src, __fp16* dst, size_t count, float scale);
void fp16_to_int8(const __fp16* src, int8_t* dst, size_t count, float scale);
float fp16_max_abs(const __fp16* src, size_t count);
void int32_to_fp16_scaled(const int32_t* src, __fp16* dst, size_t count,
                          float scale);

// BLAS kernels - INT8
void add_int8(const int8_t* a, const int8_t* b, int8_t* output,
              size_t num_elements);
void subtract_int8(const int8_t* a, const int8_t* b, int8_t* output,
                   size_t num_elements);
void multiply_int8(const int8_t* a, const int8_t* b, int8_t* output,
                   size_t num_elements);
void divide_int8(const int8_t* a, const int8_t* b, int8_t* output,
                 size_t num_elements);

// BLAS kernels - FP32
void add_f32(const float* a, const float* b, float* output,
             size_t num_elements);
void subtract_f32(const float* a, const float* b, float* output,
                  size_t num_elements);
void multiply_f32(const float* a, const float* b, float* output,
                  size_t num_elements);
void divide_f32(const float* a, const float* b, float* output,
                size_t num_elements);

// Matrix multiplication (GEMM)
void matmul_int8(const int8_t* a, const int8_t* b_transposed, int8_t* c,
                 size_t M, size_t K, size_t N, float a_scale, float b_scale,
                 float c_scale);
void matmul_f16(const __fp16* a, const __fp16* b_transposed, __fp16* c,
                size_t M, size_t K, size_t N);
void matmul_f32(const float* a, const float* b_transposed, float* c, size_t M,
                size_t K, size_t N);
void matmul_int8_to_int32(const int8_t* a, const int8_t* b_transposed,
                          int32_t* c, size_t M, size_t K, size_t N);
#if defined(__ARM_FEATURE_MATMUL_INT8)
void matmul_int8_to_int32_i8mm(const int8_t* a, const int8_t* b_transposed,
                               int32_t* c, size_t M, size_t K, size_t N);
#endif

// Reduction kernels
int64_t sum_all_int8(const int8_t* data, size_t num_elements);
void sum_axis_int8(const int8_t* input, int8_t* output, size_t outer_size,
                   size_t axis_size, size_t inner_size);
double mean_all_int8(const int8_t* data, size_t num_elements);
void mean_axis_int8(const int8_t* input, int8_t* output, size_t outer_size,
                    size_t axis_size, size_t inner_size);
double mean_all_f16(const __fp16* data, size_t num_elements);
void mean_axis_f16(const __fp16* input, __fp16* output, size_t outer_size,
                   size_t axis_size, size_t inner_size);
double variance_all_int8(const int8_t* data, size_t num_elements);
void variance_axis_int8(const int8_t* input, int8_t* output, size_t outer_size,
                        size_t axis_size, size_t inner_size);
int64_t min_all_int8(const int8_t* data, size_t num_elements);
void min_axis_int8(const int8_t* input, int8_t* output, size_t outer_size,
                   size_t axis_size, size_t inner_size);
int64_t max_all_int8(const int8_t* data, size_t num_elements);
void max_axis_int8(const int8_t* input, int8_t* output, size_t outer_size,
                   size_t axis_size, size_t inner_size);
double sum_all_f32(const float* data, size_t num_elements);
void sum_axis_f32(const float* input, float* output, size_t outer_size,
                  size_t axis_size, size_t inner_size);

// Scalar operation kernels
enum class ScalarOpType {
  ADD,
  SUBTRACT,
  MULTIPLY,
  DIVIDE,
  EXP,
  SQRT,
  COS,
  SIN
};
void scalar_op_int8(const int8_t* input, int8_t* output, size_t num_elements,
                    float scalar_value, ScalarOpType op_type);
void scalar_op_f16(const __fp16* input, __fp16* output, size_t num_elements,
                   float scalar_value, ScalarOpType op_type);
void scalar_op_f32(const float* input, float* output, size_t num_elements,
                   float scalar_value, ScalarOpType op_type);

// Neural network kernels
void silu_f32(const float* input, float* output, size_t num_elements);
void silu_f16(const __fp16* input, __fp16* output, size_t num_elements);
void silu_int8(const int8_t* input, int8_t* output, size_t num_elements,
               float input_scale, float output_scale);
void gelu_f32(const float* input, float* output, size_t num_elements);
void gelu_f16(const __fp16* input, __fp16* output, size_t num_elements);
void gelu_int8(const int8_t* input, int8_t* output, size_t num_elements,
               float input_scale, float output_scale);
void softmax_f32(const float* input, float* output, size_t batch_size,
                 size_t seq_len, size_t vocab_size);
void softmax_f16(const __fp16* input, __fp16* output, size_t batch_size,
                 size_t seq_len, size_t vocab_size);

// Attention kernels
void attention_int8(const int8_t* queries, const int8_t* keys,
                    const int8_t* values, int8_t* output, size_t batch_size,
                    size_t seq_len, size_t kv_seq_len, size_t num_q_heads,
                    size_t num_kv_heads, size_t head_dim, float scale,
                    const int8_t* mask, float q_scale, float k_scale,
                    float v_scale, float output_scale,
                    size_t position_offset = 0, size_t window_size = 0,
                    bool is_causal = true);
void attention_f16(const __fp16* queries, const __fp16* keys,
                   const __fp16* values, __fp16* output, size_t batch_size,
                   size_t seq_len, size_t kv_seq_len, size_t num_q_heads,
                   size_t num_kv_heads, size_t head_dim, float scale,
                   const __fp16* mask, size_t position_offset = 0,
                   size_t window_size = 0, bool is_causal = true);
void attention_f32(const float* queries, const float* keys, const float* values,
                   float* output, size_t batch_size, size_t seq_len,
                   size_t kv_seq_len, size_t num_q_heads, size_t num_kv_heads,
                   size_t head_dim, float scale, const float* mask,
                   size_t position_offset = 0, size_t window_size = 0,
                   bool is_causal = true);

// Normalization kernels
void rms_norm_f32(const float* input, const float* weight, float* output,
                  size_t batch_size, size_t dims, float eps);
void rms_norm_f16(const __fp16* input, const __fp16* weight, __fp16* output,
                  size_t batch_size, size_t dims, float eps);
void rms_norm_i8_f32(const int8_t* input, const float* weight, float* output,
                     size_t batch_size, size_t dims, float eps,
                     float input_scale);

// RoPE kernels
void rope_f32(const float* input, float* output, size_t batch_size,
              size_t seq_len, size_t num_heads, size_t head_dim,
              size_t start_pos, float theta);
void rope_f16(const __fp16* input, __fp16* output, size_t batch_size,
              size_t seq_len, size_t num_heads, size_t head_dim,
              size_t start_pos, float theta);
void rope_i8_f32_i8(const int8_t* input, int8_t* output, size_t batch_size,
                    size_t seq_len, size_t num_heads, size_t head_dim,
                    size_t start_pos, float theta, float input_scale,
                    float output_scale);

}  // namespace aphrodite::mobile

#endif  // APHRODITE_MOBILE_KERNELS_H
