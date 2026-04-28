#include "ggml-impl.h"
#include "ggml.h"

#include <cstdlib>

float ggml_fp16_to_fp32(ggml_fp16_t x) {
  return GGML_COMPUTE_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(float x) {
  return GGML_COMPUTE_FP32_TO_FP16(x);
}

float ggml_bf16_to_fp32(ggml_bf16_t x) {
  return GGML_BF16_TO_FP32(x);
}

ggml_bf16_t ggml_fp32_to_bf16(float x) {
  return GGML_FP32_TO_BF16(x);
}

void ggml_fp16_to_fp32_row(const ggml_fp16_t* x, float* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    y[i] = ggml_fp16_to_fp32(x[i]);
  }
}

void ggml_fp32_to_fp16_row(const float* x, ggml_fp16_t* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    y[i] = ggml_fp32_to_fp16(x[i]);
  }
}

void ggml_bf16_to_fp32_row(const ggml_bf16_t* x, float* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    y[i] = ggml_bf16_to_fp32(x[i]);
  }
}

void ggml_fp32_to_bf16_row_ref(const float* x, ggml_bf16_t* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    y[i] = ggml_fp32_to_bf16(x[i]);
  }
}

void ggml_fp32_to_bf16_row(const float* x, ggml_bf16_t* y, int64_t n) {
  ggml_fp32_to_bf16_row_ref(x, y, n);
}

struct ggml_context* ggml_init(struct ggml_init_params /*params*/) {
  return reinterpret_cast<struct ggml_context*>(std::malloc(1));
}

void ggml_free(struct ggml_context* ctx) {
  std::free(ctx);
}
