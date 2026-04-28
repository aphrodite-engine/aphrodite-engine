#pragma once

#include <cstddef>
#include <cstdint>

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

using ggml_half = uint16_t;
using ggml_fp16_t = uint16_t;

struct ggml_bf16_t {
  uint16_t bits;
};

enum ggml_type {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q8_K = 15,
  GGML_TYPE_BF16 = 30,
};

struct ggml_init_params {
  size_t mem_size;
  void* mem_buffer;
  bool no_alloc;
};

struct ggml_context;

float ggml_fp16_to_fp32(ggml_fp16_t x);
ggml_fp16_t ggml_fp32_to_fp16(float x);
float ggml_bf16_to_fp32(ggml_bf16_t x);
ggml_bf16_t ggml_fp32_to_bf16(float x);

void ggml_fp16_to_fp32_row(const ggml_fp16_t* x, float* y, int64_t n);
void ggml_fp32_to_fp16_row(const float* x, ggml_fp16_t* y, int64_t n);
void ggml_bf16_to_fp32_row(const ggml_bf16_t* x, float* y, int64_t n);
void ggml_fp32_to_bf16_row_ref(const float* x, ggml_bf16_t* y, int64_t n);
void ggml_fp32_to_bf16_row(const float* x, ggml_bf16_t* y, int64_t n);

ggml_context* ggml_init(ggml_init_params params);
void ggml_free(ggml_context* ctx);
