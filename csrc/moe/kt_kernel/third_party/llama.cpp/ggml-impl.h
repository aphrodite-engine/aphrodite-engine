#pragma once

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>

static inline float kt_ggml_fp32_from_bits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static inline uint32_t kt_ggml_fp32_to_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
  const uint32_t w = static_cast<uint32_t>(h) << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
  const float exp_scale = 0x1.0p-112f;
  const float normalized_value =
      kt_ggml_fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
      kt_ggml_fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff
                  ? kt_ggml_fp32_to_bits(denormalized_value)
                  : kt_ggml_fp32_to_bits(normalized_value));
  return kt_ggml_fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
  float base = (std::fabs(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = kt_ggml_fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = kt_ggml_fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = kt_ggml_fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return static_cast<ggml_fp16_t>(
      (sign >> 16) |
      (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
  return kt_ggml_fp32_from_bits(static_cast<uint32_t>(h.bits) << 16);
}

static inline ggml_bf16_t ggml_compute_fp32_to_bf16(float f) {
  const uint32_t bits = kt_ggml_fp32_to_bits(f);
  const uint32_t rounding_bias = ((bits >> 16) & 1U) + UINT32_C(0x7FFF);
  return ggml_bf16_t{static_cast<uint16_t>((bits + rounding_bias) >> 16)};
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)
#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
