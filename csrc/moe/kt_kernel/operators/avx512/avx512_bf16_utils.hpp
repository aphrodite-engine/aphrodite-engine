// SPDX-License-Identifier: Apache-2.0
#ifndef CPUINFER_OPERATOR_AVX512_BF16_UTILS_H
#define CPUINFER_OPERATOR_AVX512_BF16_UTILS_H

#include <immintrin.h>

#include <algorithm>
#include <cmath>

#include "llama.cpp/ggml.h"

namespace avx512 {

static inline __m512 load_16xbf16_to_fp32(const ggml_bf16_t* src) {
  __m256i bf16 = _mm256_loadu_si256((const __m256i*)src);
  __m512i i32 = _mm512_cvtepu16_epi32(bf16);
  return _mm512_castsi512_ps(_mm512_slli_epi32(i32, 16));
}

static inline __m512bh load_32xbf16(const ggml_bf16_t* src) {
  return (__m512bh)_mm512_loadu_si512((const void*)src);
}

static inline void store_fp32_to_bf16(ggml_bf16_t* dst, __m512 src) {
#if defined(__AVX512BF16__)
  __m256bh bf16 = _mm512_cvtneps_pbh(src);
  _mm256_storeu_si256((__m256i*)dst, (__m256i)bf16);
#else
  __m512i i32 = _mm512_castps_si512(src);
  __m512i tie_bit = _mm512_and_si512(_mm512_srli_epi32(i32, 16), _mm512_set1_epi32(1));
  __m512i round = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), tie_bit);
  __m512i shifted = _mm512_srli_epi32(_mm512_add_epi32(i32, round), 16);
  __m256i packed = _mm512_cvtepi32_epi16(shifted);
  _mm256_storeu_si256((__m256i*)dst, packed);
#endif
}

static inline float hsum(__m512 v) {
  return _mm512_reduce_add_ps(v);
}

static inline __m512 exp_avx512(__m512 x) {
  const __m512 log2e = _mm512_set1_ps(1.44269504089f);
  __m512 y = _mm512_mul_ps(x, log2e);
  __m512i int_part = _mm512_cvtps_epi32(y);
  __m512 frac_part = _mm512_sub_ps(y, _mm512_cvtepi32_ps(int_part));

  const __m512 poly_1 = _mm512_set1_ps(0.9999999995f);
  const __m512 poly_2 = _mm512_set1_ps(0.6931471805f);
  const __m512 poly_3 = _mm512_set1_ps(0.2402265069f);
  const __m512 poly_4 = _mm512_set1_ps(0.0555041087f);
  const __m512 poly_5 = _mm512_set1_ps(0.0096181291f);
  const __m512 poly_6 = _mm512_set1_ps(0.0013333558f);

  __m512 frac_exp = _mm512_fmadd_ps(
      _mm512_fmadd_ps(
          _mm512_fmadd_ps(
              _mm512_fmadd_ps(_mm512_fmadd_ps(poly_6, frac_part, poly_5),
                              frac_part, poly_4),
              frac_part, poly_3),
          frac_part, poly_2),
      frac_part, poly_1);

  __m512i clamped = _mm512_max_epi32(
      _mm512_min_epi32(int_part, _mm512_set1_epi32(127)),
      _mm512_set1_epi32(-126));
  __m512i biased = _mm512_add_epi32(clamped, _mm512_set1_epi32(127));
  __m512 two_pow_i = _mm512_castsi512_ps(_mm512_slli_epi32(biased, 23));
  return _mm512_mul_ps(two_pow_i, frac_exp);
}

static inline __m512 act_fn(__m512 gate_val, __m512 up_val) {
  __m512 neg_gate_val = _mm512_sub_ps(_mm512_setzero_ps(), gate_val);
  neg_gate_val = _mm512_min_ps(neg_gate_val, _mm512_set1_ps(88.0f));
  __m512 exp_neg_gate = exp_avx512(neg_gate_val);
  __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_gate);
  __m512 act_val = _mm512_div_ps(gate_val, denom);
  return _mm512_mul_ps(act_val, up_val);
}

static inline __m512 erf_avx512(__m512 x) {
  const __m512 sign_mask = _mm512_set1_ps(-0.0f);
  __m512 sign = _mm512_and_ps(x, sign_mask);
  __m512 ax = _mm512_andnot_ps(sign_mask, x);

  const __m512 p = _mm512_set1_ps(0.3275911f);
  __m512 t = _mm512_div_ps(_mm512_set1_ps(1.0f),
                           _mm512_fmadd_ps(p, ax, _mm512_set1_ps(1.0f)));

  __m512 poly = _mm512_set1_ps(1.061405429f);
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(-1.453152027f));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(1.421413741f));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(-0.284496736f));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(0.254829592f));
  poly = _mm512_mul_ps(poly, t);

  __m512 exp_term =
      exp_avx512(_mm512_sub_ps(_mm512_setzero_ps(), _mm512_mul_ps(ax, ax)));
  __m512 y = _mm512_sub_ps(_mm512_set1_ps(1.0f),
                           _mm512_mul_ps(poly, exp_term));
  return _mm512_xor_ps(y, sign);
}

static inline __m512 gelu_fn(__m512 gate_val, __m512 up_val) {
  const __m512 inv_sqrt2 = _mm512_set1_ps(0.7071067811865476f);
  __m512 erf_val = erf_avx512(_mm512_mul_ps(gate_val, inv_sqrt2));
  __m512 act_val = _mm512_mul_ps(
      _mm512_mul_ps(_mm512_set1_ps(0.5f), gate_val),
      _mm512_add_ps(_mm512_set1_ps(1.0f), erf_val));
  return _mm512_mul_ps(act_val, up_val);
}

static inline std::pair<int, int> split_range(int total, int ith, int nth) {
  int per = total / nth;
  int rem = total % nth;
  int start = ith * per + std::min(ith, rem);
  int end = start + per + (ith < rem ? 1 : 0);
  return {start, end};
}

}  // namespace avx512

#endif
