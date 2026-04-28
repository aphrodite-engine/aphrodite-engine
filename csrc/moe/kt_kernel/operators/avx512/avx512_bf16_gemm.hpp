// SPDX-License-Identifier: Apache-2.0
#ifndef CPUINFER_OPERATOR_AVX512_BF16_GEMM_H
#define CPUINFER_OPERATOR_AVX512_BF16_GEMM_H

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

#include "avx512_bf16_utils.hpp"

namespace avx512 {

struct GemmKernelAVX512BF16 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 16;
  static constexpr int K_STEP = 32;
  static constexpr int N_BLOCK = 128;
  static constexpr int K_BLOCK = 256;
  static constexpr double ELEMENT_SIZE = 2.0;

  static void config() {}

  static int recommended_nth(int n) {
    return std::max(1, (n + N_BLOCK - 1) / N_BLOCK);
  }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    return split_range(n, ith, nth);
  }

  struct BufferA {
    ggml_bf16_t* data = nullptr;
    size_t max_m = 0;
    size_t k = 0;

    BufferA() = default;
    BufferA(size_t m, size_t k_, void* ptr) : data((ggml_bf16_t*)ptr), max_m(m), k(k_) {}

    static size_t required_size(size_t m, size_t k) {
      return m * k * sizeof(ggml_bf16_t);
    }

    void set_data(void* ptr) { data = (ggml_bf16_t*)ptr; }

    void from_mat(int m, const ggml_bf16_t* src, int ith, int nth) {
      if (ith == 0 && nth == 1) {
        std::memcpy(data, src, (size_t)m * k * sizeof(ggml_bf16_t));
        return;
      }
      auto [m_start, m_end] = split_range(m, ith, nth);
      std::memcpy(data + (size_t)m_start * k, src + (size_t)m_start * k,
                  (size_t)(m_end - m_start) * k * sizeof(ggml_bf16_t));
    }
  };

  struct BufferB {
    ggml_bf16_t* b = nullptr;
    size_t n = 0;
    size_t k = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, void* ptr) : b((ggml_bf16_t*)ptr), n(n_), k(k_) {}

    static size_t required_size(size_t n, size_t k) {
      const size_t n_padded = ((n + N_STEP - 1) / N_STEP) * N_STEP;
      return n_padded * k * sizeof(ggml_bf16_t);
    }

    void from_mat(const ggml_bf16_t* src, int ith, int nth) {
      auto [n_start, n_end] = split_range((int)n, ith, nth);
      // Pack B in 16-output blocks. For each pair of K values, the 16
      // destination columns are laid out as 16 bf16 pairs so vdpbf16ps
      // accumulates a full 16-wide output vector.
      for (int ni = n_start; ni < n_end; ++ni) {
        const int block = ni / N_STEP;
        const int lane = ni % N_STEP;
        const ggml_bf16_t* src_row = src + (size_t)ni * k;
        for (size_t ki = 0; ki < k; ki += 2) {
          ggml_bf16_t* dst = b + ((size_t)block * k + ki) * N_STEP + lane * 2;
          dst[0] = src_row[ki];
          dst[1] = ki + 1 < k ? src_row[ki + 1] : ggml_bf16_t{0};
        }
      }
    }

    void copy_row_slice_to(ggml_bf16_t* dst, int row, int k_start, int len) const {
      const int block = row / N_STEP;
      const int lane = row % N_STEP;
      for (int offset = 0; offset < len; ++offset) {
        const int ki = k_start + offset;
        const size_t pair_ki = (size_t)(ki & ~1);
        const int pair_offset = ki & 1;
        dst[offset] = b[((size_t)block * k + pair_ki) * N_STEP + lane * 2 + pair_offset];
      }
    }
  };

  struct BufferC {
    float* data = nullptr;
    size_t max_m = 0;
    size_t n = 0;

    BufferC() = default;
    BufferC(size_t m, size_t n_, void* ptr) : data((float*)ptr), max_m(m), n(n_) {}

    static size_t required_size(size_t m, size_t n) {
      return m * n * sizeof(float);
    }

    void set_data(void* ptr) { data = (float*)ptr; }

    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
      auto [n_start, n_end] = split_range_n((int)n, ith, nth);
      for (int mi = 0; mi < m; mi++) {
        float* src_row = data + (size_t)mi * n;
        ggml_bf16_t* dst_row = dst + (size_t)mi * n;
        int j = n_start;
        for (; j + 16 <= n_end; j += 16) {
          store_fp32_to_bf16(dst_row + j, _mm512_loadu_ps(src_row + j));
        }
        for (; j < n_end; j++) {
          dst_row[j] = GGML_FP32_TO_BF16(src_row[j]);
        }
      }
    }
  };
};

static inline void gemm_bf16(
    int m, int n, int k,
    GemmKernelAVX512BF16::BufferA& a,
    GemmKernelAVX512BF16::BufferB& b,
    GemmKernelAVX512BF16::BufferC& c,
    int ith, int nth) {
  auto [n_start, n_end] = split_range(n, ith, nth);

  const int block_start = (n_start / GemmKernelAVX512BF16::N_STEP) * GemmKernelAVX512BF16::N_STEP;
  const int block_end = ((n_end + GemmKernelAVX512BF16::N_STEP - 1) / GemmKernelAVX512BF16::N_STEP) *
                        GemmKernelAVX512BF16::N_STEP;

  for (int nb = block_start; nb < block_end; nb += GemmKernelAVX512BF16::N_STEP) {
    const int valid_start = std::max(nb, n_start);
    const int valid_end = std::min(nb + GemmKernelAVX512BF16::N_STEP, n_end);
    if (valid_start >= valid_end) {
      continue;
    }
    const int block = nb / GemmKernelAVX512BF16::N_STEP;
    for (int mi = 0; mi < m; mi++) {
      const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
      __m512 acc = _mm512_setzero_ps();
      int ki = 0;
#if defined(__AVX512BF16__)
      for (; ki + 1 < k; ki += 2) {
        const uint32_t a_pair = (uint32_t)a_row[ki].bits | ((uint32_t)a_row[ki + 1].bits << 16);
        const __m512bh a_vec = (__m512bh)_mm512_set1_epi32((int)a_pair);
        const __m512bh b_vec = load_32xbf16(b.b + ((size_t)block * k + ki) * GemmKernelAVX512BF16::N_STEP);
        acc = _mm512_dpbf16_ps(acc, a_vec, b_vec);
      }
#endif
      if (ki < k) {
        const float a_scalar = GGML_BF16_TO_FP32(a_row[ki]);
        const ggml_bf16_t* b_tail = b.b + ((size_t)block * k + ki) * GemmKernelAVX512BF16::N_STEP;
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, acc);
        for (int lane = 0; lane < GemmKernelAVX512BF16::N_STEP && nb + lane < n; ++lane) {
          tmp[lane] += a_scalar * GGML_BF16_TO_FP32(b_tail[lane * 2]);
        }
        acc = _mm512_load_ps(tmp);
      }
      if (valid_start == nb && valid_end == nb + GemmKernelAVX512BF16::N_STEP) {
        _mm512_storeu_ps(c.data + (size_t)mi * n + nb, acc);
      } else {
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, acc);
        for (int ni = valid_start; ni < valid_end; ++ni) {
          c.data[(size_t)mi * n + ni] = tmp[ni - nb];
        }
      }
    }
  }
}

static inline void vec_mul(
    int m, int n, int k,
    std::shared_ptr<GemmKernelAVX512BF16::BufferA>& a,
    std::shared_ptr<GemmKernelAVX512BF16::BufferB>& b,
    std::shared_ptr<GemmKernelAVX512BF16::BufferC>& c,
    int ith, int nth) {
  gemm_bf16(m, n, k, *a, *b, *c, ith, nth);
}

static inline void mat_mul(
    int m, int n, int k,
    std::shared_ptr<GemmKernelAVX512BF16::BufferA>& a,
    std::shared_ptr<GemmKernelAVX512BF16::BufferB>& b,
    std::shared_ptr<GemmKernelAVX512BF16::BufferC>& c,
    int ith, int nth) {
  gemm_bf16(m, n, k, *a, *b, *c, ith, nth);
}

}  // namespace avx512

#endif
