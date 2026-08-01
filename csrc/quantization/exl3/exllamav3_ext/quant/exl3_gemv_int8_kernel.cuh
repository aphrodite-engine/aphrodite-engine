#pragma once

// Device side of the fused int8-activation GEMV (see comment below). Kernels
// are instantiated in comp_units/exl3_gemv_int8_inst_*.cu; the host code in
// exl3_gemv_int8.cu includes this header only for the launch-geometry constants
// and __host__ __device__ helpers

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cooperative_groups.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"

namespace cg_gemv = cooperative_groups;

// Fused int8-activation GEMV for mul1 (cb 2) tensors, single cooperative
// launch.
//
// The mul1 codebook value is affine in the byte sum of x * 0x83DCD12D, so with
// activations quantized to int8, dp4a(x * M, splat(a_int8), acc) evaluates
// codebook(x) * a in a single instruction with exact int32 accumulation, and
//
//   y[n] = k_inv * (q * acc1[n] + q2 * acc2[n]) + (1024 * k_inv + k_bias) * (q
//   * sum1 + q2 * sum2)
//
// recovers the result. In residual mode (EXL3_INT8_GEMV=1) the int8 rounding
// error r = a - q*i, which has range +/-q/2 by construction, is quantized with
// the fixed scale q2 = q/254 and accumulated by a second dp4a sharing the
// decoded products, for ~15-16 bit effective activation precision (KL at parity
// with the fp16 kernel or better). EXL3_INT8_GEMV=2 is the plain int8 mode.
//
// One cooperative launch with the same argument list as exl3_gemm_kernel;
// `locks` doubles as workspace: [2*size_n int32 accumulators][{float q, int s1,
// int s2, pad} per row][partial max per block]. Phases (persistent 256-thread
// blocks):
//   1a. input Hadamard A -> A_had (as in exl3_gemm_kernel) + accumulator
//   zeroing; for m == 1 also a
//       per-block partial max of the transformed activations
//   1b. (m > 1 only) per-row scale and exact int sums
//   per row:
//     2. GEMV over (256-column x k-slice) units; activation splats are
//     quantized INLINE while
//        staging to shared memory (for m == 1, every block re-derives q from
//        the phase-1a partial maxes - fmax is order-independent so this is
//        deterministic - and block 0 computes the exact epilogue sums
//        concurrently); grid.sync
//     3. epilogue: affine correction + output Hadamard + svh + accumulator
//     reset, warp-striped
//
// K == 4 uses the wide unit: a warp owns an ADJACENT BLOCK PAIR; lane l's uint2
// = words {2m, 2m+1} of the contiguous 64-word region are exactly the primary
// words of runs t = 8*(2m), 8*(2m+1), which scatter to the SAME two n values
// (accumulator count unchanged, splats merge into two uint4 loads, single-stage
// butterfly, boundary word via one shuffle). 256 B contiguous per warp per
// block row with two-row prefetch. Other K use a generic narrow unit (two
// blocks per warp, pointer-based extraction). Natural register allocation (2-3
// resident blocks/SM) measures faster than forcing higher occupancy: with wide
// loads the per-warp ILP is worth more than the extra warps.

#define NUM_THREADS 256

// ---------------------------------------------------------------------------------------------------------
// Extract 8 packed 16-bit trellis windows for positions t0..t0+7 as masked
// uint32s. Index math copied from the corresponding paths in exl3_dq.cuh
// (dq8_aligned_*, dq8, dq4, dq2x2).

__device__ __forceinline__ int dp4a_us(uint32_t a, uint32_t b, int c) {
  int d;
  asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// i0/i2 land in [0, 2*words); a compare+subtract replaces the modulo (words is
// not a power of two for odd bit widths)
template <int bits>
__device__ __forceinline__ int wrap_idx(int i) {
  constexpr int words = bits * 256 / 32;
  return i >= words ? i - words : i;
}

template <int bits>
__device__ __forceinline__ void ext4w(const uint32_t* ptr, int t0, uint32_t& w0,
                                      uint32_t& w1, uint32_t& w2,
                                      uint32_t& w3) {
  int b0 = (t0 + 257) * bits - 16;
  int b2 = b0 + 3 * bits + 16;
  int i0 = b0 / 32;
  int i2 = (b2 - 1) / 32;
  int s2 = (i2 + 1) * 32 - b2;
  uint32_t a = ptr[wrap_idx<bits>(i0)];
  uint32_t b = ptr[wrap_idx<bits>(i2)];
  w3 = fshift(b, a, s2) & 0xffff;
  w2 = fshift(b, a, s2 + bits) & 0xffff;
  w1 = fshift(b, a, s2 + bits * 2) & 0xffff;
  w0 = fshift(b, a, s2 + bits * 3) & 0xffff;
}

template <int bits>
__device__ __forceinline__ void ext2w(const uint32_t* ptr, int t0, uint32_t& w0,
                                      uint32_t& w1) {
  int b0 = (t0 + 257) * bits - 16;
  int b2 = b0 + bits + 16;
  int i0 = b0 / 32;
  int i2 = (b2 - 1) / 32;
  int s2 = (i2 + 1) * 32 - b2;
  uint32_t a = ptr[wrap_idx<bits>(i0)];
  uint32_t b = ptr[wrap_idx<bits>(i2)];
  w1 = fshift(b, a, s2) & 0xffff;
  w0 = fshift(b, a, s2 + bits) & 0xffff;
}

template <int bits>
__device__ __forceinline__ void ext8w(const uint32_t* ptr, int t0, uint32_t& w0,
                                      uint32_t& w1, uint32_t& w2, uint32_t& w3,
                                      uint32_t& w4, uint32_t& w5, uint32_t& w6,
                                      uint32_t& w7) {
  if constexpr (bits == 1) {
    uint32_t i1 = t0 >> 5;
    uint32_t i0 = (i1 + 7) & 7;
    uint32_t a = ptr[i0];
    uint32_t b = ptr[i1];
    b = fshift(b, a, ((~t0) & 24));
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 1);
    BFE16_IMM(w5, b, 2);
    BFE16_IMM(w4, b, 3);
    BFE16_IMM(w3, b, 4);
    BFE16_IMM(w2, b, 5);
    BFE16_IMM(w1, b, 6);
    BFE16_IMM(w0, b, 7);
  } else if constexpr (bits == 2) {
    uint32_t i1 = t0 >> 4;
    uint32_t i0 = (i1 + 15) & 15;
    uint32_t a = ptr[i0];
    uint32_t b = ptr[i1];
    b = fshift(b, a, ((~t0) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
  } else if constexpr (bits == 3) {
    int b1 = (t0 + 257) * bits;
    int b0 = b1 - 16;
    int b2 = b1 + bits * 7;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[wrap_idx<bits>(i0)];
    uint32_t b = ptr[wrap_idx<bits>(i2)];
    w7 = fshift(b, a, s2);
    w6 = w7 >> bits;
    w5 = w6 >> bits;
    w4 = w5 >> bits;
    w3 = fshift(b, a, s2 + bits * 4);
    w2 = w3 >> bits;
    w1 = w2 >> bits;
    w0 = w1 >> bits;
    w7 &= 0xffff;
    w6 &= 0xffff;
    w5 &= 0xffff;
    w4 &= 0xffff;
    w3 &= 0xffff;
    w2 &= 0xffff;
    w1 &= 0xffff;
    w0 &= 0xffff;
  } else if constexpr (bits == 4) {
    uint32_t i1 = t0 >> 3;
    uint32_t i0 = (i1 + 31) & 31;
    uint32_t a = ptr[i0];
    uint32_t b = ptr[i1];
    uint32_t s;
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
  } else if constexpr (bits == 7) {
    ext2w<bits>(ptr, t0, w0, w1);
    ext2w<bits>(ptr, t0 + 2, w2, w3);
    ext2w<bits>(ptr, t0 + 4, w4, w5);
    ext2w<bits>(ptr, t0 + 6, w6, w7);
  } else  // 5, 6, 8
  {
    ext4w<bits>(ptr, t0, w0, w1, w2, w3);
    ext4w<bits>(ptr, t0 + 4, w4, w5, w6, w7);
  }
}

// 4bpw extraction from two already-loaded words (same window order)
__device__ __forceinline__ void extract8_4bits_words(
    uint32_t a, uint32_t b, uint32_t& w0, uint32_t& w1, uint32_t& w2,
    uint32_t& w3, uint32_t& w4, uint32_t& w5, uint32_t& w6, uint32_t& w7) {
  uint32_t s;
  FSHF_IMM(s, b, a, 20);
  w7 = b & 0xffff;
  BFE16_IMM(w6, b, 4);
  BFE16_IMM(w5, b, 8);
  BFE16_IMM(w4, b, 12);
  BFE16_IMM(w3, b, 16);
  w2 = s & 0xffff;
  BFE16_IMM(w1, s, 4);
  BFE16_IMM(w0, s, 8);
}

// ---------------------------------------------------------------------------------------------------------
// Device building blocks (also reusable from batched mgemm/MoE-style kernels)

// Quantize one k-slice of the (Hadamard-transformed) activation row into int8
// byte-splats in shared memory, plus optional error-feedback residual splats
template <bool residual>
__device__ __forceinline__ void gemv_int8_stage_splats(
    const half* __restrict__ A_had_row, float q, uint32_t* __restrict__ sh_as,
    uint32_t* __restrict__ sh_as2, int kb0, int nrows) {
  __syncthreads();
  float rq = 1.0f / q;
  float rq2 = rq * 254.0f;
  for (int i = threadIdx.x; i < nrows * 16; i += NUM_THREADS) {
    float a = __half2float(A_had_row[(kb0 << 4) + i]);
    int v = __float2int_rn(a * rq);
    v = max(-127, min(127, v));
    sh_as[i] = ((uint32_t)(uint8_t)(int8_t)v) * 0x01010101u;
    if constexpr (residual) {
      float r = a - q * (float)v;
      int v2 = __float2int_rn(r * rq2);
      v2 = max(-127, min(127, v2));
      sh_as2[i] = ((uint32_t)(uint8_t)(int8_t)v2) * 0x01010101u;
    }
  }
  __syncthreads();
}

// Block-wide: exact int8(+residual) sums of one activation row, writes {q, s1,
// s2} to qsums_row
template <bool residual>
__device__ __forceinline__ void gemv_int8_row_sums(
    const half* __restrict__ Ar, float q, int size_k,
    float* __restrict__ qsums_row) {
  __shared__ int sh_s1[32];
  __shared__ int sh_s2[32];
  int t = threadIdx.x;
  float rq = 1.0f / q;
  float rq2 = rq * 254.0f;
  int s1 = 0, s2 = 0;
  for (int i = t; i < size_k; i += NUM_THREADS) {
    float a = __half2float(Ar[i]);
    int v = __float2int_rn(a * rq);
    v = max(-127, min(127, v));
    s1 += v;
    if constexpr (residual) {
      float r = a - q * (float)v;
      int v2 = __float2int_rn(r * rq2);
      s2 += max(-127, min(127, v2));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    s1 += __shfl_xor_sync(0xffffffff, s1, o);
    s2 += __shfl_xor_sync(0xffffffff, s2, o);
  }
  if ((t & 31) == 0) {
    sh_s1[t >> 5] = s1;
    sh_s2[t >> 5] = s2;
  }
  __syncthreads();
  if (t < 32) {
    int v1 = t < (NUM_THREADS >> 5) ? sh_s1[t] : 0;
    int v2 = t < (NUM_THREADS >> 5) ? sh_s2[t] : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      v1 += __shfl_xor_sync(0xffffffff, v1, o);
      v2 += __shfl_xor_sync(0xffffffff, v2, o);
    }
    if (t == 0) {
      qsums_row[0] = q;
      ((int*)qsums_row)[1] = v1;
      ((int*)qsums_row)[2] = v2;
    }
  }
  __syncthreads();
}

// The K cap (fall back to the regular kernel in the DRAM-bound regime) is
// per-arch; see exl3_gemv_int8_max_k in exl3_gemv_int8.cu
#define GEMV_STAGE_D \
  4  // cp.async pipeline depth (rows) for the smem-staged unit
#define GEMV_STAGE_MAX_BYTES (8 * GEMV_STAGE_D * 16 * 8 * 4)  // 8 warps, K = 8

// Per-slice-scale ("sq") kernel parameters
#define SQ_KSPLIT_CAP 64
#define SQ_MINROWS 16    // minimum slice height (staging overhead amortization)
#define SQ_ROWS_MAX 512  // shared memory cap
#define SQ_COUNTERS_CAP 4096  // fixed counter region: size_n up to 1M
#define SQ_WS_RESERVED \
  (SQ_COUNTERS_CAP +   \
   4 * SQ_KSPLIT_CAP * 4)  // sq-private region at workspace start (counters +
                           // per-slice-per-row scales, m <= 4)

// Wide unit (4 bpw): one (256-column x k-slice) unit, warp per adjacent block
// pair, uint2 loads. M activation rows share the decoded weights: extraction
// and the B stream are amortized, each row only adds its dp4a chains against
// its own splat region (sh_as + r * slice_stride; residual splats at (M + r) *
// slice_stride - at M = 1 this is exactly the coop kernel's sh_as/sh_as2
// layout). Row accumulators go to accs + r * acc_stride, atomically or with
// plain stores (exclusive per-slice partials of the sq kernel).
template <int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_unit_wide(
    const uint16_t* __restrict__ B, int* __restrict__ accs, size_t acc_stride,
    const uint32_t* __restrict__ sh_as, int slice_stride, int nb256, int kb0,
    int nrows, int size_n) {
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int nbp = nb256 * 8 + warp;
  const int row_stride = size_n * 2;  // u32 per block row at 4 bpw
  const uint32_t* bp = ((const uint32_t*)B) + (size_t)kb0 * row_stride +
                       (size_t)nbp * 64 + 2 * lane;
  int c2 = (lane & 1) ? 4 : 0;
  int shfl_src = (lane & 16) | ((lane + 15) & 15);

  int iacc0[M] = {}, iacc1[M] = {}, jacc0[M] = {}, jacc1[M] = {};
  uint2 r0 = *(const uint2*)bp;
  uint2 r1 = {};
  if (nrows > 1) r1 = *(const uint2*)(bp + row_stride);

  for (int kb = 0; kb < nrows; ++kb) {
    uint2 r2 = {};
    if (kb + 2 < nrows)
      r2 = *(const uint2*)(bp + (size_t)(kb + 2) * row_stride);
    uint32_t prev = __shfl_sync(0xffffffff, r0.y, shfl_src);

    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
    extract8_4bits_words(prev, r0.x, w0, w1, w2, w3, w4, w5, w6,
                         w7);  // run t = 8*(2m)
    extract8_4bits_words(r0.x, r0.y, v0, v1, v2, v3, v4, v5, v6,
                         v7);  // run t = 8*(2m+1)
    w0 *= 0x83DCD12Du;
    w1 *= 0x83DCD12Du;
    w2 *= 0x83DCD12Du;
    w3 *= 0x83DCD12Du;
    w4 *= 0x83DCD12Du;
    w5 *= 0x83DCD12Du;
    w6 *= 0x83DCD12Du;
    w7 *= 0x83DCD12Du;
    v0 *= 0x83DCD12Du;
    v1 *= 0x83DCD12Du;
    v2 *= 0x83DCD12Du;
    v3 *= 0x83DCD12Du;
    v4 *= 0x83DCD12Du;
    v5 *= 0x83DCD12Du;
    v6 *= 0x83DCD12Du;
    v7 *= 0x83DCD12Du;

#pragma unroll
    for (int r = 0; r < M; ++r) {
      const uint32_t* as = sh_as + r * slice_stride + (kb << 4);
      uint4 as0 = *(const uint4*)(as + c2);
      uint4 as8 = *(const uint4*)(as + c2 + 8);
      iacc0[r] = dp4a_us(w0, as0.x, iacc0[r]);
      iacc0[r] = dp4a_us(w1, as0.y, iacc0[r]);
      iacc0[r] = dp4a_us(w2, as8.x, iacc0[r]);
      iacc0[r] = dp4a_us(w3, as8.y, iacc0[r]);
      iacc1[r] = dp4a_us(w4, as0.x, iacc1[r]);
      iacc1[r] = dp4a_us(w5, as0.y, iacc1[r]);
      iacc1[r] = dp4a_us(w6, as8.x, iacc1[r]);
      iacc1[r] = dp4a_us(w7, as8.y, iacc1[r]);
      iacc0[r] = dp4a_us(v0, as0.z, iacc0[r]);
      iacc0[r] = dp4a_us(v1, as0.w, iacc0[r]);
      iacc0[r] = dp4a_us(v2, as8.z, iacc0[r]);
      iacc0[r] = dp4a_us(v3, as8.w, iacc0[r]);
      iacc1[r] = dp4a_us(v4, as0.z, iacc1[r]);
      iacc1[r] = dp4a_us(v5, as0.w, iacc1[r]);
      iacc1[r] = dp4a_us(v6, as8.z, iacc1[r]);
      iacc1[r] = dp4a_us(v7, as8.w, iacc1[r]);
      if constexpr (residual) {
        const uint32_t* as2 = sh_as + (M + r) * slice_stride + (kb << 4);
        uint4 bs0 = *(const uint4*)(as2 + c2);
        uint4 bs8 = *(const uint4*)(as2 + c2 + 8);
        jacc0[r] = dp4a_us(w0, bs0.x, jacc0[r]);
        jacc0[r] = dp4a_us(w1, bs0.y, jacc0[r]);
        jacc0[r] = dp4a_us(w2, bs8.x, jacc0[r]);
        jacc0[r] = dp4a_us(w3, bs8.y, jacc0[r]);
        jacc1[r] = dp4a_us(w4, bs0.x, jacc1[r]);
        jacc1[r] = dp4a_us(w5, bs0.y, jacc1[r]);
        jacc1[r] = dp4a_us(w6, bs8.x, jacc1[r]);
        jacc1[r] = dp4a_us(w7, bs8.y, jacc1[r]);
        jacc0[r] = dp4a_us(v0, bs0.z, jacc0[r]);
        jacc0[r] = dp4a_us(v1, bs0.w, jacc0[r]);
        jacc0[r] = dp4a_us(v2, bs8.z, jacc0[r]);
        jacc0[r] = dp4a_us(v3, bs8.w, jacc0[r]);
        jacc1[r] = dp4a_us(v4, bs0.z, jacc1[r]);
        jacc1[r] = dp4a_us(v5, bs0.w, jacc1[r]);
        jacc1[r] = dp4a_us(v6, bs8.z, jacc1[r]);
        jacc1[r] = dp4a_us(v7, bs8.w, jacc1[r]);
      }
    }

    r0 = r1;
    r1 = r2;
  }

// Lanes l, l^1 share both n values
#pragma unroll
  for (int r = 0; r < M; ++r) {
    iacc0[r] += __shfl_xor_sync(0xffffffff, iacc0[r], 1);
    iacc1[r] += __shfl_xor_sync(0xffffffff, iacc1[r], 1);
    if constexpr (residual) {
      jacc0[r] += __shfl_xor_sync(0xffffffff, jacc0[r], 1);
      jacc1[r] += __shfl_xor_sync(0xffffffff, jacc1[r], 1);
    }
  }
  if (!(lane & 1)) {
    int nb = nbp * 2 + (lane >> 4);
    int n0 = nb * 16 + ((lane & 15) >> 1);
#pragma unroll
    for (int r = 0; r < M; ++r) {
      int* acc = accs + r * acc_stride;
      if constexpr (atomic) {
        atomicAdd(acc + n0, iacc0[r]);
        atomicAdd(acc + n0 + 8, iacc1[r]);
        if constexpr (residual) {
          atomicAdd(acc + size_n + n0, jacc0[r]);
          atomicAdd(acc + size_n + n0 + 8, jacc1[r]);
        }
      } else {
        acc[n0] = iacc0[r];
        acc[n0 + 8] = iacc1[r];
        if constexpr (residual) {
          acc[size_n + n0] = jacc0[r];
          acc[size_n + n0 + 8] = jacc1[r];
        }
      }
    }
  }
}

// One k-row of an adjacent block pair, generic K: extract + dp4a for both
// blocks from block pointers (global or shared memory), M rows sharing the
// extraction
template <int bits, int M, bool residual>
__device__ __forceinline__ void gemv_int8_pair_row(
    const uint32_t* blockA, const uint32_t* blockB,
    const uint32_t* as_kb,  // sh_as + (kb << 4)
    int slice_stride, int c2, int t0, int* ia0, int* ia1, int* ib0, int* ib1,
    int* ja0, int* ja1, int* jb0, int* jb1) {
  uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
  ext8w<bits>(blockA, t0, w0, w1, w2, w3, w4, w5, w6, w7);
  w0 *= 0x83DCD12Du;
  w1 *= 0x83DCD12Du;
  w2 *= 0x83DCD12Du;
  w3 *= 0x83DCD12Du;
  w4 *= 0x83DCD12Du;
  w5 *= 0x83DCD12Du;
  w6 *= 0x83DCD12Du;
  w7 *= 0x83DCD12Du;
#pragma unroll
  for (int r = 0; r < M; ++r) {
    const uint32_t* as = as_kb + r * slice_stride;
    uint2 as01 = *(const uint2*)(as + c2);
    uint2 as89 = *(const uint2*)(as + c2 + 8);
    ia0[r] = dp4a_us(w0, as01.x, ia0[r]);
    ia0[r] = dp4a_us(w1, as01.y, ia0[r]);
    ia0[r] = dp4a_us(w2, as89.x, ia0[r]);
    ia0[r] = dp4a_us(w3, as89.y, ia0[r]);
    ia1[r] = dp4a_us(w4, as01.x, ia1[r]);
    ia1[r] = dp4a_us(w5, as01.y, ia1[r]);
    ia1[r] = dp4a_us(w6, as89.x, ia1[r]);
    ia1[r] = dp4a_us(w7, as89.y, ia1[r]);
    if constexpr (residual) {
      const uint32_t* as2 = as_kb + (M + r) * slice_stride;
      uint2 bs01 = *(const uint2*)(as2 + c2);
      uint2 bs89 = *(const uint2*)(as2 + c2 + 8);
      ja0[r] = dp4a_us(w0, bs01.x, ja0[r]);
      ja0[r] = dp4a_us(w1, bs01.y, ja0[r]);
      ja0[r] = dp4a_us(w2, bs89.x, ja0[r]);
      ja0[r] = dp4a_us(w3, bs89.y, ja0[r]);
      ja1[r] = dp4a_us(w4, bs01.x, ja1[r]);
      ja1[r] = dp4a_us(w5, bs01.y, ja1[r]);
      ja1[r] = dp4a_us(w6, bs89.x, ja1[r]);
      ja1[r] = dp4a_us(w7, bs89.y, ja1[r]);
    }
  }

  ext8w<bits>(blockB, t0, w0, w1, w2, w3, w4, w5, w6, w7);
  w0 *= 0x83DCD12Du;
  w1 *= 0x83DCD12Du;
  w2 *= 0x83DCD12Du;
  w3 *= 0x83DCD12Du;
  w4 *= 0x83DCD12Du;
  w5 *= 0x83DCD12Du;
  w6 *= 0x83DCD12Du;
  w7 *= 0x83DCD12Du;
#pragma unroll
  for (int r = 0; r < M; ++r) {
    const uint32_t* as = as_kb + r * slice_stride;
    uint2 as01 = *(const uint2*)(as + c2);
    uint2 as89 = *(const uint2*)(as + c2 + 8);
    ib0[r] = dp4a_us(w0, as01.x, ib0[r]);
    ib0[r] = dp4a_us(w1, as01.y, ib0[r]);
    ib0[r] = dp4a_us(w2, as89.x, ib0[r]);
    ib0[r] = dp4a_us(w3, as89.y, ib0[r]);
    ib1[r] = dp4a_us(w4, as01.x, ib1[r]);
    ib1[r] = dp4a_us(w5, as01.y, ib1[r]);
    ib1[r] = dp4a_us(w6, as89.x, ib1[r]);
    ib1[r] = dp4a_us(w7, as89.y, ib1[r]);
    if constexpr (residual) {
      const uint32_t* as2 = as_kb + (M + r) * slice_stride;
      uint2 bs01 = *(const uint2*)(as2 + c2);
      uint2 bs89 = *(const uint2*)(as2 + c2 + 8);
      jb0[r] = dp4a_us(w0, bs01.x, jb0[r]);
      jb0[r] = dp4a_us(w1, bs01.y, jb0[r]);
      jb0[r] = dp4a_us(w2, bs89.x, jb0[r]);
      jb0[r] = dp4a_us(w3, bs89.y, jb0[r]);
      jb1[r] = dp4a_us(w4, bs01.x, jb1[r]);
      jb1[r] = dp4a_us(w5, bs01.y, jb1[r]);
      jb1[r] = dp4a_us(w6, bs89.x, jb1[r]);
      jb1[r] = dp4a_us(w7, bs89.y, jb1[r]);
    }
  }
}

// Shared reduction tail for the pair-per-warp generic units (atomic, or
// exclusive plain stores for the per-slice partials of the sq kernel): four
// lanes share each n
template <int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_pair_tail(
    int* __restrict__ accs, size_t acc_stride, int nbp, int lane, int size_n,
    int* ia0, int* ia1, int* ib0, int* ib1, int* ja0, int* ja1, int* jb0,
    int* jb1) {
#pragma unroll
  for (int r = 0; r < M; ++r) {
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
      ia0[r] += __shfl_xor_sync(0xffffffff, ia0[r], o);
      ia1[r] += __shfl_xor_sync(0xffffffff, ia1[r], o);
      ib0[r] += __shfl_xor_sync(0xffffffff, ib0[r], o);
      ib1[r] += __shfl_xor_sync(0xffffffff, ib1[r], o);
      if constexpr (residual) {
        ja0[r] += __shfl_xor_sync(0xffffffff, ja0[r], o);
        ja1[r] += __shfl_xor_sync(0xffffffff, ja1[r], o);
        jb0[r] += __shfl_xor_sync(0xffffffff, jb0[r], o);
        jb1[r] += __shfl_xor_sync(0xffffffff, jb1[r], o);
      }
    }
  }
  if ((lane & 3) == 0) {
    int nA = (nbp * 2) * 16 + (lane >> 2);
    int nB = nA + 16;
#pragma unroll
    for (int r = 0; r < M; ++r) {
      int* acc = accs + r * acc_stride;
      if constexpr (atomic) {
        atomicAdd(acc + nA, ia0[r]);
        atomicAdd(acc + nA + 8, ia1[r]);
        atomicAdd(acc + nB, ib0[r]);
        atomicAdd(acc + nB + 8, ib1[r]);
        if constexpr (residual) {
          atomicAdd(acc + size_n + nA, ja0[r]);
          atomicAdd(acc + size_n + nA + 8, ja1[r]);
          atomicAdd(acc + size_n + nB, jb0[r]);
          atomicAdd(acc + size_n + nB + 8, jb1[r]);
        }
      } else {
        acc[nA] = ia0[r];
        acc[nA + 8] = ia1[r];
        acc[nB] = ib0[r];
        acc[nB + 8] = ib1[r];
        if constexpr (residual) {
          acc[size_n + nA] = ja0[r];
          acc[size_n + nA + 8] = ja1[r];
          acc[size_n + nB] = jb0[r];
          acc[size_n + nB + 8] = jb1[r];
        }
      }
    }
  }
}

// Narrow generic unit (any K): one (256-column x k-slice) unit, warp per
// adjacent block pair processed sequentially with pointer-based extraction
// straight from global memory
template <int bits, int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_unit_narrow(
    const uint16_t* __restrict__ B, int* __restrict__ accs, size_t acc_stride,
    const uint32_t* __restrict__ sh_as, int slice_stride, int nb256, int kb0,
    int nrows, int size_n) {
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int nbp = nb256 * 8 + warp;
  const int row_stride = size_n * bits / 2;
  const uint32_t* bp = ((const uint32_t*)B) + (size_t)kb0 * row_stride +
                       (size_t)nbp * (bits * 16);
  int c2 = 2 * (lane & 3);
  int ia0[M] = {}, ia1[M] = {}, ib0[M] = {}, ib1[M] = {};
  int ja0[M] = {}, ja1[M] = {}, jb0[M] = {}, jb1[M] = {};

  for (int kb = 0; kb < nrows; ++kb) {
    const uint32_t* blockA = bp + (size_t)kb * row_stride;
    gemv_int8_pair_row<bits, M, residual>(
        blockA, blockA + 8 * bits, sh_as + (kb << 4), slice_stride, c2,
        lane << 3, ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
  }
  gemv_int8_pair_tail<M, residual, atomic>(accs, acc_stride, nbp, lane, size_n,
                                           ia0, ia1, ib0, ib1, ja0, ja1, jb0,
                                           jb1);
}

// Smem-staged generic unit: the warp stages its block pair's rows into a
// warp-private shared memory slice with cp.async (coalesced 16 B chunks, one
// commit group per row, GEMV_STAGE_D rows deep) and extracts from shared
// memory. Used for the K where scattered pointer extraction leaves the most
// load latency exposed (3, 5, 7); warp-private slices need no block-level
// synchronization.
template <int bits, int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_unit_smem(
    const uint16_t* __restrict__ B, int* __restrict__ accs, size_t acc_stride,
    const uint32_t* __restrict__ sh_as, int slice_stride,
    uint32_t* __restrict__ sh_b, int nb256, int kb0, int nrows, int size_n) {
  constexpr int D = GEMV_STAGE_D;
  constexpr int pairwords = 16 * bits;
  constexpr int chunks = pairwords / 4;  // 16 B cp.async chunks per pair row
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int nbp = nb256 * 8 + warp;
  const int row_stride = size_n * bits / 2;
  const uint32_t* bp =
      ((const uint32_t*)B) + (size_t)kb0 * row_stride + (size_t)nbp * pairwords;
  uint32_t* sb = sh_b + warp * (D * pairwords);

  auto stage_row = [&](int kb) {
    if (kb < nrows && lane < chunks)
      cp_async(sb + (kb % D) * pairwords + lane * 4,
               bp + (size_t)kb * row_stride + lane * 4);
    cp_async_fence();
  };
#pragma unroll
  for (int r = 0; r < D - 1; ++r) stage_row(r);

  int c2 = 2 * (lane & 3);
  int ia0[M] = {}, ia1[M] = {}, ib0[M] = {}, ib1[M] = {};
  int ja0[M] = {}, ja1[M] = {}, jb0[M] = {}, jb1[M] = {};

  for (int kb = 0; kb < nrows; ++kb) {
    cp_async_wait<D - 2>();
    // Also orders the previous iteration's smem reads (all lanes) before the
    // overwrite below
    __syncwarp();
    stage_row(kb + D - 1);

    const uint32_t* blockA = sb + (kb % D) * pairwords;
    gemv_int8_pair_row<bits, M, residual>(
        blockA, blockA + 8 * bits, sh_as + (kb << 4), slice_stride, c2,
        lane << 3, ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
  }
  gemv_int8_pair_tail<M, residual, atomic>(accs, acc_stride, nbp, lane, size_n,
                                           ia0, ia1, ib0, ib1, ja0, ja1, jb0,
                                           jb1);
}

// K values routed to the smem-staged unit (measured on 3090: K=3 -13%; narrow
// wins for 2/6/8 which are at the ALU floor / DRAM-bound, wide covers 4)
__host__ __device__ constexpr bool gemv_int8_stage_smem(int bits) {
  return bits == 3 || bits == 5 || bits == 7;
}

// Epilogue for one row: affine correction + output Hadamard + svh scale +
// accumulator reset, striped over all warps of the grid. sh_tmp: 128 floats per
// warp.
template <bool c_fp32, bool residual>
__device__ __forceinline__ void gemv_int8_epilogue(
    int* __restrict__ accs, void* __restrict__ C_row,
    const float* __restrict__ qrow, const half* __restrict__ svh,
    float* __restrict__ sh_tmp, int size_n, float output_scale) {
  int lane = threadIdx.x & 31;
  int warps_grid = gridDim.x * NUM_THREADS / 32;
  int this_warp = threadIdx.x / 32 + NUM_THREADS / 32 * blockIdx.x;
  float* tmp = sh_tmp + (threadIdx.x / 32) * 128;

  float k_inv = __half2float(__ushort_as_half(0x1eee));
  float k_bias = __half2float(__ushort_as_half(0xc931));
  float q = qrow[0];
  float q2 = q * (1.0f / 254.0f);
  float suma = q * (float)((const int*)qrow)[1];
  if constexpr (residual) suma += q2 * (float)((const int*)qrow)[2];
  float corr = (1024.0f * k_inv + k_bias) * suma;

  int total_chunks = size_n / 128;
  for (; this_warp < total_chunks; this_warp += warps_grid) {
    int base = this_warp * 128;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int j = lane * 4 + i;
      float s = q * (float)accs[base + j];
      accs[base + j] = 0;
      if constexpr (residual) {
        s += q2 * (float)accs[size_n + base + j];
        accs[size_n + base + j] = 0;
      }
      tmp[j] = k_inv * s + corr;
    }
    __syncwarp();
    // gridDim.y == 1, so the inner's blockIdx.y-based scale offset is zero;
    // pre-offset svh
    if constexpr (c_fp32)
      had_ff_r_128_inner<false, true>(tmp, ((float*)C_row) + base, svh + base,
                                      0.088388347648f * output_scale);
    else
      had_fh_r_128_inner<false, true>(tmp, ((half*)C_row) + base, svh + base,
                                      0.088388347648f * output_scale);
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------------------------------------
// Per-slice-scale ("sq") kernel for m == 1: kills ALL global pre-GEMV
// coordination. Each k-slice is Hadamard-transformed, maxed and quantized
// block-locally during staging with its own q_s = slicemax/127 (deterministic,
// so duplicating blocks agree bit-exactly - and strictly finer quantization
// than a global q), units write exact per-slice int32 partials with plain
// stores (exclusive writers - no atomics, no accumulator zeroing), and a
// per-256-column completion counter gates an inline epilogue that combines
// sum_s q_s*acc_s in fixed slice order (deterministic) plus the per-slice
// affine corrections. One REGULAR launch: no cooperative machinery, no
// grid.sync, no A_had traffic. Same 10 kernel arguments (A_had unused).
// Workspace layout (fixed offsets so mixed sq/coop calls of any shape can share
// the buffer):
//   [counters: SQ_COUNTERS_CAP][{q_s, s1_s, s2_s, pad} x
//   SQ_KSPLIT_CAP][partials: ksplit x pstride]
// The coop kernels use the region beyond SQ_WS_RESERVED. Counters are zeroed
// when the workspace is (re)allocated and reset by the finishing block, so they
// stay zero between calls.

// Stage one slice for M activation rows: Hadamard from A into sh_ah, slice max
// -> q_s, splats + exact sums per row. Bit-identical in every block that stages
// the same slice. Rows >= size_m (padding at M = 4, m = 3) get zero splats and
// dummy scales.
template <int M, bool residual>
__device__ __forceinline__ void gemv_int8_stage_slice(
    const half* __restrict__ A, int size_m, int size_k,
    const half* __restrict__ suh,
    float* __restrict__ qs,  // qsums + 4 * slice * M, rows consecutive,
                             // published (idempotent)
    half* __restrict__ sh_ah,
    uint32_t* __restrict__ sh_as,  // M regions of slice_stride words (+ M more
                                   // for residual)
    int slice_stride,
    float* __restrict__ sh_red,  // 33 floats
    int kb0, int nrows) {
  int t = threadIdx.x;
  int nel = nrows * 16;
#pragma unroll
  for (int r = 0; r < M; ++r) {
    uint32_t* as = sh_as + r * slice_stride;
    uint32_t* as2 = sh_as + (M + r) * slice_stride;
    float* qsr = qs + 4 * r;
    __syncthreads();
    if (r >= size_m) {
      for (int i = t; i < nel; i += NUM_THREADS) {
        as[i] = 0;
        if constexpr (residual) as2[i] = 0;
      }
      if (t == 0) {
        qsr[0] = 1.0f;
        ((int*)qsr)[1] = 0;
        ((int*)qsr)[2] = 0;
      }
      continue;
    }
    const half* Ar = A + (size_t)r * size_k;
    for (int sp = t >> 5; sp < (nel >> 7); sp += NUM_THREADS >> 5)
      had_hf_r_128_inner<true, false>(
          Ar + (kb0 << 4) + (sp << 7), sh_ah + (sp << 7),
          suh + (kb0 << 4) + (sp << 7), 0.088388347648f);
    __syncthreads();

    float mx = 0.0f;
    for (int i = t; i < nel; i += NUM_THREADS)
      mx = fmaxf(mx, fabsf(__half2float(sh_ah[i])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    if ((t & 31) == 0) sh_red[t >> 5] = mx;
    __syncthreads();
    if (t < 32) {
      float v = t < (NUM_THREADS >> 5) ? sh_red[t] : 0.0f;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
      if (t == 0) sh_red[32] = fmaxf(v, 1e-8f) / 127.0f;
    }
    __syncthreads();
    float q_s = sh_red[32];

    float rq = 1.0f / q_s;
    float rq2 = rq * 254.0f;
    int l1 = 0, l2 = 0;
    for (int i = t; i < nel; i += NUM_THREADS) {
      float a = __half2float(sh_ah[i]);
      int v = __float2int_rn(a * rq);
      v = max(-127, min(127, v));
      as[i] = ((uint32_t)(uint8_t)(int8_t)v) * 0x01010101u;
      l1 += v;
      if constexpr (residual) {
        float rr = a - q_s * (float)v;
        int v2 = __float2int_rn(rr * rq2);
        v2 = max(-127, min(127, v2));
        as2[i] = ((uint32_t)(uint8_t)(int8_t)v2) * 0x01010101u;
        l2 += v2;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      l1 += __shfl_xor_sync(0xffffffff, l1, o);
      l2 += __shfl_xor_sync(0xffffffff, l2, o);
    }
    if ((t & 31) == 0) {
      ((int*)sh_red)[t >> 5] = l1;
      sh_red[16 + (t >> 5)] = __int_as_float(l2);
    }
    __syncthreads();
    if (t < 32) {
      int v1 = t < (NUM_THREADS >> 5) ? ((int*)sh_red)[t] : 0;
      int v2 = t < (NUM_THREADS >> 5) ? __float_as_int(sh_red[16 + t]) : 0;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        v1 += __shfl_xor_sync(0xffffffff, v1, o);
        v2 += __shfl_xor_sync(0xffffffff, v2, o);
      }
      if (t == 0) {
        qsr[0] = sh_red[32];
        ((int*)qsr)[1] = v1;
        ((int*)qsr)[2] = v2;
      }
    }
  }
  __syncthreads();
}

// Epilogue for one 256-column group: deterministic fixed-order combine over the
// per-slice partials, warp per (row, 128-span). Reads bypass L1 (__ldcg): the
// contributions arrived from other blocks with no grid-wide barrier.
template <int M, bool c_fp32, bool residual>
__device__ __forceinline__ void gemv_int8_epilogue_group_sq(
    const int* __restrict__ partials, const float* __restrict__ qsums,
    int pstride, int ksplit, int size_m, void* __restrict__ C,
    const half* __restrict__ svh, float* __restrict__ sh_tmp, int nb256,
    int size_n) {
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = warp >> 1;
  if (warp >= 2 * M || row >= size_m) return;
  float k_inv = __half2float(__ushort_as_half(0x1eee));
  float k_bias = __half2float(__ushort_as_half(0xc931));
  float aff = 1024.0f * k_inv + k_bias;

  int base = nb256 * 256 + (warp & 1) * 128;
  float* tmp = sh_tmp + warp * 128;
  float acc[4] = {};
  float corr = 0.0f;
  for (int sl = 0; sl < ksplit; ++sl) {
    int idx = sl * M + row;
    float q_s = qsums[4 * idx];
    float suma = q_s * (float)((const int*)qsums)[4 * idx + 1];
    const int* p = partials + (size_t)idx * pstride + base;
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] += q_s * (float)__ldcg(p + lane * 4 + i);
    if constexpr (residual) {
      float q2_s = q_s * (1.0f / 254.0f);
      suma += q2_s * (float)((const int*)qsums)[4 * idx + 2];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc[i] += q2_s * (float)__ldcg(p + size_n + lane * 4 + i);
    }
    corr += aff * suma;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) tmp[lane * 4 + i] = k_inv * acc[i] + corr;
  __syncwarp();
  if constexpr (c_fp32)
    had_ff_r_128_inner<false, true>(tmp,
                                    ((float*)C) + (size_t)row * size_n + base,
                                    svh + base, 0.088388347648f);
  else
    had_fh_r_128_inner<false, true>(tmp,
                                    ((half*)C) + (size_t)row * size_n + base,
                                    svh + base, 0.088388347648f);
}

// Shared-memory cap for the sq decomposition: row halfs (32 B/row) + M splat
// regions (64 B/row each, x2 residual), within ~80 KB so the stage region and
// epilogue staging still fit under the opt-in max
__host__ __device__ constexpr int gemv_int8_sq_rows_max(int M, bool residual) {
  int cap = (80 * 1024) / (32 + 64 * M * (residual ? 2 : 1));
  cap &= ~7;
  return cap < SQ_ROWS_MAX ? cap : SQ_ROWS_MAX;
}

template <int bits, int M, bool c_fp32, bool residual>
__global__ __launch_bounds__(NUM_THREADS) void exl3_gemv_int8_sq_kernel(
    const half* __restrict__ A, const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,  // 1 <= size_m <= M
    const int size_k, const int size_n, int* __restrict__ locks,
    const half* __restrict__ suh, half* __restrict__ A_had,
    const half* __restrict__ svh) {
  extern __shared__ uint32_t shmem[];

  // Work decomposition: rows_per multiple of 8 (whole 128-spans for the local
  // Hadamard); the host mirrors this for the shared memory size and workspace
  // bound. Swept on 3090: the wall time is ~one unit's duration as long as all
  // (slice x 256-column) units fit in one wave of the grid, so take the
  // smallest slice height with units <= gridDim.x - but at least half-wave
  // occupancy up to 32 rows, since very short slices inflate ksplit and the
  // epilogue's serial per-slice combine
  int rows_total = size_k >> 4;
  int nb256_total = size_n / 256;
  int r = CEIL_DIVIDE(rows_total * nb256_total, (int)gridDim.x);
  int rows_per = (MAX(r, MIN(2 * r, 32)) + 7) & ~7;
  rows_per = MAX(rows_per, SQ_MINROWS);
  rows_per = MIN(rows_per, gemv_int8_sq_rows_max(M, residual));
  rows_per = MIN(rows_per, (rows_total + 7) & ~7);
  int ksplit = CEIL_DIVIDE(rows_total, rows_per);
  int units = nb256_total * ksplit;
  int slice_stride = rows_per * 16;
  int pstride = size_n * (residual ? 2 : 1);

  int* counters = locks;
  float* qsums = (float*)(locks + SQ_COUNTERS_CAP);
  int* partials = locks + SQ_WS_RESERVED;

  // Shared layout: [slice halfs][M x splats (+ M x residual splats)][B
  // stage][epilogue tmp]
  half* sh_ah = (half*)shmem;
  uint32_t* sh_as = shmem + rows_per * 8;
  uint32_t* sh_b = sh_as + slice_stride * M * (residual ? 2 : 1);
  float* sh_tmp =
      (float*)(sh_b +
               (gemv_int8_stage_smem(bits) ? 8 * GEMV_STAGE_D * 16 * bits : 0));
  __shared__ float sh_red[33];
  __shared__ int sh_last;

  int t = threadIdx.x;
  int prev_slice = -1;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    int slice = unit / nb256_total;
    int nb256 = unit % nb256_total;
    int kb0 = slice * rows_per;
    int nrows = MIN(rows_per, rows_total - kb0);
    if (slice != prev_slice) {
      gemv_int8_stage_slice<M, residual>(A, size_m, size_k, suh,
                                         qsums + 4 * slice * M, sh_ah, sh_as,
                                         slice_stride, sh_red, kb0, nrows);
      prev_slice = slice;
    }
    int* pacc = partials + (size_t)slice * M * pstride;
    if constexpr (bits == 4)
      gemv_int8_unit_wide<M, residual, false>(
          B, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);
    else if constexpr (gemv_int8_stage_smem(bits))
      gemv_int8_unit_smem<bits, M, residual, false>(B, pacc, pstride, sh_as,
                                                    slice_stride, sh_b, nb256,
                                                    kb0, nrows, size_n);
    else
      gemv_int8_unit_narrow<bits, M, residual, false>(
          B, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);

    // Completion counter: the ksplit-th contributor runs the epilogue for this
    // 256-column group. (sh_last reuse across iterations is ordered by the next
    // iteration's __syncthreads.)
    __threadfence();
    __syncthreads();
    if (t == 0)
      sh_last = (atomicAdd(&counters[nb256], 1) == ksplit - 1) ? 1 : 0;
    __syncthreads();
    if (sh_last) {
      gemv_int8_epilogue_group_sq<M, c_fp32, residual>(partials, qsums, pstride,
                                                       ksplit, size_m, C, svh,
                                                       sh_tmp, nb256, size_n);
      if (t == 0) counters[nb256] = 0;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------
// Cooperative kernel: same argument list as exl3_gemm_kernel

template <int bits, bool c_fp32, bool residual>
__global__ __launch_bounds__(NUM_THREADS) void exl3_gemv_int8_coop_kernel(
    const half* __restrict__ A, const uint16_t* __restrict__ B,
    void* __restrict__ C, const int size_m, const int size_k, const int size_n,
    int* __restrict__ locks, const half* __restrict__ suh,
    half* __restrict__ A_had, const half* __restrict__ svh) {
  auto grid = cg_gemv::this_grid();
  extern __shared__ uint32_t shmem[];

  int* accs = locks;
  float* qsums = (float*)(locks + 2 * size_n);
  float* partial_max = qsums + 4 * size_m;  // [gridDim.x], m == 1 fast path

  // Phase 1a: input Hadamard (as in exl3_gemm_kernel) + zero the accumulators.
  // For m == 1, also track a per-block partial max of the transformed
  // activations.
  {
    __shared__ float sh_m[32];
    int total_warps = size_m * size_k / 128;
    int warps_grid = gridDim.x * NUM_THREADS / 32;
    int this_warp = threadIdx.x / 32 + NUM_THREADS / 32 * blockIdx.x;
    int lane = threadIdx.x & 31;
    float mx = 0.0f;
    for (; this_warp < total_warps; this_warp += warps_grid) {
      had_hf_r_128_inner<true, false>(
          A + this_warp * 128, A_had + this_warp * 128,
          suh + (this_warp * 128) % size_k, 0.088388347648f);
      if (size_m == 1) {
        __syncwarp();
        const half2* h2 = (const half2*)(A_had + this_warp * 128);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          half2 v = h2[lane * 2 + i];
          mx = fmaxf(mx, fabsf(__half2float(__low2half(v))));
          mx = fmaxf(mx, fabsf(__half2float(__high2half(v))));
        }
      }
    }
    if (size_m == 1) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
      if (lane == 0) sh_m[threadIdx.x >> 5] = mx;
      __syncthreads();
      if (threadIdx.x < 32) {
        float v = threadIdx.x < (NUM_THREADS >> 5) ? sh_m[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
          v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        if (threadIdx.x == 0) partial_max[blockIdx.x] = v;
      }
    }
    int tg = blockIdx.x * NUM_THREADS + threadIdx.x;
    for (int i = tg; i < 2 * size_n; i += gridDim.x * NUM_THREADS) accs[i] = 0;
  }
  grid.sync();

  // Phase 1b (m > 1 only): per-row activation scale and exact sums
  if (size_m > 1) {
    __shared__ float sh_r[33];
    for (int row = blockIdx.x; row < size_m; row += gridDim.x) {
      const half* Ar = A_had + (size_t)row * size_k;
      int t = threadIdx.x;
      float mx = 0.0f;
      for (int i = t; i < size_k; i += NUM_THREADS)
        mx = fmaxf(mx, fabsf(__half2float(Ar[i])));
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
      if ((t & 31) == 0) sh_r[t >> 5] = mx;
      __syncthreads();
      if (t < 32) {
        float v = t < (NUM_THREADS >> 5) ? sh_r[t] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
          v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        if (t == 0) sh_r[32] = fmaxf(v, 1e-8f) / 127.0f;
      }
      __syncthreads();
      gemv_int8_row_sums<residual>(Ar, sh_r[32], size_k, qsums + 4 * row);
    }
    grid.sync();
  }

  // Work decomposition for phase 2 (host must mirror this for the shared memory
  // size). Units are 256 columns (warp per adjacent block pair) x k-slice;
  // fine-grained (~4 per block) because with grid.sync at the end of the phase,
  // tail imbalance stalls the whole grid.
  int rows_total = size_k >> 4;
  int nb256_total = size_n / 256;
  int smem_rows_max = residual ? 384 : 768;
  int ksplit = CEIL_DIVIDE(4 * gridDim.x, nb256_total);
  ksplit = MAX(ksplit, CEIL_DIVIDE(rows_total, smem_rows_max));
  ksplit = MIN(ksplit, rows_total);
  int rows_per = CEIL_DIVIDE(rows_total, ksplit);
  int units = nb256_total * ksplit;

  uint32_t* sh_as = shmem;
  uint32_t* sh_as2 = shmem + rows_per * 16;
  uint32_t* sh_b =
      shmem + rows_per * 16 * (residual ? 2 : 1);  // smem-staged unit region

  __shared__ float sh_q[33];

  for (int row = 0; row < size_m; ++row) {
    const half* Ar = A_had + (size_t)row * size_k;
    const float* qrow = qsums + 4 * row;

    // Row scale: for m == 1, derived from the phase-1a partial maxes
    // (deterministic: fmax is order-independent). Block 0 also computes the
    // exact sums for the epilogue, concurrently with the other blocks' GEMV
    // work.
    float q_row;
    if (size_m == 1) {
      float v = 0.0f;
      for (int i = threadIdx.x; i < gridDim.x; i += NUM_THREADS)
        v = fmaxf(v, partial_max[i]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
      if ((threadIdx.x & 31) == 0) sh_q[threadIdx.x >> 5] = v;
      __syncthreads();
      if (threadIdx.x < 32) {
        float w = threadIdx.x < (NUM_THREADS >> 5) ? sh_q[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
          w = fmaxf(w, __shfl_xor_sync(0xffffffff, w, o));
        if (threadIdx.x == 0) sh_q[32] = fmaxf(w, 1e-8f) / 127.0f;
      }
      __syncthreads();
      q_row = sh_q[32];
      if (blockIdx.x == 0)
        gemv_int8_row_sums<residual>(Ar, q_row, size_k, qsums);
    } else
      q_row = qrow[0];

    // Phase 2: GEMV over (nb256, k-slice) units, splats quantized inline during
    // smem staging
    int prev_slice = -1;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
      int slice = unit / nb256_total;
      int nb256 = unit % nb256_total;
      int kb0 = slice * rows_per;
      int nrows = MIN(rows_per, rows_total - kb0);
      if (slice != prev_slice) {
        gemv_int8_stage_splats<residual>(Ar, q_row, sh_as, sh_as2, kb0, nrows);
        prev_slice = slice;
      }
      if constexpr (bits == 4)
        gemv_int8_unit_wide<1, residual>(B, accs, 0, sh_as, rows_per * 16,
                                         nb256, kb0, nrows, size_n);
      else if constexpr (gemv_int8_stage_smem(bits))
        gemv_int8_unit_smem<bits, 1, residual>(B, accs, 0, sh_as, rows_per * 16,
                                               sh_b, nb256, kb0, nrows, size_n);
      else
        gemv_int8_unit_narrow<bits, 1, residual>(
            B, accs, 0, sh_as, rows_per * 16, nb256, kb0, nrows, size_n);
    }
    grid.sync();

    // Phase 3: epilogue + output Hadamard (shared memory reused as per-warp
    // staging)
    void* C_row = c_fp32 ? (void*)(((float*)C) + (size_t)row * size_n)
                         : (void*)(((half*)C) + (size_t)row * size_n);
    gemv_int8_epilogue<c_fp32, residual>(accs, C_row, qrow, svh, (float*)shmem,
                                         size_n, 1.0f);

    if (row + 1 < size_m) grid.sync();
  }
}
