#pragma once

// This used to force integer MAD on sm_86 via inline asm, which outperformed
// the IMUL emitted by older nvcc versions on the RTX 3090. As of CUDA 13.2 the
// workaround has inverted: the plain multiply is ~4% faster end-to-end at m=1.
// Kept as a hook in case it regresses again.
template <uint32_t w>
__device__ __forceinline__ uint32_t mul_const_u32(uint32_t x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)
  uint32_t r;
  asm volatile(
      "{ .reg .u32 z,t;"
      "  mov.u32 t, %laneid;"  // runtime SR
      "  sub.u32 z, t, t;"     // z = 0 but data-dependent
      "  mad.lo.u32 %0, %1, %2, z;"
      "}"
      : "=r"(r)
      : "r"(x), "n"(w));
  return r;
#else
  return x * w;
#endif
}

// Decode two mul1 (cb 2) codebook entries from precomputed products x0 = idx0 *
// 0x83DCD12D, x1 = idx1 * 0x83DCD12D
__device__ inline half2 decode_mul1_product_2(uint32_t x0, uint32_t x1) {
  const uint32_t acc = 0x6400u;  // 0x6400 -> 1024.0 ..  0x67FF -> 2047.0
  // uint32_t sum0;
  // uint32_t sum1;
  // asm ("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(sum0) : "r"(x0),
  // "r"(0), "r"(acc) : ); asm ("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" :
  // "=r"(sum1) : "r"(x1), "r"(0), "r"(acc) : );
  uint32_t sum0 = __dp4a(x0, 0x01010101u, acc);
  uint32_t sum1 = __dp4a(x1, 0x01010101u, acc);
  half2 k_inv_h2 =
      __half2half2(__ushort_as_half(0x1eee));  //  0.00677 = 1/147.7
  half2 k_bias_h2 = __half2half2(
      __ushort_as_half(0xc931));  // -10.39 = (-1024.0 - 510.0) * k_inv_h
  half_uint16 h0((uint16_t)sum0);
  half_uint16 h1((uint16_t)sum1);
  return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
}

// Ditto mcg (cb 1)
__device__ inline half2 decode_mcg_product_2(uint32_t x0, uint32_t x1) {
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
  half2_uint32 xu0(x0);
  half2_uint32 xu1(x1);
  half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
  half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
  return __hadd2(d0, d1);
}

template <int cb>
__device__ inline half decode_3inst(uint32_t x) {
  if constexpr (cb == 0) {
    x *= 89226354u;
    x += 64248484u;
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x));
    half2_uint32 xu(x);
    return __hadd(__low2half(xu.as_half2), __high2half(xu.as_half2));
  }
  if constexpr (cb == 1) {
    x *= 0xCBAC1FEDu;
    // x = mul_const_u32<0xCBAC1FEDu>(x);

    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x));
    half2_uint32 xu(x);
    return __hadd(__low2half(xu.as_half2), __high2half(xu.as_half2));
  }
  if constexpr (cb == 2) {
    x *= 0x83DCD12Du;
    const uint32_t acc = 0x6400u;  // 0x6400 -> 1024.0 ..  0x67FF -> 2047.0
    // Byte sum via dp4a, bit-identical to the previous vabsdiff4(x, 0, acc) but
    // native on Blackwell where vabsdiff4 is emulated. dp4a also wins om Ampere
    // now, possibly after compiler changes, and ties on Ada uint32_t sum; asm
    // ("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(sum) : "r"(x),
    // "r"(0), "r"(acc) : );
    uint32_t sum = __dp4a(x, 0x01010101u, acc);
    const __half k_inv_h = __ushort_as_half(0x1eee);  //  0.00677 = 1/147.7
    const __half k_bias_h =
        __ushort_as_half(0xc931);  // -10.39 = (-1024.0 - 510.0) * k_inv_h
    half_uint16 h((uint16_t)sum);
    return __hfma(h.as_half, k_inv_h, k_bias_h);
  }
}

template <int cb>
__device__ inline half2 decode_3inst_2(uint32_t x0, uint32_t x1) {
  if constexpr (cb == 0) {
    x0 *= 89226354u;
    x1 *= 89226354u;
    x0 += 64248484u;
    x1 += 64248484u;
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
    half2_uint32 xu0(x0);
    half2_uint32 xu1(x1);
    half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
    half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
    return __hadd2(d0, d1);
  }
  if constexpr (cb == 1) {
    // x0 = mul_const_u32<0xCBAC1FEDu>(x0);
    // x1 = mul_const_u32<0xCBAC1FEDu>(x1);
    x0 *= 0xCBAC1FEDu;
    x1 *= 0xCBAC1FEDu;
    return decode_mcg_product_2(x0, x1);
  }
  if constexpr (cb == 2) {
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    return decode_mul1_product_2(x0, x1);
  }
}

template <int cb>
__device__ inline float decode_3inst_f(uint64_t x) {
  return __half2float(decode_3inst<cb>(x));
}

template <int cb>
__device__ inline float decode_3inst_f_diff(uint64_t x, float d) {
  return __half2float(decode_3inst<cb>(x)) - d;
}

// "2MAD" procedural codebook, much more overhead than 3INST, slightly better
// distribution at 2bpw Not used currently

//__device__ inline half decode_2mad(uint64_t x)
//{
//    x = x * 264435761u + 1013904223u;
//    x = ((x * 1664525u) >> 32) + x;
//    int32_t c = (int32_t) __dp4a((uint32_t) x, 0x01010101u, 0xFFFFFE02u);
//    half y = __hmul(__int2half_rn(c), __float2half_rn(0.008415));
//    return y;
//}
//
//__device__ inline float decode_2mad_f(uint64_t x)
//{
//    x = x * 264435761u + 1013904223u;
//    x = ((x * 1664525u) >> 32) + x;
//    int32_t c = (int32_t) __dp4a((uint32_t) x, 0x01010101u, 0xFFFFFE02u);
//    float y = __int2float_rn(c) * 0.008415f;
//    return y;
//}
//
//__device__ inline float decode_2mad_f_diff(uint64_t x, float d)
//{
//    x = x * 264435761u + 1013904223u;
//    x = ((x * 1664525u) >> 32) + x;
//    int32_t c = (int32_t) __dp4a((uint32_t) x, 0x01010101u, 0xFFFFFE02u);
//    float y = fma(__int2float_rn(c), 0.008415f, -d);
//    return y;
//}
