#pragma once

#include "attention_generic.cuh"

#include <cuComplex.h>

namespace aphrodite {

// Define custom complex64 vector data types.
struct Complex4_ {
  cuFloatComplex x;
  cuFloatComplex y;
};

struct Complex8_ {
  cuFloatComplex x;
  cuFloatComplex y;
  cuFloatComplex z;
  cuFloatComplex w;
};

// Complex64 vector types for Q, K, V.
template <>
struct Vec<cuFloatComplex, 1> {
  using Type = cuFloatComplex;
};
template <>
struct Vec<cuFloatComplex, 2> {
  using Type = Complex4_;
};
template <>
struct Vec<cuFloatComplex, 4> {
  using Type = Complex8_;
};

// Complex64 accumulator vector types corresponding to Vec.
template <>
struct FloatVec<cuFloatComplex> {
  using Type = cuFloatComplex;
};
template <>
struct FloatVec<Complex4_> {
  using Type = Complex4_;
};
template <>
struct FloatVec<Complex8_> {
  using Type = Complex8_;
};

// Vector addition.
inline __device__ cuFloatComplex add(cuFloatComplex a, cuFloatComplex b) {
  return cuCaddf(a, b);
}

inline __device__ Complex4_ add(Complex4_ a, Complex4_ b) {
  Complex4_ c;
  c.x = cuCaddf(a.x, b.x);
  c.y = cuCaddf(a.y, b.y);
  return c;
}

inline __device__ Complex8_ add(Complex8_ a, Complex8_ b) {
  Complex8_ c;
  c.x = cuCaddf(a.x, b.x);
  c.y = cuCaddf(a.y, b.y);
  c.z = cuCaddf(a.z, b.z);
  c.w = cuCaddf(a.w, b.w);
  return c;
}

// Vector multiplication.
template <>
inline __device__ cuFloatComplex mul(cuFloatComplex a, cuFloatComplex b) {
  return cuCmulf(a, b);
}

template <>
inline __device__ Complex4_ mul(Complex4_ a, Complex4_ b) {
  Complex4_ c;
  c.x = cuCmulf(a.x, b.x);
  c.y = cuCmulf(a.y, b.y);
  return c;
}

template <>
inline __device__ Complex4_ mul(cuFloatComplex a, Complex4_ b) {
  Complex4_ c;
  c.x = cuCmulf(a, b.x);
  c.y = cuCmulf(a, b.y);
  return c;
}

template <>
inline __device__ Complex8_ mul(Complex8_ a, Complex8_ b) {
  Complex8_ c;
  c.x = cuCmulf(a.x, b.x);
  c.y = cuCmulf(a.y, b.y);
  c.z = cuCmulf(a.z, b.z);
  c.w = cuCmulf(a.w, b.w);
  return c;
}

template <>
inline __device__ Complex8_ mul(cuFloatComplex a, Complex8_ b) {
  Complex8_ c;
  c.x = cuCmulf(a, b.x);
  c.y = cuCmulf(a, b.y);
  c.z = cuCmulf(a, b.z);
  c.w = cuCmulf(a, b.w);
  return c;
}

// Vector fused multiply-add.
inline __device__ cuFloatComplex fma(cuFloatComplex a, cuFloatComplex b,
                                     cuFloatComplex c) {
  return cuCfmaf(a, b, c);
}

inline __device__ Complex4_ fma(Complex4_ a, Complex4_ b, Complex4_ c) {
  Complex4_ d;
  d.x = cuCfmaf(a.x, b.x, c.x);
  d.y = cuCfmaf(a.y, b.y, c.y);
  return d;
}

inline __device__ Complex4_ fma(cuFloatComplex a, Complex4_ b, Complex4_ c) {
  Complex4_ d;
  d.x = cuCfmaf(a, b.x, c.x);
  d.y = cuCfmaf(a, b.y, c.y);
  return d;
}

inline __device__ Complex8_ fma(Complex8_ a, Complex8_ b, Complex8_ c) {
  Complex8_ d;
  d.x = cuCfmaf(a.x, b.x, c.x);
  d.y = cuCfmaf(a.y, b.y, c.y);
  d.z = cuCfmaf(a.z, b.z, c.z);
  d.w = cuCfmaf(a.w, b.w, c.w);
  return d;
}

inline __device__ Complex8_ fma(cuFloatComplex a, Complex8_ b, Complex8_ c) {
  Complex8_ d;
  d.x = cuCfmaf(a, b.x, c.x);
  d.y = cuCfmaf(a, b.y, c.y);
  d.z = cuCfmaf(a, b.z, c.z);
  d.w = cuCfmaf(a, b.w, c.w);
  return d;
}

template <>
inline __device__ cuFloatComplex sum(cuFloatComplex v) {
  return v;
}

template <>
inline __device__ Complex4_ sum(Complex4_ v) {
  Complex4_ acc;
  acc.x = cuCaddf(v.x, v.y);
  acc.y = make_cuFloatComplex(0.f, 0.f);
  return acc;
}

template <>
inline __device__ Complex8_ sum(Complex8_ v) {
  Complex4_ acc1;
  Complex4_ acc2;
  acc1.x = cuCaddf(v.x, v.y);
  acc1.y = cuCaddf(v.z, v.w);
  acc2.x = make_cuFloatComplex(0.f, 0.f);
  acc2.y = make_cuFloatComplex(0.f, 0.f);
  return add(acc1, acc2);
}

inline __device__ cuFloatComplex dot(cuFloatComplex a, cuFloatComplex b) {
  return cuCmulf(a, b);
}

inline __device__ Complex4_ dot(Complex4_ a, Complex4_ b) {
  Complex4_ c;
  c.x = cuCmulf(a.x, b.x);
  c.y = cuCmulf(a.y, b.y);
  return c;
}

inline __device__ Complex8_ dot(Complex8_ a, Complex8_ b) {
  Complex8_ c;
  c.x = cuCmulf(a.x, b.x);
  c.y = cuCmulf(a.y, b.y);
  c.z = cuCmulf(a.z, b.z);
  c.w = cuCmulf(a.w, b.w);
  return c;
}

inline __device__ void from_float(cuFloatComplex& dst, cuFloatComplex src) {
  dst = src;
}

inline __device__ void from_float(Complex4_& dst, Complex4_ src) { dst = src; }

inline __device__ void from_float(Complex8_& dst, Complex8_ src) { dst = src; }

inline __device__ cuFloatComplex to_float(cuFloatComplex u) { return u; }

inline __device__ Complex4_ to_float(Complex4_ u) { return u; }

inline __device__ Complex8_ to_float(Complex8_ u) { return u; }

}  // namespace aphrodite
