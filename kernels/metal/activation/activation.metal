#include <metal_stdlib>
using namespace metal;

struct GatedParams {
  uint d;
  uint num_tokens;
  uint op_id;      // 0: silu_and_mul, 1: mul_and_silu, 2: gelu_and_mul, 3: gelu_tanh_and_mul, 4: fatrelu_and_mul
  float threshold; // only for fatrelu
};

struct ElemParams {
  uint d;
  uint num_tokens;
  uint op_id; // 0: gelu_new, 1: gelu_fast, 2: gelu_quick
};

inline float silu(float x) {
  return x / (1.0f + exp(-x));
}

// Fast erf approximation (Abramowitz & Stegun 7.1.26)
inline float erf_approx(float x) {
  const float a1 = 0.254829592f;
  const float a2 = -0.284496736f;
  const float a3 = 1.421413741f;
  const float a4 = -1.453152027f;
  const float a5 = 1.061405429f;
  const float p  = 0.3275911f;
  float sign = x < 0.0f ? -1.0f : 1.0f;
  float ax = fabs(x);
  float t = 1.0f / (1.0f + p * ax);
  float y = 1.0f - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * exp(-ax * ax);
  return sign * y;
}

inline float gelu_none(float x) {
  const float inv_sqrt2 = 0.7071067811865475244f; // 1/sqrt(2)
  return 0.5f * x * (1.0f + erf_approx(x * inv_sqrt2));
}

inline float gelu_tanh(float x) {
  const float beta = 0.79788456f;
  const float kappa = 0.044715f;
  float x3 = x * x * x;
  float inner = beta * (x + kappa * x3);
  return 0.5f * x * (1.0f + tanh(inner));
}

inline float gelu_new(float x) {
  const float w1 = 0.79788456f;
  const float w2 = 0.044715f;
  float x3 = x * x * x;
  float t = tanh(w1 * (x + w2 * x3));
  return 0.5f * x * (1.0f + t);
}

inline float gelu_fast(float x) {
  const float w1 = 0.79788456f;
  const float w2 = 0.044715f;
  float t = tanh((x * w1) * (1.0f + x * w2 * x));
  return 0.5f * x * (1.0f + t);
}

inline float gelu_quick(float x) {
  return x / (1.0f + exp(-1.702f * x));
}

kernel void activation_gated_float(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant GatedParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = p.num_tokens * p.d;
  if (gid >= total) return;
  uint token = gid / p.d;
  uint idx = gid - token * p.d;
  uint base = token * (p.d * 2u);
  float x = in[base + idx];
  float y = in[base + p.d + idx];
  float z;
  switch (p.op_id) {
    case 0: z = silu(x) * y; break;
    case 1: z = x * silu(y); break;
    case 2: z = gelu_none(x) * y; break;
    case 3: z = gelu_tanh(x) * y; break;
    case 4: z = (x > p.threshold ? x : 0.0f) * y; break;
    default: z = 0.0f; break;
  }
  out[token * p.d + idx] = z;
}

kernel void activation_gated_half(
    device const half* in [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant GatedParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = p.num_tokens * p.d;
  if (gid >= total) return;
  uint token = gid / p.d;
  uint idx = gid - token * p.d;
  uint base = token * (p.d * 2u);
  float x = float(in[base + idx]);
  float y = float(in[base + p.d + idx]);
  float z;
  switch (p.op_id) {
    case 0: z = silu(x) * y; break;
    case 1: z = x * silu(y); break;
    case 2: z = gelu_none(x) * y; break;
    case 3: z = gelu_tanh(x) * y; break;
    case 4: z = (x > p.threshold ? x : 0.0f) * y; break;
    default: z = 0.0f; break;
  }
  out[token * p.d + idx] = half(z);
}

kernel void activation_elem_float(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant ElemParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = p.num_tokens * p.d;
  if (gid >= total) return;
  uint token = gid / p.d;
  uint idx = gid - token * p.d;
  float x = in[token * p.d + idx];
  float z;
  switch (p.op_id) {
    case 0: z = gelu_new(x); break;
    case 1: z = gelu_fast(x); break;
    case 2: z = gelu_quick(x); break;
    default: z = 0.0f; break;
  }
  out[token * p.d + idx] = z;
}

kernel void activation_elem_half(
    device const half* in [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant ElemParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = p.num_tokens * p.d;
  if (gid >= total) return;
  uint token = gid / p.d;
  uint idx = gid - token * p.d;
  float x = float(in[token * p.d + idx]);
  float z;
  switch (p.op_id) {
    case 0: z = gelu_new(x); break;
    case 1: z = gelu_fast(x); break;
    case 2: z = gelu_quick(x); break;
    default: z = 0.0f; break;
  }
  out[token * p.d + idx] = half(z);
}


