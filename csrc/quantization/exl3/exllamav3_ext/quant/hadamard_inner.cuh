#pragma once
#include "../compat.cuh"

#define ACT_SILU 0
#define ACT_GELU 1

// Hadamard transform 128-element vector across one warp, with optional pre and post scales

__device__ inline half hreduce(half2 x)
{
    return __hadd(__low2half(x), __high2half(x));
}

__device__ inline void shuffle_had_f4x32(float& h0, float& h1, float& h2, float& h3, const int lane_id)
{
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        uint32_t i0 = __float_as_uint(h0);
        uint32_t i1 = __float_as_uint(h1);
        uint32_t i2 = __float_as_uint(h2);
        uint32_t i3 = __float_as_uint(h3);
        uint64_t h01 =  (uint64_t) i0 | (((uint64_t) i1) << 32);
        uint64_t h23 =  (uint64_t) i2 | (((uint64_t) i3) << 32);
        uint64_t ph01 = __shfl_xor_sync(0xffffffff, h01, i);
        uint64_t ph23 = __shfl_xor_sync(0xffffffff, h23, i);
        float ph0 = __uint_as_float((uint32_t) (ph01 & 0xffffffff));
        float ph1 = __uint_as_float((uint32_t) (ph01 >> 32));
        float ph2 = __uint_as_float((uint32_t) (ph23 & 0xffffffff));
        float ph3 = __uint_as_float((uint32_t) (ph23 >> 32));
        int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
        i0 ^= sfm & 0x80000000;
        i1 ^= sfm & 0x80000000;
        i2 ^= sfm & 0x80000000;
        i3 ^= sfm & 0x80000000;
        h0 = __uint_as_float(i0) + ph0;
        h1 = __uint_as_float(i1) + ph1;
        h2 = __uint_as_float(i2) + ph2;
        h3 = __uint_as_float(i3) + ph3;
    }
}

__device__ inline void shuffle_had_f2x32(float& v, float& w, const int lane_id)
{
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        uint64_t vw = ((uint64_t) __float_as_uint(v)) | (((uint64_t) __float_as_uint(w)) << 32);
        uint64_t pvw = __shfl_xor_sync(0xffffffff, vw, i);
        float pv = __uint_as_float((uint32_t) (pvw & 0xffffffff));
        float pw = __uint_as_float((uint32_t) (pvw >> 32));
        uint32_t vi = __float_as_uint(v);
        uint32_t wi = __float_as_uint(w);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        vi ^= (sfm & 0x80000000);
        wi ^= (sfm & 0x80000000);
        v = __uint_as_float(vi) + pv;
        w = __uint_as_float(wi) + pw;
    }
}

__device__ inline float shuffle_had_fx32(float v, const int lane_id)
{
    for (int i = 1; i < 32; i <<= 1)
    {
        float pv = __shfl_xor_sync(0xffffffff, v, i);
        uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        *vi ^= (sfm & 0x80000000);
        v = v + pv;
    }
    return v;
}

__device__ inline half2 shuffle_had_h2x32(half2 v, int lane_id)
{
    for (int i = 1; i < 32; i <<= 1)
    {
        half2 pv = __shfl_xor_sync(0xffffffff, v, i);
        uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        *vi ^= (sfm & 0x80008000);
        v = __hadd2(v, pv);
    }
    return v;
}

// Half vector, half scales

inline __device__
void had_hf_r_128_inner
(
    const half* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
    const half* __restrict__ pre_scale,
    const half* __restrict__ post_scale,
    const float r_scale
)
{
    int t = threadIdx.x & 31;

    // Load
    half4 v = ((half4*) input_ptr)[t];

    // Pre scale
    if (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) pre_scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }

    // 4 element had
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    // 32 element had, warp shuffle
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

    // Post scale
    if (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) post_scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }

    // Store
    ((half4*) output_ptr)[t] = v;
}

// Float vector, half scales

inline __device__
void had_ff_r_128_inner
(
    const float* __restrict__ input_ptr,
    float* __restrict__ output_ptr,
    const half* __restrict__ pre_scale,
    const half* __restrict__ post_scale,
    const float r_scale
)
{
    int t = threadIdx.x & 31;

    // Load
    float4 v = ((float4*) input_ptr)[t];

    // Pre scale
    if (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) pre_scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }

    // 4 element had
    float v0 = v.x;
    float v1 = v.y;
    float v2 = v.z;
    float v3 = v.w;
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;

    // 32 element had, warp shuffle
    shuffle_had_f2x32(v.x, v.y, t);
    shuffle_had_f2x32(v.z, v.w, t);
    v.x *= r_scale;
    v.y *= r_scale;
    v.z *= r_scale;
    v.w *= r_scale;

    // Post scale
    if (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) post_scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }

    // Store
    ((float4*) output_ptr)[t] = v;
}

// Fused op: o <- in_had(silu(out_had(g)) * out_had(u))

inline __device__
void had_hf_r_128_guad_inner
(
    const half* __restrict__ input_ptr_g,
    const half* __restrict__ input_ptr_u,
    half* __restrict__ output_ptr,
    const half* __restrict__ post_scale_g,
    const half* __restrict__ post_scale_u,
    const half* __restrict__ pre_scale_d,
    const float r_scale,
    const float act_limit,
    const int act_function
)
{
    int t = threadIdx.x & 31;

    auto had = [&](half4& v)
    {
        // 4 element had
        float v0 = __half2float(__low2half(v.x));
        float v1 = __half2float(__high2half(v.x));
        float v2 = __half2float(__low2half(v.y));
        float v3 = __half2float(__high2half(v.y));
        float s0 = v0 + v1;
        float d0 = v0 - v1;
        float s1 = v2 + v3;
        float d1 = v2 - v3;
        float h0 = s0 + s1;
        float h1 = d0 + d1;
        float h2 = s0 - s1;
        float h3 = d0 - d1;

        // 32 element had, warp shuffle
        shuffle_had_f4x32(h0, h1, h2, h3, t);
        v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
        v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

        return v;
    };

    auto _silu = [&](const half2& x)
    {
        half2 one = __float2half2_rn(1.0f);
        half2 neg_x = __hneg2(x);
        half2 e = h2exp(neg_x);
        half2 sum = __hadd2(one, e);
        half2 r = h2rcp(sum);
        half2 result = __hmul2(x, r);
        return result;
    };

    auto _gelu = [&](const half2& x)
    {
        float2 xf = __half22float2(x);
        const float c = 0.797884560803f;  // sqrt(2/Pi)
        xf.x = 0.5f * xf.x * (1.0f + tanh_opt(c * (xf.x + 0.044715f * xf.x * xf.x * xf.x)));
        xf.y = 0.5f * xf.y * (1.0f + tanh_opt(c * (xf.y + 0.044715f * xf.y * xf.y * xf.y)));
        return __float22half2_rn(xf);
    };

    // Load
    half4 vg = ((half4*) input_ptr_g)[t];
    half4 vu = ((half4*) input_ptr_u)[t];

    // Hadamard
    vg = had(vg);
    vu = had(vu);

    // Post scale  TODO: should maybe do this in float32
    int i = blockIdx.y * 32 + t;
    half4 scales_g = ((half4*) post_scale_g)[i];
    half4 scales_u = ((half4*) post_scale_u)[i];
    vg.x = __hmul2(vg.x, scales_g.x);
    vg.y = __hmul2(vg.y, scales_g.y);
    vu.x = __hmul2(vu.x, scales_u.x);
    vu.y = __hmul2(vu.y, scales_u.y);

    // Activation
    switch (act_function)
    {
        case ACT_SILU:
            vg.x = _silu(vg.x);
            vg.y = _silu(vg.y);
            break;

        case ACT_GELU:
            vg.x = _gelu(vg.x);
            vg.y = _gelu(vg.y);
            break;

        default:
            break;
    }

    // Optional activation limits
    if (act_limit != 0.0f)
    {
        vu.x = __hmax2(vu.x, __float2half2_rn(-act_limit));
        vu.y = __hmax2(vu.y, __float2half2_rn(-act_limit));
        vu.x = __hmin2(vu.x, __float2half2_rn(act_limit));
        vu.y = __hmin2(vu.y, __float2half2_rn(act_limit));
        vg.x = __hmin2(vg.x, __float2half2_rn(act_limit));
        vg.y = __hmin2(vg.y, __float2half2_rn(act_limit));
    }

    // Gate
    vg.x = __hmul2(vg.x, vu.x);
    vg.y = __hmul2(vg.y, vu.y);

    // Pre scale (d)
    half4 scales_d = ((half4*) pre_scale_d)[i];
    vg.x = __hmul2(vg.x, scales_d.x);
    vg.y = __hmul2(vg.y, scales_d.y);

    // Hadamard
    vg = had(vg);

    // Store
    ((half4*) output_ptr)[t] = vg;
}

// Fused op: o += float(out_had(i)), atomic

inline __device__
void had_hf_r_128_d_inner
(
    const half* __restrict__ input_ptr,
    float* __restrict__ output_ptr,
    const half* __restrict__ post_scale,
    const float r_scale
)
{
    int t = threadIdx.x & 31;

    // Load
    half4 v = ((half4*) input_ptr)[t];

    // 4 element had
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    // 32 element had, warp shuffle
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    h0 *= r_scale;
    h1 *= r_scale;
    h2 *= r_scale;
    h3 *= r_scale;

    // Post scale
    int i = blockIdx.y * 32 + t;
    half4 scales = ((half4*) post_scale)[i];
    h0 *= __low2float(scales.x);
    h1 *= __high2float(scales.x);
    h2 *= __low2float(scales.y);
    h3 *= __high2float(scales.y);

    // Reshuffle in shmem for coalesced store with atomicAdd
    extern __shared__ float temp_shared[];
    int warp_id = threadIdx.x / 32;
    float* sh = temp_shared + warp_id * 128;
    sh[t * 4 + 0] = h0;
    sh[t * 4 + 1] = h1;
    sh[t * 4 + 2] = h2;
    sh[t * 4 + 3] = h3;
    __syncwarp();
    atomicAdd(output_ptr +  0 + t, sh[ 0 + t]);
    atomicAdd(output_ptr + 32 + t, sh[32 + t]);
    atomicAdd(output_ptr + 64 + t, sh[64 + t]);
    atomicAdd(output_ptr + 96 + t, sh[96 + t]);
}
