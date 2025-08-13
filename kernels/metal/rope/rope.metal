#include <metal_stdlib>
using namespace metal;

struct RotaryParams {
  uint head_size;
  uint rot_dim;      // 2 * embed_dim
  uint embed_dim;    // rot_dim / 2
  uint total_heads;  // number of heads per token (q or k)
  uint num_tokens;   // number of tokens
};

static inline void decode_gid(uint gid, uint embed_dim, uint total_heads,
                              thread uint &token_idx, thread uint &head_idx,
                              thread uint &j_embed) {
  j_embed = gid % embed_dim;
  uint u = gid / embed_dim;
  head_idx = u % total_heads;
  token_idx = u / total_heads;
}

// NEoX-style rotary: first half are x, second half are y
kernel void rotary_neox_float(
  device float *q                  [[ buffer(0) ]],
  device const long *positions     [[ buffer(1) ]],
  device const float *cos_sin      [[ buffer(2) ]], // [max_pos, rot_dim]
  constant RotaryParams &params    [[ buffer(3) ]],
  uint gid                         [[ thread_position_in_grid ]]) {
  const uint embed_dim = params.embed_dim;
  const uint total_heads = params.total_heads;
  const uint num_tokens = params.num_tokens;
  if (gid >= num_tokens * total_heads * embed_dim) return;

  uint token_idx, head_idx, j;
  decode_gid(gid, embed_dim, total_heads, token_idx, head_idx, j);

  const long pos = positions[token_idx];
  const uint base = (token_idx * total_heads + head_idx) * params.head_size;
  const uint x_index = base + j;
  const uint y_index = base + embed_dim + j;

  const uint cache_base = uint(pos) * params.rot_dim;
  const float c = cos_sin[cache_base + j];
  const float s = cos_sin[cache_base + embed_dim + j];

  const float x = q[x_index];
  const float y = q[y_index];

  q[x_index] = x * c - y * s;
  q[y_index] = y * c + x * s;
}

kernel void rotary_neox_half(
  device half *q                   [[ buffer(0) ]],
  device const long *positions     [[ buffer(1) ]],
  device const float *cos_sin      [[ buffer(2) ]],
  constant RotaryParams &params    [[ buffer(3) ]],
  uint gid                         [[ thread_position_in_grid ]]) {
  const uint embed_dim = params.embed_dim;
  const uint total_heads = params.total_heads;
  const uint num_tokens = params.num_tokens;
  if (gid >= num_tokens * total_heads * embed_dim) return;

  uint token_idx, head_idx, j;
  decode_gid(gid, embed_dim, total_heads, token_idx, head_idx, j);

  const long pos = positions[token_idx];
  const uint base = (token_idx * total_heads + head_idx) * params.head_size;
  const uint x_index = base + j;
  const uint y_index = base + embed_dim + j;

  const uint cache_base = uint(pos) * params.rot_dim;
  const float c = cos_sin[cache_base + j];
  const float s = cos_sin[cache_base + embed_dim + j];

  const float x = (float)q[x_index];
  const float y = (float)q[y_index];

  q[x_index] = (half)(x * c - y * s);
  q[y_index] = (half)(y * c + x * s);
}

// GPT-J style rotary: interleaved pairs (0,1), (2,3), ... within rot_dim
kernel void rotary_gptj_float(
  device float *q                  [[ buffer(0) ]],
  device const long *positions     [[ buffer(1) ]],
  device const float *cos_sin      [[ buffer(2) ]],
  constant RotaryParams &params    [[ buffer(3) ]],
  uint gid                         [[ thread_position_in_grid ]]) {
  const uint embed_dim = params.embed_dim;
  const uint total_heads = params.total_heads;
  const uint num_tokens = params.num_tokens;
  if (gid >= num_tokens * total_heads * embed_dim) return;

  uint token_idx, head_idx, j;
  decode_gid(gid, embed_dim, total_heads, token_idx, head_idx, j);

  const long pos = positions[token_idx];
  const uint base = (token_idx * total_heads + head_idx) * params.head_size;
  const uint x_index = base + (2u * j + 0u);
  const uint y_index = base + (2u * j + 1u);

  const uint cache_base = uint(pos) * params.rot_dim;
  const float c = cos_sin[cache_base + j];
  const float s = cos_sin[cache_base + params.embed_dim + j];

  const float x = q[x_index];
  const float y = q[y_index];

  q[x_index] = x * c - y * s;
  q[y_index] = y * c + x * s;
}

kernel void rotary_gptj_half(
  device half *q                   [[ buffer(0) ]],
  device const long *positions     [[ buffer(1) ]],
  device const float *cos_sin      [[ buffer(2) ]],
  constant RotaryParams &params    [[ buffer(3) ]],
  uint gid                         [[ thread_position_in_grid ]]) {
  const uint embed_dim = params.embed_dim;
  const uint total_heads = params.total_heads;
  const uint num_tokens = params.num_tokens;
  if (gid >= num_tokens * total_heads * embed_dim) return;

  uint token_idx, head_idx, j;
  decode_gid(gid, embed_dim, total_heads, token_idx, head_idx, j);

  const long pos = positions[token_idx];
  const uint base = (token_idx * total_heads + head_idx) * params.head_size;
  const uint x_index = base + (2u * j + 0u);
  const uint y_index = base + (2u * j + 1u);

  const uint cache_base = uint(pos) * params.rot_dim;
  const float c = cos_sin[cache_base + j];
  const float s = cos_sin[cache_base + params.embed_dim + j];

  const float x = (float)q[x_index];
  const float y = (float)q[y_index];

  q[x_index] = (half)(x * c - y * s);
  q[y_index] = (half)(y * c + x * s);
}
