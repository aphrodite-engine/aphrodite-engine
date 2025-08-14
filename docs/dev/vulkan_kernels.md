## Aphrodite Vulkan kernels reference (internal)


### Conventions
- Dtypes: use fp16 or fp32 variants as needed.
- Shapes use row-major indexing consistent with our CUDA kernels. All buffers are raw device buffers; strides are passed via push constants when required by shader.
- For fused operations not present as single shader, compose multiple dispatches in the same command buffer.

---

## LayerNorm / RMSNorm

CUDA entrypoints:
- `void rms_norm(Tensor& out, Tensor& input, Tensor& weight, double epsilon)`
- `void fused_add_rms_norm(Tensor& input, Tensor& residual, Tensor& weight, double epsilon)`

Vulkan shaders:
- RMSNorm: `rms_norm.comp` → SPIR-V name: `rms_norm_f32`/`rms_norm_f16`
- LayerNorm (mean-variance): `norm.comp` → SPIR-V name: `norm_f32`
- GroupNorm helper: `group_norm.comp` (not required here)

Bindings and parameters (rms_norm.comp):
- Includes `generic_binary_head.comp` and `types.comp`.
- local_size: 512 x 1 x 1
- Push constants via `generic_binary_head.comp` (`p`):
  - `ne00` (ncols), `nb01/nb02/nb03` (strides), `param1` (epsilon)
  - `get_aoffset()`, `get_boffset()`, `get_doffset()` are header helpers
- Buffers:
  - binding 0: `A_TYPE data_a[]`  (input)
  - binding 1: `B_TYPE data_b[]`  (weight) [only if `do_multiply` true]
  - binding 3: `D_TYPE data_d[]`  (output)
- Specialization constant `do_multiply` (const_id 1):
  - false: write `scale * x`
  - true:  write `scale * x * weight[col]` (handles wrap when `ncols > p.ne10`)

Dispatch shape:
- Workgroups: `[nrows, nchannels, samples]` from vulkan head headers
- For simple 2D: set `gl_NumWorkGroups = [num_tokens, 1, 1]`, `BLOCK_SIZE=512` threads

Wrapper mapping:
- `rms_norm`: single dispatch of `rms_norm.comp` with `do_multiply=true`, `data_b=weight`, `p.param1=epsilon`.
- `fused_add_rms_norm`: compose:
  1) `add.comp` into `residual` or a temp: `input = input + residual` (vectorized add)
  2) `rms_norm.comp` with `do_multiply=true` using `weight`, `epsilon` over the result

Notes:
- CUDA `rms_norm` writes to `out` shaped `[num_tokens, hidden]`. vulkan flattens and uses strides; map `input_stride` to `p.nb01` and contiguous output to linear `d_offset`.

---

## Rotary embeddings (RoPE)

CUDA entrypoints:
- `void rotary_embedding(Tensor& positions, Tensor& query, optional<Tensor> key, int64_t head_size, Tensor& cos_sin_cache, bool is_neox)`
- `void batched_rotary_embedding(Tensor& positions, Tensor& query, optional<Tensor> key, int64_t head_size, Tensor& cos_sin_cache, bool is_neox, int64_t rot_dim, Tensor& cos_sin_cache_offsets)`

Vulkan shaders:
- Shared header: `rope_head.comp` (bindings, push constants, YARN helpers)
- GPT-NeoX layout (split halves): `rope_neox.comp` → SPV: `rope_neox_f32`/`_f16`
- GPT-J/interleaved pairs: `rope_norm.comp` → SPV: `rope_norm_f32`/`_f16`
- Batched/multi sections: `rope_multi.comp` → SPV: `rope_multi_f32`/`_f16`
- Vision variant: `rope_vision.comp` (not required for LLM)

Bindings (`rope_head.comp`):
- binding 0: `A_TYPE data_a[]` (input Q/K flattened; we dispatch separately per tensor or pack)
- binding 1: `int data_pos[]` (positions per channel/token)
- binding 2: `float data_ff[]` (optional frequency factors; set `has_ff=0` if unused)
- binding 3: `D_TYPE data_d[]` (output)

Push constants struct `p` (rope_head.comp):
- `uint ncols`          // columns (full hidden or pair count depending shader)
- `uint n_dims`         // rotational dims applied (<= hidden)
- `float freq_scale`    // YARN scaling
- `uint p_delta_rows`   // rows per channel (seq length per batch slice)
- `float freq_base`     // unused in current rope_* kernels
- `float ext_factor`    // YARN interpolation factor
- `float attn_factor`   // magnitude scale factor
- `float corr_dims[2]`  // YARN correction dims
- `float theta_scale`   // base^(i/2) like scaler
- `uint has_ff`         // 1 if `data_ff` used
- `uint ne02`           // auxiliary dim for multi/vision
- `uint s1, s2`         // strides to locate row/channel in `data_a`
- `int sections[4]`     // per-section sizes for `rope_multi`/`rope_vision`
- `uint is_back`        // 1 for backward (negate theta)

Dispatch shape:
- `rope_*` kernels use `layout(local_size_x = 1, local_size_y = 256, local_size_z = 1)` and index by `gl_GlobalInvocationID.x` as row and `gl_GlobalInvocationID.y` stepping col pairs.
- For `[num_tokens, heads, head_size]`, set:
  - `p.p_delta_rows = seq_len` or rows per channel in the flattening scheme
  - `p.s1` = stride between rows (e.g., head_size or head_size/2 depending kernel)
  - `p.s2` = stride between channels (heads * stride per head)
  - `p.n_dims` = `rot_dim` (<= hidden), `p.ncols` = total columns per row
- Choose shader:
  - `is_neox=true` → `rope_neox.comp` (split halves)
  - else → `rope_norm.comp` (interleaved pairs)
  - batched LoRA offsets → `rope_multi.comp` and populate `sections[]`, `ne02`

Wrapper mapping:
- `rotary_embedding`: one dispatch over Q (and over K if passed, or pack Q/K into a larger buffer and dispatch once by setting shapes appropriately). Feed `data_pos` from `positions` view and set YARN params to no-op if not used: `freq_scale=1`, `ext_factor=0`, `attn_factor=1`, `corr_dims={0,0}`, `theta_scale=base`.
- `batched_rotary_embedding`: prefer `rope_multi.comp` to model per-section offsets; otherwise loop over tokens and dispatch `rope_norm/neox` with position offsets applied in `data_pos` and sections.

Note on cos/sin cache:
- CUDA uses a precomputed `cos_sin_cache`. vulkan shaders compute cos/sin from `positions` and YARN params on-the-fly. To match numerics with cache-based paths, set `freq_scale=1`, `ext_factor=0`, `attn_factor=1`, and precompute `theta_scale` consistent with cache generation. If we must consume a cache directly, a tiny adapter shader would be needed.

---

## Activations (elementwise and gated)

CUDA entrypoints:
- Gated GLU-style: `silu_and_mul`, `mul_and_silu`, `gelu_and_mul`, `gelu_tanh_and_mul`, `fatrelu_and_mul`
- Elementwise: `gelu_new`, `gelu_fast`, `gelu_quick`

Vulkan shaders (elementwise):
- `silu.comp` → SPV: `silu_f32`/`_f16` (x / (1 + exp(-x)))
- `gelu.comp` → SPV: `gelu_f32`/`_f16` (tanh-style GELU)
- `gelu_erf.comp` → SPV: `gelu_erf_f32`/`_f16` (erf-style GELU)
- `gelu_quick.comp` → SPV: `gelu_quick_f32`/`_f16` (x*sigmoid(1.702x))
- Also available: `relu.comp`, `leaky_relu.comp`, `tanh.comp`, `sigmoid.comp`

Vulkan shaders (GLU gating):
- Shared infra: `glu_head.comp`, `glu_main.comp`
- SWiGLU (SiLU gate): `swiglu.comp` → SPV: `swiglu_f32`/`_f16`
- GEGLU (tanh GELU gate): `geglu.comp` → SPV: `geglu_f32`/`_f16`
- GEGLU erf: `geglu_erf.comp` → SPV: `geglu_erf_f32`/`_f16`
- GEGLU quick: `geglu_quick.comp` → SPV: `geglu_quick_f32`/`_f16`

Bindings (`glu_head.comp`):
- binding 0: `A_TYPE data_a[]`  // primary input [N, 2*d] for default/swap modes
- binding 1: `A_TYPE data_b[]`  // optional second input for split mode
- binding 2: `D_TYPE data_d[]`  // output [N, d]

Push constants (`glu_head.comp`):
- `uint N`     // number of rows (tokens)
- `uint ne00`  // columns in `data_a` (2*d when default/swap)
- `uint ne20`  // output width `d`
- `uint mode`  // 0 default (act on first half), 1 swapped (act on second half), 2 split (a and b separate)
- `float alpha` // unused here
- `float limit` // unused here

Dispatch shape:
- local_size: 512 x 1 x 1. Global size: `[ceil_div(N*ne20, 512), 1, 1]` set via xyz decomposition used in `glu_main.comp` (`i` linearizes row,col).

Wrapper mapping:
- `silu_and_mul`: dispatch `swiglu.comp` with `mode=0`, `N=num_tokens`, `ne00=2*d`, `ne20=d`. Input packs [x, y]; output is `silu(x) * y`.
- `mul_and_silu`: same as above with `mode=1` (act on second half): `x * silu(y)`.
- `gelu_and_mul`: dispatch `geglu.comp` with `mode=0` (tanh GELU gate).
- `gelu_tanh_and_mul`: same as above (matches CUDA tanh approximation).
- `gelu_new` (elementwise tanh-based), `gelu_fast` (elementwise tanh fast), `gelu_quick`: use `gelu.comp`, `gelu.comp`/`gelu_erf.comp` as needed, `gelu_quick.comp` respectively with `p.KX = d`, `N = num_tokens` in `generic_head.comp`-style launchers.
- `fatrelu_and_mul`: not present as GLU shader; implement via a custom variant modeled on `glu_main.comp` with op `fatrelu(a, threshold) * b`, or approximate with `relu.comp` plus custom gate. If needed later, clone `swiglu.comp` pattern.
