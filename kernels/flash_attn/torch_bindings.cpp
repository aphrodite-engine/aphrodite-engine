#include "../core/registration.h"
#include "flash_api.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, aphrodite_flash_attn_c) {
  aphrodite_flash_attn_c.def("fwd(Tensor! q, Tensor k, Tensor v, Tensor!? out, Tensor? alibi_slopes, "
          "float p_dropout, float softmax_scale, bool is_causal, int window_size_left, int window_size_right, "
          "float softcap, bool return_softmax, Generator? gen)"
          "-> Tensor[]");
  aphrodite_flash_attn_c.impl("fwd", torch::kCUDA, &mha_fwd);

  aphrodite_flash_attn_c.def("varlen_fwd(Tensor! q, Tensor k, Tensor v, Tensor!? out, Tensor cu_seqlens_q, "
          "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? block_table, Tensor? alibi_slopes, "
          "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
          "bool is_causal, int window_size_left, int window_size_right, float softcap, bool return_softmax, "
          "Generator? gen) -> Tensor[]");
  aphrodite_flash_attn_c.impl("varlen_fwd", torch::kCUDA, &mha_varlen_fwd);

  aphrodite_flash_attn_c.def("fwd_kvcache(Tensor! q, Tensor kcache, Tensor vcache, Tensor? k, Tensor? v, Tensor? seqlens_k, "
          "Tensor? rotary_cos, Tensor? rotary_sin, Tensor? cache_batch_idx, Tensor? block_table, Tensor? alibi_slopes, "
          "Tensor!? out, float softmax_scale, bool is_causal, int window_size_left, int window_size_right, "
          "float softcap, bool is_rotary_interleaved, int num_splits) -> Tensor[]");
  aphrodite_flash_attn_c.impl("fwd_kvcache", torch::kCUDA, &mha_fwd_kvcache);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)