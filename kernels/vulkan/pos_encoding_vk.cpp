#include "ops_vk.h"
#include "vk_dispatch.h"

namespace aphrodite {
namespace vk {

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox) {
  const int64_t num_tokens = positions.numel();
  const int q_hidden_size = query.numel() / num_tokens;
  const int k_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  const int64_t num_heads = q_hidden_size / head_size;
  const int64_t num_kv_heads = key.has_value() ? k_hidden_size / head_size : num_heads;
  const int positions_ndim = positions.dim();
  const int seq_dim_idx = positions_ndim - 1;
  const int64_t query_stride = query.stride(seq_dim_idx);
  const int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;
  const int query_ndim = query.dim();
  const int64_t head_stride = (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;
  RopeParams p{is_neox, num_tokens, head_size, num_heads, num_kv_heads,
               /*rotDim*/ head_size * 2, query_stride, key_stride, head_stride};
  dispatch_rope(p, positions, query, key, cos_sin_cache);
}

void batched_rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                              std::optional<torch::Tensor> key,
                              int64_t head_size,
                              torch::Tensor& cos_sin_cache, bool is_neox,
                              int64_t rot_dim,
                              torch::Tensor& cos_sin_cache_offsets) {
  (void)cos_sin_cache_offsets; // not used yet; rope_multi variant can be added
  const int64_t num_tokens = positions.numel();
  const int q_hidden_size = query.numel() / num_tokens;
  const int k_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  const int64_t num_heads = q_hidden_size / head_size;
  const int64_t num_kv_heads = key.has_value() ? k_hidden_size / head_size : num_heads;
  const int positions_ndim = positions.dim();
  const int seq_dim_idx = positions_ndim - 1;
  const int64_t query_stride = query.stride(seq_dim_idx);
  const int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;
  const int query_ndim = query.dim();
  const int64_t head_stride = (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;
  RopeParams p{is_neox, num_tokens, head_size, num_heads, num_kv_heads,
               rot_dim, query_stride, key_stride, head_stride};
  dispatch_rope(p, positions, query, key, cos_sin_cache);
}

}  // namespace vk
}  // namespace aphrodite


