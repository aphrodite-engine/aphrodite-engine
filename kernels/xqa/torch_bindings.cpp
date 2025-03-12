#include "../core/registration.h"
#include "xqa_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, xqa_ops) {
  // PagedAttention xqa.
  xqa_ops.def(
      "xqa_paged_attention("
      "    Tensor! out,"
      "    Tensor query, Tensor key_value_cache,"
      "    int num_heads, int num_kv_heads, int rotary_embedding_dim,"
      "    float scale, Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, str kv_cache_dtype,"
      "    float k_scale, float v_scale) -> ()");
  xqa_ops.impl("xqa_paged_attention", torch::kCUDA, &xqa_paged_attention);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)