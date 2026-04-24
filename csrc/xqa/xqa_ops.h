#pragma once

#include <torch/all.h>

void xqa_paged_attention(torch::Tensor& out, torch::Tensor& query,
                         torch::Tensor& key_value_cache, int64_t num_heads,
                         int64_t num_kv_heads, int64_t rotary_embedding_dim,
                         double scale, torch::Tensor& block_tables,
                         torch::Tensor& seq_lens, int64_t block_size,
                         int64_t max_seq_len, const std::string kv_cache_dtype,
                         double k_scale, double v_scale);