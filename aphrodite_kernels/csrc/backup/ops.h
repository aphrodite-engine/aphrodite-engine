#pragma once

#include <torch/extension.h>

void paged_attention_v1(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, float k_scale = 1.0f, float k_zp = 0.0f,
    float v_scale = 1.0f, float v_zp = 0.0f);

void paged_attention_v2(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, float k_scale = 1.0f, float k_zp = 0.0f,
    float v_scale = 1.0f, float v_zp = 0.0f);

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              float epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, float epsilon);

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      torch::Tensor& key, int head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

#ifndef USE_ROCM
torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int split_k_iters);

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int split_k_iters, int thx,
                             int thy);

torch::Tensor autoquant_s4_f16_gemm(torch::Tensor _in_feats,
                                    torch::Tensor _kernel,
                                    torch::Tensor _scales_zeros);

void autoquant_convert_s4_k_m8(torch::Tensor _weight_dest,
                               torch::Tensor _quant_scales_zeros_dest,
                               torch::Tensor _workspace,
                               torch::Tensor _quant_weight_src,
                               torch::Tensor _quant_scales,
                               torch::Tensor _quant_zeros, int m, int k,
                               int group_size);

torch::Tensor marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                          torch::Tensor& b_scales, torch::Tensor& workspace,
                          int64_t size_m, int64_t size_n, int64_t size_k);

at::Tensor e8p_mm_origorder(const at::Tensor& A, const at::Tensor& B,
                            const at::Tensor& CB);

void decompress_e8p_origorder(torch::Tensor YIs, torch::Tensor CB,
                              torch::Tensor& Y);
#endif

void squeezellm_gemm(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
                     torch::Tensor lookup_table);

torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int bit);

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int bit);

torch::Tensor aqlm_gemm(const torch::Tensor& input, const torch::Tensor& codes,
                        const torch::Tensor& codebooks,
                        const torch::Tensor& scales,
                        const torch::Tensor& codebook_partition_sizes,
                        const std::optional<torch::Tensor>& bias);

torch::Tensor ggml_dequantize(torch::Tensor X, int8_t type, int64_t m,
                              int64_t n);

torch::Tensor ggml_mul_mat_vec(torch::Tensor W,  // quant weight
                               torch::Tensor X,  // input
                               int8_t type, int64_t m);

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W,  // quant weight
                                  torch::Tensor X,  // input
                                  int8_t type, int64_t row);

torch::Tensor ggml_mul_mat_a8(torch::Tensor W,  // quant weight
                              torch::Tensor X,  // input
                              int8_t type, int64_t row);

uintptr_t make_q_matrix(torch::Tensor q_weight, torch::Tensor q_perm,
                        torch::Tensor q_invperm, torch::Tensor q_scale,
                        torch::Tensor q_scale_max, torch::Tensor q_groups,
                        torch::Tensor q_group_map);

torch::Tensor exl2_gemm(torch::Tensor a, uintptr_t b);

void moe_align_block_size(torch::Tensor topk_ids, int num_experts,
                          int block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_pad);

#ifndef USE_ROCM
using fptr_t = uint64_t;
fptr_t init_custom_ar(torch::Tensor& meta, torch::Tensor& rank_data,
                      const std::vector<std::string>& handles,
                      const std::vector<int64_t>& offsets, int rank,
                      bool full_nvlink);
bool should_custom_ar(torch::Tensor& inp, int max_size, int world_size,
                      bool full_nvlink);
void all_reduce_reg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out);
void all_reduce_unreg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& reg_buffer,
                      torch::Tensor& out);
void dispose(fptr_t _fa);
int meta_size();
void register_buffer(fptr_t _fa, torch::Tensor& t,
                     const std::vector<std::string>& handles,
                     const std::vector<int64_t>& offsets);
std::pair<std::vector<uint8_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(
    fptr_t _fa);
void register_graph_buffers(fptr_t _fa, const std::vector<std::string>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);
#endif
