#pragma once

#include <torch/library.h>

#include "core/scalar_type.hpp"

#ifndef USE_ROCM
// AQLM
torch::Tensor aqlm_gemm(const torch::Tensor& input, const torch::Tensor& codes,
                        const torch::Tensor& codebooks,
                        const torch::Tensor& scales,
                        const std::vector<int64_t>& codebook_partition_sizes,
                        const std::optional<torch::Tensor>& bias);

torch::Tensor aqlm_dequant(
    const torch::Tensor& codes, const torch::Tensor& codebooks,
    const std::vector<int64_t>& codebook_partition_sizes);

// AWQ
torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int64_t split_k_iters);

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy);

torch::Tensor awq_group_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, torch::Tensor _topk_weights,
                             torch::Tensor _sorted_token_ids_ptr,
                             torch::Tensor _expert_ids_ptr,
                             torch::Tensor _num_tokens_post_padded,
                             bool mul_weights, int64_t split_k_iters);
#endif

// GPTQ
torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit);

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit);

torch::Tensor group_gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_gptq_qzeros,
                              torch::Tensor b_gptq_scales,
                              torch::Tensor b_g_idx, torch::Tensor topk_weights,
                              torch::Tensor sorted_token_ids_ptr,
                              torch::Tensor expert_ids_ptr,
                              torch::Tensor num_tokens_post_padded,
                              bool mul_weights, bool use_exllama);

torch::Tensor dequant_gptq(torch::Tensor b_q_weight,
                           torch::Tensor b_gptq_qzeros,
                           torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                           int64_t bits, bool use_exllama);

#ifndef USE_ROCM
// Marlin
torch::Tensor marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                          torch::Tensor& b_scales, torch::Tensor& workspace,
                          int64_t size_m, int64_t size_n, int64_t size_k);

torch::Tensor gptq_marlin_24_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                                  torch::Tensor& b_meta,
                                  torch::Tensor& b_scales,
                                  torch::Tensor& workspace,
                                  aphrodite::ScalarTypeTorchPtr const& b_q_type,
                                  int64_t size_m, int64_t size_n,
                                  int64_t size_k);

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& b_zeros,
                               torch::Tensor& g_idx, torch::Tensor& perm,
                               torch::Tensor& workspace,
                               aphrodite::ScalarTypeTorchPtr const& b_q_type,
                               int64_t size_m, int64_t size_n, int64_t size_k,
                               bool is_k_full, bool has_zp,
                               bool use_fp32_reduce);

torch::Tensor gptq_marlin_repack(torch::Tensor& b_q_weight, torch::Tensor& perm,
                                 int64_t size_k, int64_t size_n,
                                 int64_t num_bits);

torch::Tensor gptq_marlin_repack_meta(torch::Tensor& b_q_weight,
                                      torch::Tensor& perm, c10::SymInt size_k,
                                      c10::SymInt size_n, int64_t num_bits);

torch::Tensor awq_marlin_repack(torch::Tensor& b_q_weight, int64_t size_k,
                                int64_t size_n, int64_t num_bits);

torch::Tensor awq_marlin_repack_meta(torch::Tensor& b_q_weight,
                                     c10::SymInt size_k, c10::SymInt size_n,
                                     int64_t num_bits);

torch::Tensor fp8_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                              torch::Tensor& b_scales, torch::Tensor& workspace,
                              int64_t num_bits, int64_t size_m, int64_t size_n,
                              int64_t size_k);

// GGUF
torch::Tensor ggml_dequantize(torch::Tensor W, int64_t type, int64_t m,
                              int64_t n);

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X,
                                  int64_t type, int64_t row);

torch::Tensor ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type,
                              int64_t row);

// QuIP#
at::Tensor e8p_mm_origorder(const at::Tensor& A, const at::Tensor& B,
                            const at::Tensor& CB);

void decompress_e8p_origorder(torch::Tensor YIs, torch::Tensor CB,
                              torch::Tensor& Y);

  #ifndef _WIN32
// Cutlass Kernels
bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability);

void cutlass_scaled_mm(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       c10::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_azp(torch::Tensor& out, torch::Tensor const& a,
                           torch::Tensor const& b,
                           torch::Tensor const& a_scales,
                           torch::Tensor const& b_scales,
                           torch::Tensor const& azp_adj,
                           c10::optional<torch::Tensor> const& azp,
                           c10::optional<torch::Tensor> const& bias);

// Machete Kernels
namespace machete {

std::vector<std::string> supported_schedules(
    aphrodite::ScalarTypeTorchPtr const& btype);

torch::Tensor gemm(torch::Tensor const& A, torch::Tensor const& B,
                   aphrodite::ScalarTypeTorchPtr const& btype,
                   c10::optional<torch::Tensor> const& scales,
                   c10::optional<torch::Tensor> const& zeros,
                   c10::optional<int64_t> group_size,
                   c10::optional<torch::Tensor> const& C,
                   c10::optional<double> alpha, c10::optional<double> beta,
                   c10::optional<std::string> schedule);

torch::Tensor prepack_B(torch::Tensor const& B,
                        aphrodite::ScalarTypeTorchPtr const& btype);

};  // namespace machete

torch::Tensor marlin_qqq_gemm(torch::Tensor const& a,
                              torch::Tensor const& b_q_weight,
                              torch::Tensor const& s_tok,
                              torch::Tensor const& s_ch,
                              torch::Tensor const& s_group,
                              torch::Tensor& workspace, int64_t size_m,
                              int64_t size_n, int64_t size_k);
  #endif

torch::Tensor fp_eXmY_linear_forward_cuda(int64_t EXPONENT, int64_t MANTISSA,
                                          torch::Tensor _in_feats,
                                          torch::Tensor _weights,
                                          torch::Tensor _scales,
                                          int64_t splitK = 1);
#endif

void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& scale,
                              c10::optional<torch::Tensor> const& azp);

void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                               torch::Tensor& scales,
                               c10::optional<torch::Tensor> const& azp);

// SqueezeLLM
void squeezellm_gemm(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
                     torch::Tensor lookup_table);

// FP8
void static_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                             torch::Tensor const& scale);

void dynamic_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale,
    c10::optional<torch::Tensor> const& scale_ub);
