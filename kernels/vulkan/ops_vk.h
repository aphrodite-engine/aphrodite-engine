#pragma once

#include <optional>
#include <torch/torch.h>

namespace aphrodite {
namespace vk {

// LayerNorm / RMSNorm
void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);

// Rotary embeddings
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);

void batched_rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                              std::optional<torch::Tensor> key,
                              int64_t head_size,
                              torch::Tensor& cos_sin_cache, bool is_neox,
                              int64_t rot_dim,
                              torch::Tensor& cos_sin_cache_offsets);

// Activations (GLU-style and element-wise)
void silu_and_mul(torch::Tensor& out, torch::Tensor& input);
void mul_and_silu(torch::Tensor& out, torch::Tensor& input);
void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);
void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);
void fatrelu_and_mul(torch::Tensor& out, torch::Tensor& input,
                     double threshold);

void gelu_new(torch::Tensor& out, torch::Tensor& input);
void gelu_fast(torch::Tensor& out, torch::Tensor& input);
void gelu_quick(torch::Tensor& out, torch::Tensor& input);

}  // namespace vk
}  // namespace aphrodite


