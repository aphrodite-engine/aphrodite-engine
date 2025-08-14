#include "ops_vk.h"
#include "vk_dispatch.h"

namespace aphrodite {
namespace vk {

// NOTE: Stub implementations to document intended behavior and signatures.
// Vulkan dispatch wiring will be added in follow-up edits.

void silu_and_mul(torch::Tensor& out, torch::Tensor& input) {
  const int64_t d2 = input.size(-1);
  const int64_t d = d2 / 2;
  const int64_t num_tokens = input.numel() / d2;
  GluParams params{num_tokens, d, /*mode*/0};
  dispatch_glu(GluOpKind::SiLU, params, out, input);
}

void mul_and_silu(torch::Tensor& out, torch::Tensor& input) {
  const int64_t d2 = input.size(-1);
  const int64_t d = d2 / 2;
  const int64_t num_tokens = input.numel() / d2;
  GluParams params{num_tokens, d, /*mode*/1};
  dispatch_glu(GluOpKind::SiLU, params, out, input);
}

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input) {
  const int64_t d2 = input.size(-1);
  const int64_t d = d2 / 2;
  const int64_t num_tokens = input.numel() / d2;
  GluParams params{num_tokens, d, /*mode*/0};
  dispatch_glu(GluOpKind::GELU_Tanh, params, out, input);
}

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input) {
  const int64_t d2 = input.size(-1);
  const int64_t d = d2 / 2;
  const int64_t num_tokens = input.numel() / d2;
  GluParams params{num_tokens, d, /*mode*/0};
  dispatch_glu(GluOpKind::GELU_Tanh, params, out, input);
}

void fatrelu_and_mul(torch::Tensor& out, torch::Tensor& input,
                     double threshold) {
  const int64_t d2 = input.size(-1);
  const int64_t d = d2 / 2;
  const int64_t num_tokens = input.numel() / d2;
  GluParams params{num_tokens, d, /*mode*/0};
  dispatch_glu(GluOpKind::FatReLU, params, out, input, threshold);
}

void gelu_new(torch::Tensor& out, torch::Tensor& input) {
  const int64_t d = input.size(-1);
  const int64_t num_tokens = input.numel() / d;
  UnaryParams p{num_tokens, d};
  dispatch_unary_gelu_tanh(p, out, input);
}

void gelu_fast(torch::Tensor& out, torch::Tensor& input) {
  const int64_t d = input.size(-1);
  const int64_t num_tokens = input.numel() / d;
  UnaryParams p{num_tokens, d};
  dispatch_unary_gelu_tanh(p, out, input);
}

void gelu_quick(torch::Tensor& out, torch::Tensor& input) {
  const int64_t d = input.size(-1);
  const int64_t num_tokens = input.numel() / d;
  UnaryParams p{num_tokens, d};
  dispatch_unary_gelu_quick(p, out, input);
}

}  // namespace vk
}  // namespace aphrodite


