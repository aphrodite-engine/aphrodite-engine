#include "ops_vk.h"
#include "vk_dispatch.h"

namespace aphrodite {
namespace vk {

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon) {
  const int hidden = input.size(-1);
  const int64_t num_tokens = input.numel() / hidden;
  const int64_t input_stride = input.stride(-2);
  RmsNormParams p{num_tokens, hidden, input_stride, static_cast<float>(epsilon), true};
  dispatch_rms_norm(p, out, input, weight);
}

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon) {
  // input = input + residual
  dispatch_add(input, input, residual);
  // then in-place normalize scaled by weight
  const int hidden = input.size(-1);
  const int64_t num_tokens = input.numel() / hidden;
  const int64_t input_stride = input.stride(-2);
  RmsNormParams p{num_tokens, hidden, input_stride, static_cast<float>(epsilon), true};
  dispatch_rms_norm(p, input, input, weight);
}

}  // namespace vk
}  // namespace aphrodite


