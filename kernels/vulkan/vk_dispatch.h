#pragma once

#include <optional>
#include <torch/torch.h>
// Forward-declare minimal Vulkan runtime types
struct VkInstance_T;
struct VkDevice_T;
struct VkQueue_T;
using VkInstance = VkInstance_T*;
using VkDevice = VkDevice_T*;
using VkQueue = VkQueue_T*;

namespace aphrodite {
namespace vk {

struct GluParams {
  // Number of tokens and hidden dims
  int64_t numTokens;
  int64_t hiddenDim; // d
  // Mode: 0 default (act first half), 1 swapped (act second half), 2 split (separate a/b)
  int mode;
};

enum class GluOpKind {
  SiLU,
  GELU_Tanh,
  GELU_Erf,
  GELU_Quick,
  FatReLU
};

struct UnaryParams {
  int64_t numTokens;
  int64_t width; // d
};

struct RmsNormParams {
  int64_t numTokens;
  int64_t hiddenSize;
  int64_t inputStride; // stride between rows in input
  float epsilon;
  bool multiplyWeight; // true for scaling by weight
};

struct RopeParams {
  bool isNeoX;
  int64_t numTokens;
  int64_t headSize;
  int64_t numHeads;
  int64_t numKvHeads; // may equal numHeads
  int64_t rotDim;     // rot_dim
  int64_t queryStride;
  int64_t keyStride;  // 0 if no key
  int64_t headStride; // stride between heads in last-2 dim layout
};

// GLU-style dispatch: out: [..., d], input: [..., 2*d] (or split via mode=2)
void dispatch_glu(GluOpKind kind, const GluParams& params,
                  torch::Tensor& out, torch::Tensor& input,
                  std::optional<double> param = std::nullopt);

// Unary elementwise dispatch: out/input [..., d]
void dispatch_unary_gelu_tanh(const UnaryParams& p,
                              torch::Tensor& out, torch::Tensor& input);
void dispatch_unary_gelu_erf(const UnaryParams& p,
                             torch::Tensor& out, torch::Tensor& input);
void dispatch_unary_gelu_quick(const UnaryParams& p,
                               torch::Tensor& out, torch::Tensor& input);

// RMSNorm dispatch
void dispatch_rms_norm(const RmsNormParams& p, torch::Tensor& out,
                       torch::Tensor& input, torch::Tensor& weight);

// Vector add (used for fused_add_rms_norm)
void dispatch_add(torch::Tensor& out, const torch::Tensor& a,
                  const torch::Tensor& b);

// RoPE dispatch over query (and optional key)
void dispatch_rope(const RopeParams& p, torch::Tensor& positions,
                   torch::Tensor& query, std::optional<torch::Tensor> key,
                   torch::Tensor& cosSinCache /*ignored for now*/);

}  // namespace vk
}  // namespace aphrodite


