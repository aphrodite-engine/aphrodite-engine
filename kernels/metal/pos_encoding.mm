// Forwarder now implemented in kernels/metal/rope/rope.mm
#include <ATen/ATen.h>
#include <torch/torch.h>

void rotary_embedding(torch::Tensor& positions,
                      torch::Tensor& query,
                      c10::optional<torch::Tensor> key,
                      int64_t head_size,
                      torch::Tensor& cos_sin_cache,
                      bool is_neox);


