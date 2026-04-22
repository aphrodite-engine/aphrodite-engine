#include <ATen/ATen.h>
#include <c10/util/Exception.h>

namespace torch {
using Tensor = at::Tensor;
}

void reshape_and_cache_nvfp4_dispatch(torch::Tensor& key,
                                      torch::Tensor& value,
                                      torch::Tensor& key_cache,
                                      torch::Tensor& value_cache,
                                      torch::Tensor& slot_mapping,
                                      torch::Tensor& k_scale,
                                      torch::Tensor& v_scale) {
  TORCH_CHECK(
      false,
      "NVFP4 KV cache kernels are not available in this Aphrodite build.");
}
