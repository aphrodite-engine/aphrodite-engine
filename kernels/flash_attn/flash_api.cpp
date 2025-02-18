#include "flash_api.h"
#include "core/registration.h"

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("fwd", &mha_fwd);
  m.impl("varlen_fwd", &mha_varlen_fwd);
  m.impl("fwd_kvcache", &mha_fwd_kvcache);
}