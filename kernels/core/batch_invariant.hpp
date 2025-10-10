#pragma once
#include <cstdlib>
#include <string>
#include <cctype>

namespace aphrodite {

// aphrodite_kernel_override_batch_invariant(); returns true
// if env APHRODITE_KERNEL_OVERRIDE_BATCH_INVARIANT=1
inline bool aphrodite_kernel_override_batch_invariant() {
  std::string env_key = "APHRODITE_KERNEL_OVERRIDE_BATCH_INVARIANT";
  const char* val = std::getenv(env_key.c_str());
  return (val && std::atoi(val) != 0) ? 1 : 0;
}

}  // namespace aphrodite
