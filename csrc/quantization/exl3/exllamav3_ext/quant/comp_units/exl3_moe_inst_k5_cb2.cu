#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k5_n128_cb2() {
  return exl3_moe_kernel<5, 128, 2>;
}
fp_exl3_moe_kernel exl3_moe_kernel_k5_n256_cb2() {
  return exl3_moe_kernel<5, 256, 2>;
}
