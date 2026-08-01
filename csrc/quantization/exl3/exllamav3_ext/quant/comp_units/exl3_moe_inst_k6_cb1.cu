#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k6_n128_cb1() {
  return exl3_moe_kernel<6, 128, 1>;
}
fp_exl3_moe_kernel exl3_moe_kernel_k6_n256_cb1() {
  return exl3_moe_kernel<6, 256, 1>;
}
