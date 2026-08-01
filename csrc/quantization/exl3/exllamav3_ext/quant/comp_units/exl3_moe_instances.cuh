#pragma once

#include "../exl3_moe_common.cuh"

typedef void (*fp_exl3_moe_kernel)(EXL3_MOE_KERNEL_ARGS);

#define EXL3_MOE_DECLARE_GETTERS(K)                     \
  fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb1(); \
  fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb1(); \
  fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2(); \
  fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb2();

EXL3_MOE_DECLARE_GETTERS(0);
EXL3_MOE_DECLARE_GETTERS(1);
EXL3_MOE_DECLARE_GETTERS(2);
EXL3_MOE_DECLARE_GETTERS(3);
EXL3_MOE_DECLARE_GETTERS(4);
EXL3_MOE_DECLARE_GETTERS(5);
EXL3_MOE_DECLARE_GETTERS(6);
EXL3_MOE_DECLARE_GETTERS(7);
EXL3_MOE_DECLARE_GETTERS(8);

#undef EXL3_MOE_DECLARE_GETTERS

extern fp_exl3_moe_kernel exl3_moe_kernel_instances[];
