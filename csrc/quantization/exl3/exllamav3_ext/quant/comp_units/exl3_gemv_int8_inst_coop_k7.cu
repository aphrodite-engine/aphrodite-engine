#include "exl3_gemv_int8_instances.cuh"
#include "../exl3_gemv_int8_kernel.cuh"

void* exl3_gemv_int8_coop_sel_k7(bool c_fp32, bool residual) {
  if (c_fp32)
    return residual ? (void*)exl3_gemv_int8_coop_kernel<7, true, true>
                    : (void*)exl3_gemv_int8_coop_kernel<7, true, false>;
  else
    return residual ? (void*)exl3_gemv_int8_coop_kernel<7, false, true>
                    : (void*)exl3_gemv_int8_coop_kernel<7, false, false>;
}
