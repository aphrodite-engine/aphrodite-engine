#pragma once

// Per-K selectors for the int8 GEMV kernel instances, each defined in its own
// compilation unit

void* exl3_gemv_int8_sq_sel_k1(int M, bool c_fp32, bool residual);
void* exl3_gemv_int8_sq_sel_k2(int M, bool c_fp32, bool residual);
void* exl3_gemv_int8_sq_sel_k3(int M, bool c_fp32, bool residual);
void* exl3_gemv_int8_sq_sel_k4(int M, bool c_fp32, bool residual);
void* exl3_gemv_int8_sq_sel_k5(int M, bool c_fp32, bool residual);
void* exl3_gemv_int8_sq_sel_k6(int M, bool c_fp32, bool residual);

void* exl3_gemv_int8_coop_sel_k1(bool c_fp32, bool residual);
void* exl3_gemv_int8_coop_sel_k2(bool c_fp32, bool residual);
void* exl3_gemv_int8_coop_sel_k3(bool c_fp32, bool residual);
void* exl3_gemv_int8_coop_sel_k4(bool c_fp32, bool residual);
void* exl3_gemv_int8_coop_sel_k5(bool c_fp32, bool residual);
void* exl3_gemv_int8_coop_sel_k6(bool c_fp32, bool residual);
void* exl3_gemv_int8_coop_sel_k7(bool c_fp32, bool residual);
void* exl3_gemv_int8_coop_sel_k8(bool c_fp32, bool residual);
