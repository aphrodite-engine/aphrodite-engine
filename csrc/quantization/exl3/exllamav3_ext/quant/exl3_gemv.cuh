#pragma once

#include "../util.h"
#include <cuda_runtime.h>

// QTIP-style small-m GEMV path (see exl3_gemv_kernel.cuh). Launched from
// exl3_gemm when the heuristic applies; also exposed directly for testing. Same
// kernel arguments as exl3_gemm_kernel, so graph recording/patching is
// identical.

// Try to dispatch a GEMM call to the GEMV kernel. Returns false (launching
// nothing) if the call is not eligible. On success *launched_kernel receives
// the kernel pointer for graph recording. `force` bypasses the shape heuristic
// but not the hard constraints.
bool exl3_gemv_try_launch(void** kernel_args, int size_m, int size_k,
                          int size_n, int K, int cb, bool c_fp32,
                          bool has_su_sv, int device, cudaStream_t stream,
                          void** launched_kernel, bool force);

// Direct entry point (testing): errors if the call is not hard-eligible
void exl3_gemv(const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
               const c10::optional<at::Tensor>& suh,
               const c10::optional<at::Tensor>& A_had,
               const c10::optional<at::Tensor>& svh, bool mcg, bool mul1);
