#pragma once

#include "../util.h"
#include <cuda_runtime.h>

#include "../graph.cuh"

// Fused int8-activation GEMV for mul1 (cb 2) tensors: single cooperative launch
// covering input Hadamard, activation quantization, dp4a GEMV and output
// Hadamard. EXL3_INT8_GEMV=2 (default) is the plain int8 mode, =1 the
// error-feedback residual mode (~15-16 bit effective activation precision), =0
// disables the path. Tensors outside the mul1 codebook are unaffected.
// Arbitrary m is handled as sequential row passes within the launch, so
// intended for the small-m regime. See benchmarks/exl3_m1_bench (variants
// 15-18) for derivation, microbenchmarks and profiling.

bool exl3_gemv_int8_enabled();

// Highest K the int8 path accepts on `device` (per-arch, EXL3_INT8_GEMV_MAX_K
// override). Also bound to Python so the mgemm fusion heuristic can mirror the
// gate
int exl3_gemv_int8_max_k(int device);

// Returns true if the operation was handled (false -> caller should fall
// through to the regular kernel)
bool exl3_gemv_int8(const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
                    const c10::optional<at::Tensor>& suh,
                    const c10::optional<at::Tensor>& A_had,
                    const c10::optional<at::Tensor>& svh, cudaStream_t stream,
                    Graph* graph);
