<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the vLLM project -->

# Vendored CuTe DSL source

`cutedsl_kernel.py` is derived from NVIDIA CUTLASS commit
`f94ec46f4f63f96003d6cfdf2014731e7672c281`:

`examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py`

Sonar integrates the kernel with its packed-weight ABI and Torch custom
operator in `cutedsl.py`. The upstream command-line reference helpers remain
in the vendored file to make future CUTLASS updates easier.

`cutedsl_grouped_kernel.py` derives from the grouped block-scaled GEMM example
at the same CUTLASS commit. It ports mixed-width operand and FP6 TMA-unpack
support from the dense implementation. NVIDIA's grouped example only accepted
same-width operands at that revision.
