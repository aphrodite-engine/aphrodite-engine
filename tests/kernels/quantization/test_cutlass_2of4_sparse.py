"""Tests for sparse cutlass kernels

Run `pytest tests/kernels/test_semi_structured.py`.
"""

import pytest
import torch

from tests.kernels.utils import baseline_scaled_mm, to_fp8, to_int8
from aphrodite import _custom_ops as ops
from aphrodite.quantization.utils.w8a8_utils import (
    sparse_cutlass_supported)
from aphrodite.platforms import current_platform

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

capability = current_platform.get_device_capability()
capability = capability[0] * 10 + capability[1]


def to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.bfloat16)


def to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float16)


def prune_to_2_4(tensor):
    # Reshape tensor to [N, 4] where N is number of groups of 4
    original_shape = tensor.shape
    reshaped = tensor.reshape(-1, 4)

    # Get indices of top 2 absolute values in each group of 4
    _, indices = torch.topk(torch.abs(reshaped), k=2, dim=1)

    # Create binary mask
    mask = torch.zeros_like(reshaped)
    mask.scatter_(dim=1,
                  index=indices,
                  src=torch.ones_like(indices, dtype=mask.dtype))

    # Apply mask and reshape back
    pruned = reshaped * mask

    # Turn all -0.0 to 0.0
    pruned[pruned == -0.0] = 0.0

    return pruned.reshape(original_shape)


# This function checks that applying an identity matrix multiplication
# to the compressed weights yields the original uncompressed weights.
def check_compress_decompress_invariance(dtype: torch.dtype, b: torch.Tensor,
                                         b_compressed: torch.Tensor,
                                         b_metadata: torch.Tensor):

    # For float16 and bfloat16, cutlass_scaled_sparse_mm's output must be the
    # same dtype as its inputs. This line addresses that constraint while
    # arbitrarily using bfloat16 for the int8/fp8 cases.
    out_dtype = torch.float16 if dtype is torch.float16 else torch.bfloat16

    eye = torch.eye(b.shape[0], device='cuda', dtype=dtype)
    eye_scale = torch.ones(1, device='cuda', dtype=torch.float32)
    b_decomp = ops.cutlass_scaled_sparse_mm(eye,
                                            b_compressed,
                                            b_metadata,
                                            eye_scale,
                                            eye_scale,
                                            out_dtype=out_dtype)

    torch.testing.assert_close(b.to(dtype=out_dtype), b_decomp)


def make_rand_sparse_tensors(
        dtype: torch.dtype, m: int, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device='cuda')
    b = torch.randn((n, k), device='cuda').t()

    if dtype == torch.int8:
        # ensure A and B aren't all zeros after rounding
        a = a * 5.0
        b = b * 5.0

    b = prune_to_2_4(b.t()).t()

    if dtype == torch.int8:
        a, b = to_int8(a), to_int8(b)
    elif dtype == torch.float8_e4m3fn:
        a, b = to_fp8(a), to_fp8(b)
    elif dtype == torch.float16:
        a, b = to_fp16(a), to_fp16(b)
    elif dtype == torch.bfloat16:
        a, b = to_bf16(a), to_bf16(b)
    else:
        raise ValueError("unsupported dtype")

    b_compressed, e = ops.cutlass_sparse_compress(b.t())
    check_compress_decompress_invariance(dtype, b, b_compressed, e)

    # Compressed B, Metadata, Original A, B
    return b_compressed, e, a, b


@pytest.mark.skipif(not sparse_cutlass_supported(),
                    reason="Sparse CUTLASS is not supported on this GPU type.")
# Test working with a subset of A and B for sparse matmul
def test_cutlass_sparse_subset():

    big_m = 1024
    m, n, k = 512, 512, 512

    # Create tensors
    b_comp, e, whole_a, b = make_rand_sparse_tensors(torch.float8_e4m3fn,
                                                     big_m, n, k)
    a = whole_a[0:m, 0:k]
    scale_a = torch.randn((1, 1), device="cuda", dtype=torch.float32) / 10
    scale_b = torch.randn((1, 1), device="cuda", dtype=torch.float32) / 10

    out = ops.cutlass_scaled_sparse_mm(a,
                                       b_comp,
                                       e,
                                       scale_a,
                                       scale_b,
                                       out_dtype=torch.bfloat16)
    baseline = baseline_scaled_mm(a,
                                  b,
                                  scale_a,
                                  scale_b,
                                  out_dtype=torch.bfloat16)

    torch.testing.assert_close(out, baseline, rtol=1e-1, atol=1e0)


MNK_FACTORS = [
    (1, 256, 128),
    (1, 16384, 1024),
    (1, 24576, 512),
    (16, 256, 512),
    (16, 16384, 128),
    (16, 24576, 4096),
    (32, 8192, 4096),
    (32, 16384, 4096),
    (33, 1024, 1024),
    (33, 8192, 128),
    (64, 2048, 512),
    (64, 16384, 1024),
    (100, 8192, 512),
    (128, 32768, 4096),
    (256, 4096, 4096),
    (512, 256, 1024),
    (512, 8192, 4096),
    (512, 16384, 128),
    (512, 24576, 128),
]


# Test working with a subset of A and B for sparse matmul
@pytest.mark.skipif(not sparse_cutlass_supported(),
                    reason="Sparse CUTLASS is not supported on this GPU type.")
@pytest.mark.parametrize("m, n, k", MNK_FACTORS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_bias", [True, False])
def test_cutlass_sparse_gemm(m: int, k: int, n: int, dtype: type[torch.dtype],
                             use_bias: bool):

    # Create tensors
    b_comp, e, a, b = make_rand_sparse_tensors(dtype, m, n, k)
    scale_a = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((1, 1), device="cuda", dtype=torch.float32)

    bias = torch.rand((n, ), device="cuda", dtype=dtype) if use_bias else None

    out = ops.cutlass_scaled_sparse_mm(a,
                                       b_comp,
                                       e,
                                       scale_a,
                                       scale_b,
                                       out_dtype=dtype,
                                       bias=bias)

    baseline = baseline_scaled_mm(a,
                                  b,
                                  scale_a,
                                  scale_b,
                                  out_dtype=dtype,
                                  bias=bias)

    torch.testing.assert_close(out, baseline, rtol=1e-2, atol=3e-1)


@pytest.mark.skipif(not sparse_cutlass_supported(),
                    reason="Sparse CUTLASS is not supported on this GPU type.")
@pytest.mark.parametrize("m, k, n", MNK_FACTORS)
@pytest.mark.skipif(not current_platform.has_device_capability(89),
                    reason="FP8 is not supported on this GPU type.")
@pytest.mark.parametrize("use_bias", [True, False])
def test_cutlass_sparse_fp8_gemm(m: int, n: int, k: int, use_bias: bool):

    # Create tensors
    b_comp, e, a, b = make_rand_sparse_tensors(torch.float8_e4m3fn, m, n, k)
    scale_a = (torch.randn((1, 1), device="cuda", dtype=torch.float32))
    scale_b = (torch.randn((1, 1), device="cuda", dtype=torch.float32))
    out_dtype = torch.bfloat16

    bias = torch.rand(
        (n, ), device="cuda", dtype=out_dtype) * 10 if use_bias else None

    out = ops.cutlass_scaled_sparse_mm(a,
                                       b_comp,
                                       e,
                                       scale_a,
                                       scale_b,
                                       out_dtype=out_dtype,
                                       bias=bias)

    baseline = baseline_scaled_mm(a,
                                  b,
                                  scale_a,
                                  scale_b,
                                  out_dtype=out_dtype,
                                  bias=bias)

    torch.testing.assert_close(out, baseline, rtol=1e-2, atol=3e-1)


@pytest.mark.skipif(not sparse_cutlass_supported(),
                    reason="Sparse CUTLASS is not supported on this GPU type.")
@pytest.mark.parametrize("m,k,n", MNK_FACTORS)
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_cutlass_sparse_int8_gemm(m: int, n: int, k: int, per_act_token: bool,
                                  per_out_ch: bool, use_bias: bool):

    # Create tensors
    b_comp, e, a, b = make_rand_sparse_tensors(torch.int8, m, n, k)
    scale_a = (torch.randn((1, 1), device="cuda", dtype=torch.float32))
    scale_b = (torch.randn((1, 1), device="cuda", dtype=torch.float32))
    out_dtype = torch.bfloat16

    bias = torch.rand(
        (n, ), device="cuda", dtype=out_dtype) * 10 if use_bias else None

    out = ops.cutlass_scaled_sparse_mm(a,
                                       b_comp,
                                       e,
                                       scale_a,
                                       scale_b,
                                       out_dtype=out_dtype,
                                       bias=bias)

    baseline = baseline_scaled_mm(a,
                                  b,
                                  scale_a,
                                  scale_b,
                                  out_dtype=out_dtype,
                                  bias=bias)

    torch.testing.assert_close(out, baseline, rtol=1e0, atol=2e0)
