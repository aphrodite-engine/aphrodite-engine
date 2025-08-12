import os
import pytest
import torch


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_import_and_register_ops():
    # Import the MPS extension to register MPS implementations
    import aphrodite._C  # noqa: F401

    # Check op namespaces are present
    assert hasattr(torch.ops, "_C"), "_C namespace missing"
    assert hasattr(torch.ops, "_C_cache_ops"), "_C_cache_ops namespace missing"
    # Check attention ops are registered
    assert hasattr(torch.ops._C, "paged_attention_v1")
    assert hasattr(torch.ops._C, "paged_attention_v2")
    # Check cache ops are registered
    assert hasattr(torch.ops._C_cache_ops, "convert_fp8")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_convert_fp8_roundtrip_shapes():
    import aphrodite._C  # ensure MPS kernels are loaded
    from aphrodite import _custom_ops as cops

    # Prepare a small tensor on MPS
    numel = 1024
    x = torch.randn(numel, device="mps", dtype=torch.float16)
    y = torch.empty(numel, device="mps", dtype=torch.uint8)

    # Should not raise
    cops.convert_fp8(y, x, 1.0, "fp8")

    assert y.dtype == torch.uint8
    assert y.is_mps
    assert y.numel() == numel


