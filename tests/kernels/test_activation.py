import time
from typing import Type

import pytest
import torch

from aphrodite.modeling.layers.activation import (FastGELU, GeluAndMul,
                                                  NewGELU, QuickGELU,
                                                  ReLUSquaredActivation,
                                                  SiluAndMul)
from tests.kernels.utils import opcheck

from .allclose_default import get_default_atol, get_default_rtol

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 4096, 5120, 13824]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_act_and_mul(
    activation: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)
    if activation == "silu":
        layer = SiluAndMul()
        fn = torch.ops._C.silu_and_mul
    elif activation == "gelu":
        layer = GeluAndMul(approximate="none")
        fn = torch.ops._C.gelu_and_mul
    elif activation == "gelu_tanh":
        layer = GeluAndMul(approximate="tanh")
        fn = torch.ops._C.gelu_tanh_and_mul
    out = layer(x)
    ref_out = layer.forward_native(x)
    # The SiLU and GELU implementations are equivalent to the native PyTorch
    # implementations, so we can do exact comparison.
    torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)


    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d, ))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    opcheck(fn, (out, x))

@pytest.mark.parametrize("activation", [(FastGELU, torch.ops._C.gelu_fast),
                                        (NewGELU, torch.ops._C.gelu_new),
                                        (QuickGELU, torch.ops._C.gelu_quick)])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_activation(
    activation: Type[torch.nn.Module],
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, d, dtype=dtype)
    layer = activation[0]()
    fn = activation[1]
    out = layer(x)
    ref_out = layer.forward_native(x)
    torch.testing.assert_close(out,
                               ref_out,
                               atol=get_default_atol(out),
                               rtol=get_default_rtol(out))
    out = torch.empty_like(x)
    opcheck(fn, (out, x))


@pytest.mark.parametrize("activation_cls, kwargs", [
    (SiluAndMul, {}),
    (GeluAndMul, {"approximate": "none"}),
    (GeluAndMul, {"approximate": "tanh"}),
    (NewGELU, {}),
    (FastGELU, {}),
    (QuickGELU, {}),
    (ReLUSquaredActivation, {}),
])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_activation_triton(
    activation_cls, kwargs, num_tokens, d, dtype, seed, device):
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    activation = activation_cls(**kwargs).to(device=device, dtype=dtype)
    # Input shape is (num_tokens, 2*d) for these activations.
    x = torch.randn(num_tokens, 2 * d, dtype=dtype, device=device)

    native_out = activation.forward_native(x)
    triton_out = activation.forward_triton(x)

    torch.testing.assert_close(triton_out, native_out, atol=1e-2, rtol=1e-2)


# TODO: enable this test after fixing the performance issue
@pytest.mark.skip("skipping performance test")
@pytest.mark.parametrize("activation_cls, kwargs", [
    (SiluAndMul, {}),
    (GeluAndMul, {"approximate": "none"}),
    (GeluAndMul, {"approximate": "tanh"}),
    (NewGELU, {}),
    (FastGELU, {}),
    (QuickGELU, {}),
    (ReLUSquaredActivation, {}),
])
@pytest.mark.parametrize("batch_size, seq_len, hidden_size", [
    (1, 2048, 4096),
    (32, 512, 4096),
])
@torch.inference_mode()
def test_activation_performance(
    activation_cls, kwargs, batch_size: int, seq_len: int, 
    hidden_size: int, device: str = "cuda"
) -> None:
    """Test that Triton implementation performance is close to CUDA.
    Note: Performance in isolation might not reflect real-world performance
    where activation is part of a larger pipeline."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.set_default_device(device)
    activation = activation_cls(**kwargs).to(device=device, dtype=torch.float16)

    # For SiluAndMul and GeluAndMul, input shape needs 2*hidden_size
    if activation_cls in [SiluAndMul, GeluAndMul]:
        x = torch.randn(batch_size, seq_len, 2 * hidden_size, 
                       dtype=torch.float16, device=device)
    else:
        x = torch.randn(batch_size, seq_len, hidden_size, 
                       dtype=torch.float16, device=device)

    # Warmup
    for _ in range(10):
        activation.forward_cuda(x)
        activation.forward_triton(x)

    # Time CUDA implementation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        activation.forward_cuda(x)
    torch.cuda.synchronize()
    cuda_time = time.perf_counter() - start

    # Time Triton implementation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        activation.forward_triton(x)
    torch.cuda.synchronize()
    triton_time = time.perf_counter() - start

    # Must be within 1% for inference shapes (batch_size=1)
    # or within 20% for other shapes
    max_slowdown = 1.01 if batch_size == 1 else 1.2

    assert triton_time <= cuda_time * max_slowdown, (
        f"{activation_cls.__name__} Triton implementation is significantly "
        "slower than CUDA "
        f"(Triton: {triton_time:.3f}s, CUDA: {cuda_time:.3f}s) "
        f"for shape (batch={batch_size}, seq={seq_len}, hidden={hidden_size}) "
        f"slowdown : {(triton_time - cuda_time) / cuda_time * 100:.2f}%"
    )
