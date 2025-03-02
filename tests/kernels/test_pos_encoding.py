from itertools import accumulate, product
from typing import Dict, List, Optional

import pytest
import torch

from aphrodite.modeling.layers.rotary_embedding import (RotaryEmbedding,
                                                        get_rope)

from .allclose_default import get_default_atol, get_default_rtol

IS_NEOX_STYLE = [True, False]
DTYPES = [torch.half, torch.bfloat16, torch.float]
HEAD_SIZES = [64, 80, 96, 112, 120, 128, 192, 256]
ROTARY_DIMS = [None, 32]  # None means rotary dim == head size
NUM_HEADS = [7, 17]  # Arbitrary values for testing
BATCH_SIZES = [1, 5]  # Arbitrary values for testing
SEQ_LENS = [11, 8192]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rotary_embedding(
    is_neox_style: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    if rotary_dim is None:
        rotary_dim = head_size
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style)
    rope = rope.to(dtype=dtype)

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query = torch.randn(batch_size,
                        seq_len,
                        num_heads * head_size,
                        dtype=dtype)
    key = torch.randn_like(query)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope.forward_native(positions, query, key)
    out_query, out_key = rope.forward(positions, query, key)
    # Compare the results.
    torch.testing.assert_close(out_query,
                               ref_query,
                               atol=get_default_atol(out_query),
                               rtol=get_default_rtol(out_query))
    torch.testing.assert_close(out_key,
                               ref_key,
                               atol=get_default_atol(out_key),
                               rtol=get_default_rtol(out_key))


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_batched_rotary_embedding(
    is_neox_style: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style, {
        "type": "linear",
        "factor": (1, )
    })
    rope = rope.to(dtype=dtype)

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query = torch.randn(batch_size,
                        seq_len,
                        num_heads * head_size,
                        dtype=dtype)
    key = torch.randn_like(query)

    # NOTE: The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope.forward_native(positions, query, key)
    out_query, out_key = rope.forward(positions,
                                      query,
                                      key,
                                      offsets=torch.zeros(batch_size * seq_len,
                                                          dtype=torch.long,
                                                          device=device))
    # Compare the results.
    torch.testing.assert_close(out_query,
                               ref_query,
                               atol=get_default_atol(out_query),
                               rtol=get_default_rtol(out_query))
    torch.testing.assert_close(out_key,
                               ref_key,
                               atol=get_default_atol(out_key),
                               rtol=get_default_rtol(out_key))


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_batched_rotary_embedding_multi_lora(
    is_neox_style: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    scaling_factors: List[int] = [1, 2, 4]
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style, {
        "type": "linear",
        "factor": tuple(scaling_factors)
    })
    rope = rope.to(dtype=dtype)

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query = torch.randn(batch_size,
                        seq_len,
                        num_heads * head_size,
                        dtype=dtype)
    key = torch.randn_like(query)

    offset_map = torch.tensor(
        list(
            accumulate([0] + [
                max_position * scaling_factor * 2
                for scaling_factor in scaling_factors[:-1]
            ])))
    query_types = torch.randint(0,
                                len(scaling_factors), (batch_size, seq_len),
                                device=device)
    query_offsets = offset_map[query_types]

    # NOTE: The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope.forward_native(positions, query, key,
                                             query_offsets)
    out_query, out_key = rope.forward(positions, query, key,
                                      query_offsets.flatten())
    # Compare the results.
    torch.testing.assert_close(out_query,
                               ref_query,
                               atol=get_default_atol(out_query),
                               rtol=get_default_rtol(out_query))
    torch.testing.assert_close(out_key,
                               ref_key,
                               atol=get_default_atol(out_key),
                               rtol=get_default_rtol(out_key))


@torch.inference_mode()
def test_rope_module_cache():
    MAX_POSITIONS = [123, 1234]
    BASES = [10000, 1000000]
    ROPE_SCALINGS = (None, {
        "type": "linear",
        "factor": (1, )
    }, {
        "type": "dynamic",
        "factor": 1
    })
    settings = (HEAD_SIZES, ROTARY_DIMS, MAX_POSITIONS, BASES, IS_NEOX_STYLE,
                ROPE_SCALINGS, DTYPES)
    rope_setting_id_map: Dict[str, int] = {}
    for setting in product(*settings):
        head_size, rotary_dim, max_position, base, \
            is_neox_stype, rope_scaling, dtype = setting
        if rotary_dim is None:
            rotary_dim = head_size
        rope = get_rope(head_size, rotary_dim, max_position, base,
                        is_neox_stype, rope_scaling, dtype)
        # different settings cannot share the same rope module
        assert id(rope) not in rope_setting_id_map.values()
        assert all(x.dtype == dtype for x in rope.buffers())
        assert all(x.dtype == dtype for x in rope.parameters())
        rope_setting_id_map[str(setting)] = id(rope)

    for setting in product(*settings):
        head_size, rotary_dim, max_position, base, \
            is_neox_stype, rope_scaling, dtype = setting
        if rotary_dim is None:
            rotary_dim = head_size
        rope = get_rope(head_size, rotary_dim, max_position, base,
                        is_neox_stype, rope_scaling, dtype)
        # check if cache take effect
        assert id(rope) == rope_setting_id_map[str(setting)]


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rotary_embedding_triton(
    is_neox_style: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    """Test Triton RoPE implementation against PyTorch native."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    # Set random seed for reproducibility
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    
    if rotary_dim is None:
        rotary_dim = head_size
        
    # Initialize RoPE
    rope = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
    )
    
    # Generate test inputs
    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query = torch.randn(batch_size, seq_len, num_heads * head_size, dtype=dtype)
    key = torch.randn_like(query)
    
    # Clone inputs for each implementation
    q_native = query.clone()
    k_native = key.clone()
    q_triton = query.clone()
    k_triton = key.clone()
    
    # Run native implementation
    q_native, k_native = rope.forward_native(positions, q_native, k_native)
    
    # Run Triton implementation
    q_triton, k_triton = rope.forward_triton(positions, q_triton, k_triton)
    
    # Compare results
    torch.testing.assert_close(
        q_triton, 
        q_native,
        atol=get_default_atol(q_native),
        rtol=get_default_rtol(q_native),
        msg=("Query tensors do not match between Triton and native "
             "implementations"
             f"for head_size={head_size}, dtype={dtype}, "
             f"is_neox_style={is_neox_style}")
    )

    torch.testing.assert_close(
        k_triton,
        k_native,
        atol=get_default_atol(k_native),
        rtol=get_default_rtol(k_native),
        msg=("Key tensors do not match between Triton and native "
             "implementations"
             f"for head_size={head_size}, dtype={dtype}, "
             f"is_neox_style={is_neox_style}")
    )


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rotary_embedding_performance(
    is_neox_style: bool,
    head_size: int,
    dtype: torch.dtype,
    device: str,
) -> None:
    """Compare performance between Triton and native implementations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    torch.set_default_device(device)
    
    # Test parameters
    batch_size = 32
    seq_len = 2048
    num_heads = 32
    max_position = 8192
    base = 10000
    rotary_dim = head_size
    num_warmup = 10
    num_repeats = 100
    
    # Initialize RoPE
    rope = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
    )
    
    # Generate inputs
    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query = torch.randn(batch_size, seq_len, num_heads * head_size, dtype=dtype)
    key = torch.randn_like(query)
    
    # Warmup
    for _ in range(num_warmup):
        q_native, k_native = rope.forward_native(positions, query.clone(),
                                                 key.clone())
        q_triton, k_triton = rope.forward_triton(positions, query.clone(),
                                                 key.clone())
    
    # Time native implementation
    torch.cuda.synchronize()
    start_native = torch.cuda.Event(enable_timing=True)
    end_native = torch.cuda.Event(enable_timing=True)
    
    start_native.record()
    for _ in range(num_repeats):
        q_native, k_native = rope.forward_native(positions, query.clone(),
                                                 key.clone())
    end_native.record()
    
    torch.cuda.synchronize()
    native_time = start_native.elapsed_time(end_native) / num_repeats
    
    # Time Triton implementation
    torch.cuda.synchronize()
    start_triton = torch.cuda.Event(enable_timing=True)
    end_triton = torch.cuda.Event(enable_timing=True)
    
    start_triton.record()
    for _ in range(num_repeats):
        q_triton, k_triton = rope.forward_triton(positions, query.clone(),
                                                 key.clone())
    end_triton.record()
    
    torch.cuda.synchronize()
    triton_time = start_triton.elapsed_time(end_triton) / num_repeats
    
    # Assert Triton implementation is not significantly slower
    assert triton_time <= native_time * 1.2, (
        f"Triton implementation is significantly slower than native "
        f"(Triton: {triton_time:.3f}ms, Native: {native_time:.3f}ms) "
        f"for head_size={head_size}, dtype={dtype}, "
        f"is_neox_style={is_neox_style}"
    )
