# Adapted from
# https://github.com/stackav-oss/conch/blob/e4ac60c47ebcaf76a5fb38982d6a8bbc169f65f6/conch/kernels/vllm/reshape_and_cache.py

from typing import Final

import torch

from aphrodite.triton_utils import tl, triton


@triton.jit  # type: ignore[misc]
def _reshape_and_cache_kernel(
    # Pointers to tensors
    key_ptr: tl.tensor,
    value_ptr: tl.tensor,
    key_cache_ptr: tl.tensor,
    value_cache_ptr: tl.tensor,
    slot_mapping_ptr: tl.tensor,
    k_scale_ptr: tl.tensor,
    v_scale_ptr: tl.tensor,
    # Scalars
    head_size: int,
    cache_block_size: int,
    # Strides of relevant tensors
    kv_token_stride: tl.int64,
    kv_head_stride: tl.int64,
    kv_cache_page_stride: tl.int64,
    kv_cache_block_stride: tl.int64,
    kv_cache_head_stride: tl.int64,
    # Constexprs
    cxpr_head_size_padded: tl.constexpr,
    cxpr_apply_fp8_scaling: tl.constexpr,
) -> None:
    """Implementation of reshape_and_cache kernel.

    Args:
        key_ptr: Pointer to tensor of new key vectors, shape: (num_tokens, num_kv_heads, head_size).
        value_ptr: Pointer to tensor of new value vectors, shape: (num_tokens, num_kv_heads, head_size).
        key_cache_ptr: Pointer to tensor of key cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        value_cache_ptr: Pointer to tensor of value cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        slot_mapping_ptr: Pointer to slot mapping tensor, shape: (num_tokens,).
        k_scale_ptr: Pointer to Fp8 scaling factor for k.
        v_scale_ptr: Pointer to Fp8 scaling factor for v.
        head_size: Dimension of attention head.
        cache_block_size: Size of each cache block / page in the KV cache.
        kv_token_stride: Stride of key/value tensors in 0th dimension.
        kv_head_stride: Stride of key/value tensors in 1st dimension.
        kv_cache_page_stride: Stride of key/value cache tensors in 0th dimension.
        kv_cache_block_stride: Stride of key/value cache tensors in 1st dimension.
        kv_cache_head_stride: Stride of key/value cache tensors in 2nd dimension.
        cxpr_head_size_padded: Head size padded to the next power of two.
        cxpr_apply_fp8_scaling: Whether or not to apply FP8 scaling.
    """
    # What token is this program processing?
    token_index = tl.program_id(0)
    kv_head_index = tl.program_id(1)

    # Get index of slot for this token from mapping tensor
    slot_index = tl.load(slot_mapping_ptr + token_index)

    # If slot index is negative its a padding token that should be ignored
    if slot_index < 0:
        return

    # Calculate index of page (value in range(0, num_pages))
    page_index = slot_index // cache_block_size
    # Calculate entry index inside of a cache block/page for this slot (value in range(0, cache_block_size))
    entry_index = slot_index % cache_block_size

    # Calculate offset into key/value tensors to get to the token for this program
    token_offset = token_index * kv_token_stride
    # Offset for this KV head
    head_offset = kv_head_index * kv_head_stride

    # Offsets for each element of the head
    head_element_offsets = tl.arange(0, cxpr_head_size_padded)
    # Mask to ensure we only load valid head elements (if head_size is not power of two)
    head_mask = head_element_offsets < head_size

    # Load key/value vectors for this token/head, shape: (cxpr_head_size_padded,)
    key = tl.load(key_ptr + token_offset + head_offset + head_element_offsets, mask=head_mask, other=0.0)
    value = tl.load(value_ptr + token_offset + head_offset + head_element_offsets, mask=head_mask, other=0.0)

    # Apply FP8 scaling if necessary
    if cxpr_apply_fp8_scaling:
        # Load, invert, and apply scaling factors
        k_scale = tl.load(k_scale_ptr)
        k_scale = 1.0 / k_scale
        key = (key * k_scale).to(key_cache_ptr.dtype.element_ty)

        v_scale = tl.load(v_scale_ptr)
        v_scale = 1.0 / v_scale
        value = (value * v_scale).to(value_cache_ptr.dtype.element_ty)

    # Calculate offset into key/value cache tensors to get to the cache block we're copying into
    page_offset = page_index * kv_cache_page_stride
    # Calculate offset in a cache block to get to the entry for we're copying into
    kv_cache_entry_offset = entry_index * kv_cache_block_stride

    # Store key/value vectors into cache
    tl.store(
        key_cache_ptr + page_offset + kv_cache_entry_offset + head_offset + head_element_offsets, key, mask=head_mask
    )
    tl.store(
        value_cache_ptr + page_offset + kv_cache_entry_offset + head_offset + head_element_offsets,
        value,
        mask=head_mask,
    )


def reshape_and_cache_launcher(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    strict: bool = False,
) -> None:
    """Launch reshape_and_cache kernel.

    Args:
        key: New key vectors, shape: (num_tokens, num_kv_heads, head_size).
        value: New value vectors, shape: (num_tokens, num_kv_heads, head_size).
        key_cache: Key cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        value_cache: Value cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        slot_mapping: Tensor listing what slots in the cache each key/value vector should be placed in, shape: (num_tokens,).
        kv_cache_dtype: String datatype of kv cache elements.
        k_scale: Fp8 scaling factor for k.
        v_scale: Fp8 scaling factor for v.
    """
    # Assume sizes already checked if calling launcher. For interface with strict size checking, call `ops.reshape_and_cache()` with `strict=True`.
    _, num_kv_heads, head_size = key.shape
    _, cache_block_size, _, _ = key_cache.shape

    # Note: In vLLM v1, slot_mapping is the only tensor that can be trusted to tell the correct number of tokens
    num_tokens = slot_mapping.size(0)

    if strict:
        assert key.shape == value.shape
        assert key_cache.shape == value_cache.shape

        assert key.stride(0) == value.stride(0)
        assert key.stride(1) == value.stride(1)
        assert key.stride(2) == value.stride(2)
        assert key.stride(2) == 1

        assert key_cache.stride(0) == value_cache.stride(0)
        assert key_cache.stride(1) == value_cache.stride(1)
        assert key_cache.stride(2) == value_cache.stride(2)
        assert key_cache.stride(3) == value_cache.stride(3)
        assert key_cache.stride(3) == 1

    apply_fp8_scaling: tl.constexpr = "fp8" in kv_cache_dtype

    if strict and apply_fp8_scaling:
        assert k_scale is not None
        assert v_scale is not None
        assert k_scale.numel() == 1
        assert v_scale.numel() == 1

    # Parallelize over the number of tokens and number of kv heads
    grid = (num_tokens, num_kv_heads)

    # Launch kernel
    _reshape_and_cache_kernel[grid](
        # Tensors
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        # Scalars
        head_size=head_size,
        cache_block_size=cache_block_size,
        # Strides of relevant tensors
        kv_token_stride=key.stride(0),
        kv_head_stride=key.stride(1),
        kv_cache_page_stride=key_cache.stride(0),
        kv_cache_block_stride=key_cache.stride(1),
        kv_cache_head_stride=key_cache.stride(2),
        # Constexprs
        cxpr_head_size_padded=triton.next_power_of_2(head_size),
        cxpr_apply_fp8_scaling=apply_fp8_scaling,
    )


def _validate_sizes(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Validate sizes of input tensors.

    Args:
        key: New key vectors, shape: (num_tokens, num_kv_heads, head_size).
        value: New value vectors, shape: (num_tokens, num_kv_heads, head_size).
        key_cache: Key cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        value_cache: Value cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        slot_mapping: Tensor listing what slots in the cache each key/value vector should be placed in, shape: (num_tokens,).

    Raises:
        ValueError if sizes are mismatched between various input tensors.
    """
    if key.shape != value.shape:
        msg = f"key.shape ({key.shape}) does not match value.shape ({value.shape})"
        raise ValueError(msg)

    expected_kv_dims: Final = 3
    if (key_dims := len(key.shape)) != expected_kv_dims:
        msg = f"Number of dimensions in key ({key_dims}) did not match expected ({expected_kv_dims})"
        raise ValueError(msg)

    _, num_kv_heads_kv, head_size_kv = key.shape

    if key_cache.shape != value_cache.shape:
        msg = f"key_cache.shape ({key_cache.shape}) does not match value_cache.shape ({value_cache.shape})"
        raise ValueError(msg)

    expected_kv_cache_dims: Final = 4
    if (key_cache_dims := len(key_cache.shape)) != expected_kv_cache_dims:
        msg = f"Number of dimensions in key cache ({key_cache_dims}) did not match expected ({expected_kv_cache_dims})"
        raise ValueError(msg)

    _, _, num_kv_heads_kvc, head_size_kvc = key_cache.shape

    if num_kv_heads_kv != num_kv_heads_kvc:
        msg = f"Number of kv heads in key/value tensors ({num_kv_heads_kv}) does not match number of kv heads in key/value cache tensors ({num_kv_heads_kvc})"
        raise ValueError(msg)

    if head_size_kv != head_size_kvc:
        msg = f"Head size in key/value tensors ({head_size_kv}) does not match head size in key/value cache tensors ({head_size_kvc})"
        raise ValueError(msg)

    expected_slot_mapping_dims: Final = 1
    if (slot_mapping_dims := len(slot_mapping.shape)) != expected_slot_mapping_dims:
        msg = f"Number of dimensions in slot mapping ({slot_mapping_dims}) did not match expected ({expected_slot_mapping_dims})"
        raise ValueError(msg)


def _validate_kv_cache_dtype(kv_cache_dtype: str) -> None:
    """Validate that KV Cache Dtype is valid and return whether to enable FP8 scaling.

    Args:
        kv_cache_dtype: String representing desired datatype of KV-cache.

    Raises:
        ValueError if kv_cache_dtype is invalid.
    """
    fp8_dtypes: Final = {"fp8", "fp8_e4m3"}
    allowed_dtypes: Final = {"auto"}.union(fp8_dtypes)

    if kv_cache_dtype not in allowed_dtypes:
        msg = f"Unsupported kv_cache_dtype: '{kv_cache_dtype}'"
        raise ValueError(msg)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    strict: bool = False,
) -> None:
    """Reshape key/value vectors and add them to the cache.

    Args:
        key: New key vectors, shape: (num_tokens, num_kv_heads, head_size).
        value: New value vectors, shape: (num_tokens, num_kv_heads, head_size).
        key_cache: Key cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        value_cache: Value cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        slot_mapping: Tensor listing what slots in the cache each key/value vector should be placed in, shape: (num_tokens,).
        kv_cache_dtype: String datatype of kv cache elements.
        k_scale: Fp8 scaling factor for k.
        v_scale: Fp8 scaling factor for v.
    """
    if strict:
        # Verify input sizes/tensor shapes
        _validate_sizes(key, value, key_cache, value_cache, slot_mapping)

        # Validate kv cache dtype is valid
        _validate_kv_cache_dtype(kv_cache_dtype)

    # Call kernel launch wrapper
    reshape_and_cache_launcher(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
        strict=strict,
    )
