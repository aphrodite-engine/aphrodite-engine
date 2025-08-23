# Adapted from
# https://github.com/stackav-oss/conch/blob/e4ac60c47ebcaf76a5fb38982d6a8bbc169f65f6/conch/kernels/attention/paged_attention.py

"""Triton implementation of Flash Attention w/ Paged KV Cache + FlashDecoding.
"""

from dataclasses import dataclass
from typing import Final

import torch

from aphrodite.platforms import current_platform
from aphrodite.triton_utils import libdevice, tl, triton


@triton.jit  # type: ignore[misc]
def _paged_attention_compute_splits_kernel(  # noqa: PLR0913, PLR0915
    # Pointers to tensors
    output_scratchpad_ptr: tl.tensor,  # (batch_size, max_num_blocks_per_sequence, num_query_heads, head_size)
    lse_scratchpad_ptr: tl.tensor,  # (batch_size, max_num_blocks_per_sequence, num_query_heads)
    query_ptr: tl.tensor,  # (batch_size, num_query_heads, head_size)
    key_cache_ptr: tl.tensor,  # (num_cache_blocks, cache_block_size, num_kv_heads, head_size)
    value_cache_ptr: tl.tensor,  # (num_cache_blocks, cache_block_size, num_kv_heads, head_size)
    block_table_ptr: tl.tensor,  # (batch_size, max_num_blocks_per_sequence)
    seq_lens_ptr: tl.tensor,  # (batch_size, )
    k_scale_ptr: tl.tensor,  # (1,)
    v_scale_ptr: tl.tensor,  # (1,)
    # Scalar arguments
    scale: float,
    num_cache_blocks_per_split: int,
    softcap: float,
    # Sizes of tensors above
    head_size: int,  # output.shape[3]
    query_group_size: int,  # num_query_heads // num_kv_heads
    # Strides for tensors above
    output_scratchpad_batch_stride: int,  # output_scratchpad.stride(0)
    output_scratchpad_cache_block_stride: int,  # output_scratchpad.stride(1)
    output_scratchpad_head_stride: int,  # output_scratchpad.stride(2)
    lse_scratchpad_batch_stride: int,  # lse_scratchpad.stride(0)
    lse_scratchpad_cache_block_stride: int,  # lse_scratchpad.stride(1)
    query_batch_stride: int,  # query.stride(0)
    query_head_stride: int,  # query.stride(1)
    kv_page_stride: int,  # key_cache.stride(0), same for key and value
    kv_cache_block_stride: int,  # key_cache.stride(1), same for key and value
    kv_head_stride: int,  # key_cache.stride(2), same for key and value
    kv_head_element_stride: int,  # key_cache.stride(3), same for key and value
    block_table_batch_stride: int,  # block_table.stride(0)
    # Constexprs
    cxpr_cache_block_size: tl.constexpr,
    cxpr_head_size_padded: tl.constexpr,
    cxpr_query_group_size_padded: tl.constexpr,
    cxpr_is_softcap: tl.constexpr,
    cxpr_apply_fp8_scaling: tl.constexpr,
    cxpr_is_rocm: tl.constexpr,
) -> None:
    """PagedAttention kernel: compute attention for a split block.

    Args:
        output_scratchpad_ptr: Pointer to tensor as scratchpad for output of each cache block, shape: (batch_size, max_num_blocks_per_sequence, num_query_heads, head_size).
        lse_scratchpad_ptr: Pointer to tensor as scratchpad for log-sum-exp of each cache block, shape: (batch_size, max_num_blocks_per_sequence, num_query_heads).
        query_ptr: Pointer to tensor storing the query, shape: (batch_size, num_query_heads, head_size).
        key_cache_ptr: Tensor with cached K values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache_ptr: Tensor with cached V values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        block_table_ptr: Pointer to tensor storing the mapping from batch to cache blocks, shape: (batch_size, max_num_blocks_per_sequence).
        seq_lens_ptr: Pointer to tensor holding the current sequence length for each sequence in the batch, shape: (batch_size, ).
        k_scale_ptr: Pointer to tensor holding fp8 scaling factor for k.
        v_scale_ptr: Pointer to tensor holding fp8 scaling factor for v.
        scale: Scaling factor, 1/sqrt(head_size).
        num_cache_blocks_per_split: The maximum number of cache blocks in each split (max num each kernel will process).
        softcap: Logits softcap to apply.
        head_size: Actual head dim, not padded to power-of-two.
        query_group_size: The number of query heads to group together, not padded to power-of-two.
        output_scratchpad_batch_stride: Stride of the output scratchpad tensor in the 0th dimension.
        output_scratchpad_cache_block_stride: Stride of the output scratchpad tensor in the 1st dimension.
        output_scratchpad_head_stride: Stride of the output scratchpad tensor in the 2nd dimension.
        lse_scratchpad_batch_stride: Stride of the log-sum-exp scratchpad tensor in the 0th dimension.
        lse_scratchpad_cache_block_stride: Stride of the log-sum-exp scratchpad tensor in the 1st dimension.
        query_batch_stride: Stride of the query tensor in the 0th dimension.
        query_head_stride: Stride of the query tensor in the 1st dimension.
        kv_page_stride: Stride of the k/v tensors in the 0th dimension.
        kv_cache_block_stride: Stride of the k/v tensors in the 1st dimension.
        kv_head_stride: Stride of the k/v tensors in the 2nd dimension.
        kv_head_element_stride: Stride of the k/v tensors in the 3rd dimension.
        block_table_batch_stride: Stride of the block table tensor in the 0th dimension.
        cxpr_cache_block_size: The size of the cache blocks (must be power of two!).
        cxpr_head_size_padded: The head size of the attention layer padded to the next power of two.
        cxpr_query_group_size_padded: The query group size padded to the next power of two.
        cxpr_is_softcap: Whether or not logits softcapping will be applied.
        cxpr_apply_fp8_scaling: Whether or not to apply FP8 scaling.
        cxpr_is_rocm: Whether or not we're on AMD.
    """
    # What batch is this program processing?
    batch_index = tl.program_id(0)
    # What "split" of the overall data (between 1 and N cache blocks) is this program processing?
    split_index = tl.program_id(1)
    # What KV head is this program processing?
    kv_head_index = tl.program_id(2)

    # Get type that we should be using for accumulating results/intermediate calculations
    dtype = output_scratchpad_ptr.dtype.element_ty

    # Load scalar current_sequence_length for the current sequence in the batch
    current_sequence_length = tl.load(seq_lens_ptr + batch_index)

    # The length of the current sequence will tell us how many cache blocks we need to read
    current_seq_num_cache_blocks = tl.cdiv(current_sequence_length, cxpr_cache_block_size)

    # What is the first cache block index in the set for this split
    starting_cache_block_index = split_index * num_cache_blocks_per_split

    # We launch the same number of splits for all sequences in the batch, so different kernel launches will have different numbers of cache blocks
    # to process. So we may have a case where one sequence in a batch has sufficiently fewer cache blocks to process such that a kernel doesn't have
    # any blocks to process.
    if starting_cache_block_index >= current_seq_num_cache_blocks:
        return

    # Offsets to each element of the padded-to-next-power-of-two head size
    head_offsets = tl.arange(0, cxpr_head_size_padded)
    # Mask to only read valid indices of the actual head size
    head_mask = head_offsets < head_size

    # Offsets for each query vector in the group
    query_group_offsets = tl.arange(0, cxpr_query_group_size_padded)
    # Mask out query heads that are just for padding
    query_group_mask = query_group_offsets < query_group_size

    # Offsets for the queries in this block
    query_offsets = query_group_offsets[:, None] * query_head_stride + head_offsets[None, :]
    # Mask out query heads that are just for padding
    query_mask = query_group_mask[:, None] & head_mask[None, :]

    # Offsets for query vector this batch/head
    query_batch_index_offset = batch_index * query_batch_stride
    query_head_index_offset = (kv_head_index * query_group_size) * query_head_stride

    # Load queries for all of the query heads that correspond to this KV head
    query = tl.load(
        query_ptr + query_batch_index_offset + query_head_index_offset + query_offsets, mask=query_mask, other=0.0
    )

    # Offset for the current kv_head in the key_cache and value_cache
    kv_head_index_offset = kv_head_index * kv_head_stride

    # Pointer arithmetic to get to the entry in the block_table for the current batch_index
    current_block_table_offset = batch_index * block_table_batch_stride
    current_block_table_ptr = block_table_ptr + current_block_table_offset

    # Scratchpad for output from this group of cache blocks
    output = tl.zeros([cxpr_query_group_size_padded, cxpr_head_size_padded], dtype=dtype)
    # Keep running max of softmax numerator (scale * Q * K)
    m_i = tl.full([cxpr_query_group_size_padded], -float("inf"), dtype=dtype)
    # Keep running denominator of softmax
    l_i = tl.full([cxpr_query_group_size_padded], 0.0, dtype=dtype)

    # Offsets for each element of the cache block
    cache_block_offsets = tl.arange(0, cxpr_cache_block_size)

    # Iterate through the cache blocks that this kernel is assigned to
    for relative_cache_block_index in range(num_cache_blocks_per_split):
        # Get the actual index of the cache block
        cache_block_index = starting_cache_block_index + relative_cache_block_index

        # If our cache block index is greater than or equal to the number of active cache blocks for the current sequence,
        # skip this block
        # Note: `break` / `continue` is unsupported in Triton, so we have to nest here
        if cache_block_index < current_seq_num_cache_blocks:
            # Calculate number of entries in this cache block (will be a value between 1 and cache_block_size)
            num_entries_in_cache_block = min(
                current_sequence_length - (cache_block_index * cxpr_cache_block_size), cxpr_cache_block_size
            )

            cache_block_mask = cache_block_offsets < num_entries_in_cache_block

            # Offset from the block_table row for the current batch by the number of cache blocks
            current_cache_block_number_ptr = current_block_table_ptr + cache_block_index
            physical_cache_block_number = tl.load(current_cache_block_number_ptr)

            # Calculate address of current cache block
            kv_cache_block_index_offset = physical_cache_block_number * kv_page_stride

            # Load the key block as (cxpr_head_size_padded, cache_block_size)
            # Note: we're loading it transposed here
            key_block_offsets = (
                cache_block_offsets[None, :] * kv_cache_block_stride + kv_head_index_offset + head_offsets[:, None]
            )

            key_block_mask = head_mask[:, None] & cache_block_mask[None, :]

            key_block = tl.load(
                key_cache_ptr + kv_cache_block_index_offset + key_block_offsets, mask=key_block_mask, other=0.0
            )

            if cxpr_apply_fp8_scaling:
                # Dequantize (multiply by scale factor)
                fp8_dtype = tl.float8e4b8 if cxpr_is_rocm else tl.float8e4nv
                k_scale = tl.load(k_scale_ptr)
                key_block = (key_block.to(fp8_dtype, bitcast=True) * k_scale).to(dtype)

            # Multiply query vector by key matrix for this cache block (and apply scaling factor)
            # query.shape -> (query_group_size, head_size)
            # key_block.shape -> (head_size, cache_block_size)
            # qk.shape -> (query_group_size, cache_block_size)
            qk = (scale * tl.dot(query, key_block)).to(dtype)

            # Need to mask out any elements that represent unused cache block entries or padding elements
            # cache_block_mask = tl.arange(0, cxpr_cache_block_size) < num_entries_in_cache_block
            qk_mask = query_group_mask[:, None] & cache_block_mask[None, :]

            # Set masked out elements to -inf
            qk = tl.where(qk_mask, qk, -float("inf")).to(dtype)

            # Handle softcapping
            if cxpr_is_softcap:
                # tanh can only accept fp32 or fp64 arguments
                qk = (softcap * libdevice.tanh((qk / softcap).to(tl.float32))).to(dtype)

            # Reduce maximum between running max and the max of (scale * Q * K) for this cache block
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1)).to(dtype)

            # Calculate numerator of softmax for this cache block
            p = tl.exp((qk - m_ij[:, None]).to(tl.float32)).to(dtype)
            # Need to mask out any elements that represent unused cache block entries or padding elements
            p = tl.where(qk_mask, p, 0.0).to(dtype)

            # Calculate sum of softmax numerator for this cache block
            l_ij = tl.sum(p, axis=1).to(dtype)

            # Calculate correction factor for this cache block
            alpha = tl.exp((m_i - m_ij).to(tl.float32)).to(dtype)

            # Apply scaling factor to running output
            output *= alpha[:, None]

            # Load the value block as (cache_block_size, cxpr_head_size_padded)
            value_block_offsets = (
                cache_block_offsets[:, None] * kv_cache_block_stride + kv_head_index_offset + head_offsets[None, :]
            )

            value_block_mask = cache_block_mask[:, None] & head_mask[None, :]

            value_block = tl.load(
                value_cache_ptr + kv_cache_block_index_offset + value_block_offsets, mask=value_block_mask, other=0.0
            )

            if cxpr_apply_fp8_scaling:
                # Dequantize (multiply by scale factor)
                fp8_dtype = tl.float8e4b8 if cxpr_is_rocm else tl.float8e4nv
                v_scale = tl.load(v_scale_ptr)
                value_block = (value_block.to(fp8_dtype, bitcast=True) * v_scale).to(dtype)

            # Multiply softmax probabilities by value matrix for this cache block
            # p.shape -> (query_group_size, cache_block_size)
            # value_block.shape -> (cache_block_size, head_size)
            # output.shape -> (query_group_size, head_size)
            output += tl.dot(p, value_block).to(dtype)

            # Update running max
            m_i = m_ij
            # Update running denominator
            l_i = l_i * alpha + l_ij

    # Apply correction for denominator of these cache blocks
    output /= l_i[:, None]

    # Calculate offsets to store the output for this head/batch/split
    output_batch_index_offset = batch_index * output_scratchpad_batch_stride
    output_split_index_offset = split_index * output_scratchpad_cache_block_stride
    output_head_index_offset = (kv_head_index * query_group_size) * output_scratchpad_head_stride

    # 2D block of shape (query_group_size_padded, head_size_padded)
    scratch_offsets = query_group_offsets[:, None] * output_scratchpad_head_stride + head_offsets[None, :]
    output_offsets = output_batch_index_offset + output_split_index_offset + output_head_index_offset + scratch_offsets

    # Store output scratchpad results
    tl.store(output_scratchpad_ptr + output_offsets, output, mask=query_mask)

    # Calculate scratchpad log(sum(exp))
    # Note: log() only accepts fp32/fp64 arguments
    lse = m_i + tl.log(l_i.to(tl.float32)).to(dtype)

    # Calculate offsets to store log-sum-exp for this head/batch/split
    lse_scratch_batch_index_offset = batch_index * lse_scratchpad_batch_stride
    lse_scratch_split_index_offset = split_index * lse_scratchpad_cache_block_stride
    lse_scratch_head_index_offsets = kv_head_index * query_group_size
    lse_scratch_offsets = (
        lse_scratch_batch_index_offset
        + lse_scratch_split_index_offset
        + lse_scratch_head_index_offsets
        + query_group_offsets
    )

    # Store lse scratchpad results
    tl.store(lse_scratchpad_ptr + lse_scratch_offsets, lse, mask=query_group_mask)


@triton.jit  # type: ignore[misc]
def _paged_attention_reduce_splits_kernel(  # noqa: PLR0913
    # Pointers to tensors
    output_ptr: tl.tensor,  # (batch_size, num_query_heads, head_size)
    output_scratchpad_ptr: tl.tensor,  # (batch_size, max_num_blocks_per_sequence, num_query_heads, head_size)
    lse_scratchpad_ptr: tl.tensor,  # (batch_size, max_num_blocks_per_sequence, num_query_heads)
    seq_lens_ptr: tl.tensor,  # (batch_size, )
    # Scalars
    num_cache_blocks_per_split: int,
    # Sizes of tensors above
    head_size: int,  # output.shape[2]
    # Strides for tensors above
    output_batch_stride: int,  # output.stride(0)
    output_head_stride: int,  # output.stride(1)
    output_scratchpad_batch_stride: int,  # output_scratchpad.stride(0)
    output_scratchpad_split_stride: int,  # output_scratchpad.stride(1)
    output_scratchpad_head_stride: int,  # output_scratchpad.stride(2)
    lse_scratchpad_batch_stride: int,  # lse_scratchpad.stride(0)
    lse_scratchpad_split_stride: int,  # lse_scratchpad.stride(1)
    # Constexprs
    cxpr_cache_block_size: tl.constexpr,
    cxpr_head_size_padded: tl.constexpr,
) -> None:
    """PagedAttention kernel: reduce results across all splits.

    Args:
        output_ptr: Pointer to tensor for final output, shape: (batch_size, num_query_heads, head_size).
        output_scratchpad_ptr: Pointer to tensor as scratchpad for output of each cache block, shape: (batch_size, max_num_blocks_per_sequence, num_query_heads, head_size).
        lse_scratchpad_ptr: Pointer to tensor as scratchpad for log-sum-exp of each cache block, shape: (batch_size, max_num_blocks_per_sequence, num_query_heads).
        seq_lens_ptr: Pointer to tensor holding the current sequence length for each sequence in the batch, shape: (batch_size, ).
        num_cache_blocks_per_split: The maximum number of cache blocks each split will process.
        head_size: Actual head dim, not padded to power-of-two.
        output_batch_stride: Stride of the output tensor in the 0th dimension.
        output_head_stride: Stride of the output tensor in the 1st dimension.
        output_scratchpad_batch_stride: Stride of the output scratchpad tensor in the 0th dimension.
        output_scratchpad_split_stride: Stride of the output scratchpad tensor in the 1st dimension.
        output_scratchpad_head_stride: Stride of the output scratchpad tensor in the 2nd dimension.
        lse_scratchpad_batch_stride: Stride of the log-sum-exp scratchpad tensor in the 0th dimension.
        lse_scratchpad_split_stride: Stride of the log-sum-exp scratchpad tensor in the 1st dimension.
        cxpr_cache_block_size: The size of the cache blocks (must be power of two!), as constexpr so that we can use for reshaping tensors.
        cxpr_head_size_padded: The head size of the attention layer padded to the next power of two.
    """
    # What batch is this program processing?
    batch_index = tl.program_id(0)
    # What head is this program processing?
    head_index = tl.program_id(1)

    # Get type that we should be using for accumulating results/intermediate calculations
    dtype = output_ptr.dtype.element_ty

    # Accumulator for the output of this batch/head
    output = tl.zeros((cxpr_head_size_padded,), dtype=dtype)
    # Running max of block lse
    # Note: this syntax lets you create a scalar of a specific datatype, see:
    # https://github.com/triton-lang/triton/issues/2939#issuecomment-1892525932
    m_i = tl.full([], -float("inf"), dtype=dtype)
    # Running final scale factor
    l_i = tl.full([], 0.0, dtype=dtype)

    # Load scalar current_sequence_length for the current batch
    current_sequence_length = tl.load(seq_lens_ptr + batch_index)

    # The length of the current sequence will tell us how many cache blocks we need to read
    current_seq_num_cache_blocks = tl.cdiv(current_sequence_length, cxpr_cache_block_size)

    # Offsets to each element of the padded-to-next-power-of-two head size
    head_offsets = tl.arange(0, cxpr_head_size_padded)
    # Mask to only read valid indices of the actual head size
    head_mask = head_offsets < head_size

    num_splits_this_seq = tl.cdiv(current_seq_num_cache_blocks, num_cache_blocks_per_split)

    # Iterate through every cache block for the current sequence
    for split_index in range(num_splits_this_seq):
        # Calculate offsets to load the output scratchpad for this head/batch/cache block
        output_scratchpad_batch_index_offset = batch_index * output_scratchpad_batch_stride
        output_scratchpad_split_index_offset = split_index * output_scratchpad_split_stride
        output_scratchpad_head_index_offset = head_index * output_scratchpad_head_stride
        output_scratchpad_offsets = (
            output_scratchpad_batch_index_offset
            + output_scratchpad_split_index_offset
            + output_scratchpad_head_index_offset
            + head_offsets
        )

        # Load output for this cache block, shape -> (cxpr_head_size_padded,)
        block_output = tl.load(output_scratchpad_ptr + output_scratchpad_offsets, mask=head_mask, other=0.0)

        # Calculate offsets to store log-sum-exp for this head/batch/cache block
        lse_scratch_batch_index_offset = batch_index * lse_scratchpad_batch_stride
        lse_scratch_split_index_offset = split_index * lse_scratchpad_split_stride
        # Load log-sum-exp for this cache block, shape -> scalar
        block_lse = tl.load(
            lse_scratchpad_ptr + lse_scratch_batch_index_offset + lse_scratch_split_index_offset + head_index
        )

        # Reduce running max lse
        m_ij = tl.maximum(m_i, block_lse).to(dtype)

        # Calculate correction factor from previous cache blocks
        # Note: exp() only accepts fp32/fp64 arguments
        alpha = tl.exp((m_i - m_ij).to(tl.float32)).to(dtype)
        # Apply correction factor
        output *= alpha

        # Calculate correction factor from this cache block
        # Note: exp() only accepts fp32/fp64 arguments
        beta = tl.exp((block_lse - m_ij).to(tl.float32)).to(dtype)
        # Apply second correction factor and accumulate running output
        output += (beta * block_output).to(dtype)

        # Update running max
        m_i = m_ij
        # Update running final scale factor
        l_i = l_i * alpha + beta

    # Apply final correction to output
    output /= l_i

    # Calculate offsets to store the output for this head/batch
    output_batch_index_offset = batch_index * output_batch_stride
    output_head_index_offset = head_index * output_head_stride
    output_offsets = output_batch_index_offset + output_head_index_offset + head_offsets
    # Store final result for this head/batch
    tl.store(output_ptr + output_offsets, output, mask=head_mask)


def paged_attention_launcher(  # noqa: PLR0913
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output_scratchpad: torch.Tensor,
    lse_scratchpad: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
    softcap: float = 0.0,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """PagedAttention kernel launcher.

    Args:
        out: Tensor to write the output of the attention calculation, shape: (batch_size, num_heads, head_size).
        query: Query tensor, shape: (batch_size, num_heads, head_size).
        key_cache: Tensor with cached K values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache: Tensor with cached V values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        output_scratchpad: Tensor used as scratchpad to share cache block outputs between two stages, shape: (batch_size, max_num_blocks_per_sequence, num_query_heads, head_size)
        lse_scratchpad: Tensor used as scratchpad to share cache block log-sum-exp between two stages, shape: (batch_size, max_num_blocks_per_sequence, num_query_heads)
        block_table: Tensor storing the mapping from batch to cache blocks, shape: (batch_size, max_num_blocks_per_sequence).
        seq_lens: Tensor with the sequence length of each index in the batch, shape: (batch_size, ).
        scale: Scaling factor, 1/sqrt(head_size).
        softcap: Logit softcap to apply (0.0 means no softcap will be applied).
        kv_cache_dtype: If this dtype is fp8, apply scaling.
        k_scale: Fp8 scaling factor for k.
        v_scale: Fp8 scaling factor for v.
    """
    assert query.shape == out.shape  # noqa: S101
    assert key_cache.shape == value_cache.shape  # noqa: S101
    assert key_cache.stride(0) == value_cache.stride(0)  # noqa: S101
    assert key_cache.stride(1) == value_cache.stride(1)  # noqa: S101
    assert key_cache.stride(2) == value_cache.stride(2)  # noqa: S101
    assert key_cache.stride(3) == value_cache.stride(3)  # noqa: S101
    assert key_cache.stride(3) == 1  # noqa: S101
    assert softcap >= 0.0  # noqa: S101

    allowed_in_out_dtypes = [torch.float32, torch.float16, torch.bfloat16]
    assert query.dtype in allowed_in_out_dtypes  # noqa: S101
    assert out.dtype == query.dtype  # noqa: S101
    assert output_scratchpad.dtype == query.dtype  # noqa: S101
    assert lse_scratchpad.dtype == query.dtype  # noqa: S101

    # Perform unchecked size accesses, assume has already been checked
    batch_size, num_query_heads, head_size = out.shape
    num_cache_blocks, cache_block_size, num_kv_heads, _ = key_cache.shape
    _, max_num_blocks_per_sequence = block_table.shape
    _, max_num_splits, _, _ = output_scratchpad.shape

    assert cache_block_size == triton.next_power_of_2(cache_block_size), "Cache block size must be a power of two!"  # noqa: S101

    # Need sizes to be constexpr in order to reshape tensors in kernel
    cxpr_cache_block_size: tl.constexpr = cache_block_size
    cxpr_head_size_padded: tl.constexpr = triton.next_power_of_2(head_size)

    # For parity with Dao Flash Attention, softcap == 0.0 means no softcapping
    cxpr_is_softcap: tl.constexpr = softcap > 0.0

    cxpr_apply_fp8_scaling: tl.constexpr = kv_cache_dtype == "fp8" or kv_cache_dtype == "fp8_e4m3"
    cxpr_is_rocm: tl.constexpr = current_platform.is_rocm()

    # How many query heads correspond to the same KV head?
    query_group_size = num_query_heads // num_kv_heads
    # We pad this size to be at least 16 so that we can use `tl.dot()` operations inside of the kernel
    cxpr_query_group_size_padded = max(16, triton.next_power_of_2(query_group_size))

    # What is the maximum number of stage 1 kernels to launch per batch/head?
    # Each kernel processes up to {cache_block_size} tokens at a time (in many cases cache_block_size=32 for vLLM), so we can process
    # a sequence up to {max_num_splits * cache_block_size} == 64 * 32 == 2048 tokens before a stage 1 kernel will process multiple cache
    # blocks. This helps to reduce the overhead of kernel launches / split reduction for long sequences.
    # Note: we may need to tune this value for a given HW platform.
    num_splits = min(max_num_blocks_per_sequence, max_num_splits)

    num_cache_blocks_per_split = triton.cdiv(max_num_blocks_per_sequence, num_splits)

    # Use default scaling factor if not provided
    if scale is None:
        scale = float(1.0 / (head_size**0.5))

    if cxpr_apply_fp8_scaling:
        assert k_scale is not None  # noqa: S101
        assert v_scale is not None  # noqa: S101
        assert k_scale.numel() == 1  # noqa: S101
        assert v_scale.numel() == 1  # noqa: S101

    # For computing attention for split block (stage 1): parallelize over batches, cache blocks, and KV heads.
    # Note: if the number of cache blocks in a sequence is very large, it is more efficient to handle multiple blocks
    # in one launch of the stage 1 kernel to reduce the overhead of reduction
    stage1_grid = (batch_size, num_splits, num_kv_heads)

    # Launch stage 1 kernel
    _paged_attention_compute_splits_kernel[stage1_grid](
        # Relevant tensors
        output_scratchpad,
        lse_scratchpad,
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        k_scale,
        v_scale,
        # Scalars
        scale,
        num_cache_blocks_per_split,
        softcap,
        # Sizes of relevant tensors
        head_size,
        query_group_size,
        # Strides of relevant tensors
        output_scratchpad.stride(0),
        output_scratchpad.stride(1),
        output_scratchpad.stride(2),
        lse_scratchpad.stride(0),
        lse_scratchpad.stride(1),
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        block_table.stride(0),
        # Constexpr sizes
        cxpr_cache_block_size,
        cxpr_head_size_padded,
        cxpr_query_group_size_padded,
        cxpr_is_softcap,
        cxpr_apply_fp8_scaling,
        cxpr_is_rocm,
    )

    # For reducing over splits (stage 2): parallelize over batches and query heads
    stage2_grid = (batch_size, num_query_heads)

    # Launch stage 2 kernel
    _paged_attention_reduce_splits_kernel[stage2_grid](
        # Relevant tensors
        out,
        output_scratchpad,
        lse_scratchpad,
        seq_lens,
        # Scalars
        num_cache_blocks_per_split,
        # Sizes of relevant tensors
        head_size,
        # Strides of relevant tensors
        out.stride(0),
        out.stride(1),
        output_scratchpad.stride(0),
        output_scratchpad.stride(1),
        output_scratchpad.stride(2),
        lse_scratchpad.stride(0),
        lse_scratchpad.stride(1),
        # Constexpr sizes
        cxpr_cache_block_size,
        cxpr_head_size_padded,
    )


@dataclass
class PagedAttentionMetadata:
    """Wrapper class holding metadata for PagedAttention kernel."""

    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    num_cache_blocks: int
    max_num_blocks_per_sequence: int
    max_num_splits: int


def _check_output_query_size_compatibility(out: torch.Tensor, query: torch.Tensor) -> None:
    """Check size compatibility of Output and Query tensors.

    Args:
        out: Output tensor.
        query: Query tensor.

    Raises:
        ValueError if sizes are mismatched.
    """
    # Output tensor should be a 3-D tensor of shape (batch_size, num_heads, head_size)
    expected_output_shape_dims: Final = 3

    if len(out_shape := out.shape) != expected_output_shape_dims:
        msg = f"Output tensor has unexpected shape (shape={out_shape}), expected {expected_output_shape_dims}-D tensor"
        raise ValueError(msg)

    # Query tensor should have same shape as output
    if len(query_shape := query.shape) != expected_output_shape_dims:
        msg = f"Query tensor has unexpected shape (shape={query_shape}), expected {expected_output_shape_dims}-D tensor"
        raise ValueError(msg)

    if query_shape != out_shape:
        msg = f"Shape of query and output tensors does not match (query.shape={query_shape}, out.shape={out_shape})"
        raise ValueError(msg)


def _check_kv_cache_size_compatibility(key_cache: torch.Tensor, value_cache: torch.Tensor, head_size: int) -> None:
    """Check size compatibility of KV cache tensor.

    Args:
        key_cache: Key cache tensor.
        value_cache: Key cache tensor.
        head_size: Size of attention head, deduced from out/query tensor sizes.

    Raises:
        ValueError if sizes are mismatched.
    """
    # Key/Value Cache tensor should be a 4-D tensor of shape (num_blocks, cache_block_size, num_kv_heads, head_size)
    expected_kv_cache_shape_dims: Final = 4

    if len(key_cache.shape) != expected_kv_cache_shape_dims:
        msg = f"key_cache tensor has unexpected shape ({key_cache.shape = }), expected {expected_kv_cache_shape_dims}-D tensor"
        raise ValueError(msg)

    if len(value_cache.shape) != expected_kv_cache_shape_dims:
        msg = f"value_cache tensor has unexpected shape ({value_cache.shape = }), expected {expected_kv_cache_shape_dims}-D tensor"
        raise ValueError(msg)

    if key_cache.shape != value_cache.shape:
        msg = f"Shape of key and value cache tensors do not match ({key_cache.shape = }, {value_cache.shape = })"
        raise ValueError(msg)

    if key_cache.size(-1) != head_size:
        msg = (
            f"Last dimension in key/value_cache shape ({key_cache.shape = }) does not match head_size ({head_size = })"
        )


def _check_block_table_size_compatibility(block_table: torch.Tensor, batch_size: int) -> None:
    """Check size compatibility of block_table tensor.

    Args:
        block_table: Block tables tensor.
        batch_size: Expected size of batch.

    Raises:
        ValueError if sizes are mismatched.
    """
    batch_size_block_table: int = block_table.shape[0]

    if batch_size_block_table != batch_size:
        msg = f"Batch size from block_table tensor ({batch_size_block_table}) does not match batch_size from output/query tensors ({batch_size})"
        raise ValueError(msg)


def _determine_max_num_kv_splits(max_seqlen_k: int) -> int:
    # Note: this is a pretty basic heuristic, we could try and do something more sophisticated in the future
    if max_seqlen_k > 8192:
        return 64

    if max_seqlen_k > 2048:
        return 32

    if max_seqlen_k > 1024:
        return 16

    if max_seqlen_k > 512:
        return 8

    if max_seqlen_k > 256:
        return 4

    if max_seqlen_k > 128:
        return 2

    return 1


def _check_size_compatibility(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
) -> PagedAttentionMetadata:
    """Check size compatibility of tensors for PagedAttention and return Metadata if successful.

    Args:
        out: Tensor to write the output of the attention calculation, shape: (batch_size, num_heads, head_size).
        query: Query tensor, shape: (batch_size, num_heads, head_size).
        key_cache: Tensor holding cache for K, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache: Tensor holding cache for V, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        block_table: Tensor storing the mapping from batch to cache blocks, shape: (batch_size, max_num_blocks_per_sequence).

    Raises:
        ValueError if sizes are mismatched.

    Returns:
        Metadata dataclass holding information about tensor sizes.
    """
    _check_output_query_size_compatibility(out, query)
    batch_size, num_query_heads, head_size = out.shape

    _check_kv_cache_size_compatibility(key_cache, value_cache, head_size)
    num_cache_blocks, cache_block_size, num_kv_heads, _ = key_cache.shape

    _check_block_table_size_compatibility(block_table, batch_size)
    _, max_num_blocks_per_sequence = block_table.shape

    return PagedAttentionMetadata(
        batch_size=batch_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_cache_blocks=num_cache_blocks,
        max_num_blocks_per_sequence=max_num_blocks_per_sequence,
        max_num_splits=_determine_max_num_kv_splits(max_num_blocks_per_sequence * cache_block_size),
    )


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor | None = None,
    scale: float | None = None,
    softcap: float = 0.0,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """PagedAttention interface to verify sizes and launch kernel.

    Args:
        query: Query tensor, shape: (batch_size, num_heads, head_size).
        key_cache: Tensor holding cache for K, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache: Tensor holding cache for V, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        block_table: Tensor storing the mapping from batch to cache blocks, shape: (batch_size, max_num_blocks_per_sequence).
        seq_lens: Tensor with the sequence length of each index in the batch, shape: (batch_size, ).
        output: Tensor to write the output of the attention calculation, shape: (batch_size, num_heads, head_size).
        scale: Scaling factor, 1/sqrt(head_size).
        softcap: Logit softcap to apply (0.0 means no softcap will be applied).
        kv_cache_dtype: Data type of the KV cache.
        k_scale: FP8 scaling factor for key cache.
        v_scale: FP8 scaling factor for value cache.
    """
    # Allocate output tensor if not provided
    if output is None:
        output = torch.zeros_like(query, device=query.device, dtype=query.dtype)

    # Check sizes of input tensors
    metadata = _check_size_compatibility(output, query, key_cache, value_cache, block_table)

    # Note: we could allocate this scratch outside of this function so that we could reuse it. vLLM allocates the scratch memory inline like this

    # Allocate additional memory for intermediate result (of shape (head_size,)) for each batch/query head/cache block
    output_scratchpad = torch.zeros(
        (metadata.batch_size, metadata.max_num_splits, metadata.num_query_heads, metadata.head_size),
        dtype=output.dtype,
        device=output.device,
    )

    # Allocate additional memory for intermediate log-sum-exp ("lse", scalar value per-cache block) for each batch/query head/cache block
    lse_scratchpad = torch.zeros(
        (metadata.batch_size, metadata.max_num_splits, metadata.num_query_heads),
        dtype=output.dtype,
        device=output.device,
    )

    paged_attention_launcher(
        output,
        query,
        key_cache,
        value_cache,
        output_scratchpad,
        lse_scratchpad,
        block_table,
        seq_lens,
        scale,
        softcap,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    return output
