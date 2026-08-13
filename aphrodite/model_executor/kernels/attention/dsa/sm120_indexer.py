# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 FP8 sparse-attention indexer score kernels."""

from functools import cache

import torch

from aphrodite.platforms import current_platform
from aphrodite.triton_utils import tl, triton


@cache
def use_sm120_dsa_indexer() -> bool:
    """Whether the native SM120 sparse-indexer kernels should be used."""
    return current_platform.is_cuda() and current_platform.is_device_capability_family(120)


@triton.jit
def _sm120_fp8_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    starts_ptr,
    ends_ptr,
    logits_ptr,
    num_kv_tokens,
    stride_q_m: tl.int64,
    stride_q_h: tl.int64,
    stride_q_d: tl.int64,
    stride_k_n: tl.int64,
    stride_k_d: tl.int64,
    stride_w_m: tl.int64,
    stride_w_h: tl.int64,
    stride_o_m: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    kv = tile * BLOCK_KV + tl.arange(0, BLOCK_KV)
    heads = tl.arange(0, NUM_HEADS)[:, None]
    dims = tl.arange(0, HEAD_DIM)

    start = tl.load(starts_ptr + row)
    end = tl.load(ends_ptr + row)
    valid = (kv >= start) & (kv < end) & (kv < num_kv_tokens)

    q = tl.load(
        q_ptr + row * stride_q_m + heads * stride_q_h + dims[None, :] * stride_q_d,
    )
    k = tl.load(
        k_ptr + dims[:, None] * stride_k_d + kv[None, :] * stride_k_n,
        mask=valid[None, :],
        other=0.0,
    )
    scale = tl.load(k_scale_ptr + kv, mask=valid, other=0.0).to(tl.float32)
    weights = tl.load(weights_ptr + row * stride_w_m + heads * stride_w_h).to(tl.float32)

    scores = tl.dot(q, k, input_precision="ieee").to(tl.float32)
    scores = tl.maximum(scores * scale[None, :], 0.0)
    scores = tl.sum(scores * weights, axis=0)
    scores = tl.where(valid, scores, -float("inf"))
    tl.store(logits_ptr + row * stride_o_m + kv, scores, mask=kv < num_kv_tokens)


@triton.jit
def _sm120_fp8_paged_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    max_model_len,
    stride_q_b: tl.int64,
    stride_q_n: tl.int64,
    stride_q_h: tl.int64,
    stride_q_d: tl.int64,
    stride_k_block: tl.int64,
    stride_k_token: tl.int64,
    stride_k_d: tl.int64,
    stride_s_block: tl.int64,
    stride_s_token: tl.int64,
    stride_ctx_b: tl.int64,
    stride_ctx_n: tl.int64,
    stride_bt_b: tl.int64,
    stride_bt_block: tl.int64,
    stride_w_m: tl.int64,
    stride_w_h: tl.int64,
    stride_o_m: tl.int64,
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    batch = row // NEXT_N
    q_index = row % NEXT_N
    logical = tile * BLOCK_KV + tl.arange(0, BLOCK_KV)
    heads = tl.arange(0, NUM_HEADS)[:, None]
    dims = tl.arange(0, HEAD_DIM)

    context_len = tl.load(context_lens_ptr + batch * stride_ctx_b + q_index * stride_ctx_n)
    valid = (logical < context_len) & (logical < max_model_len)
    logical_block = logical // PAGE_SIZE
    block_offset = logical % PAGE_SIZE
    physical_block = tl.load(
        block_tables_ptr + batch * stride_bt_b + logical_block * stride_bt_block,
        mask=valid,
        other=0,
    )

    q = tl.load(
        q_ptr + batch * stride_q_b + q_index * stride_q_n + heads * stride_q_h + dims[None, :] * stride_q_d,
    )
    k = tl.load(
        k_ptr
        + physical_block[None, :] * stride_k_block
        + block_offset[None, :] * stride_k_token
        + dims[:, None] * stride_k_d,
        mask=valid[None, :],
        other=0.0,
    )
    scale = tl.load(
        k_scale_ptr + physical_block * stride_s_block + block_offset * stride_s_token,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    weights = tl.load(weights_ptr + row * stride_w_m + heads * stride_w_h).to(tl.float32)

    scores = tl.dot(q, k, input_precision="ieee").to(tl.float32)
    scores = tl.maximum(scores * scale[None, :], 0.0)
    scores = tl.sum(scores * weights, axis=0)
    scores = tl.where(valid, scores, -float("inf"))
    tl.store(logits_ptr + row * stride_o_m + logical, scores, mask=logical < max_model_len)


def sm120_fp8_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Compute ragged FP8 indexer logits on SM120."""
    num_rows, num_heads, head_dim = q.shape
    num_kv_tokens = k.shape[0]
    logits = torch.empty((num_rows, num_kv_tokens), dtype=torch.float32, device=q.device)
    block_kv = 64
    _sm120_fp8_mqa_logits_kernel[(num_rows, triton.cdiv(num_kv_tokens, block_kv))](
        q,
        k,
        k_scales.reshape(-1),
        weights,
        starts,
        ends,
        logits,
        num_kv_tokens,
        *q.stride(),
        *k.stride(),
        *weights.stride(),
        logits.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_KV=block_kv,
        num_warps=4,
        num_stages=2,
    )
    return logits


def sm120_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute paged FP8 indexer logits on SM120."""
    batch_size, next_n, num_heads, head_dim = q.shape
    page_size = kv_cache.shape[1]
    num_pages = kv_cache.shape[0]
    page_stride = kv_cache.stride(0)
    cache_bytes = kv_cache.view(torch.uint8)

    # indexer_k_quant_and_cache stores a page in planar form: all
    # ``page_size * head_dim`` FP8 key bytes first, followed by one FP32 scale
    # per token. The logical cache tensor has a per-token trailing width, but
    # slicing that dimension would incorrectly interpret the scale bytes as
    # interleaved with each key row.
    k_bytes = torch.as_strided(
        cache_bytes,
        size=(num_pages, page_size, head_dim),
        stride=(page_stride, head_dim, 1),
    )
    scale_bytes = torch.as_strided(
        cache_bytes,
        size=(num_pages, page_size, 4),
        stride=(page_stride, 4, 1),
        storage_offset=page_size * head_dim,
    )
    k = k_bytes.view(torch.float8_e4m3fn)
    k_scales = scale_bytes.view(torch.float32).squeeze(-1)
    if context_lens.ndim == 1:
        context_lens = context_lens[:, None].expand(-1, next_n)

    logits = torch.empty(
        (batch_size * next_n, max_model_len),
        dtype=torch.float32,
        device=q.device,
    )
    block_kv = 64
    _sm120_fp8_paged_mqa_logits_kernel[(batch_size * next_n, triton.cdiv(max_model_len, block_kv))](
        q,
        k,
        k_scales,
        weights,
        context_lens,
        block_tables,
        logits,
        max_model_len,
        *q.stride(),
        *k.stride(),
        *k_scales.stride(),
        *context_lens.stride(),
        *block_tables.stride(),
        *weights.stride(),
        logits.stride(0),
        NEXT_N=next_n,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_KV=block_kv,
        num_warps=4,
        num_stages=2,
    )
    return logits
