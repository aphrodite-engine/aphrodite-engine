# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the native SM120 sparse-indexer score kernels."""

import pytest
import torch

from aphrodite.model_executor.kernels.attention.dsa.sm120_indexer import (
    sm120_fp8_mqa_logits,
    sm120_fp8_paged_mqa_logits,
)
from aphrodite.platforms import current_platform

PAGE_SIZE = 64
HEAD_DIM = 128
NUM_HEADS = 64


def _sm120_available() -> bool:
    if not current_platform.is_cuda():
        return False
    try:
        return current_platform.is_device_capability_family(120)
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 GPU")


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    scores = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    return (scores.mul(scales[None, None, :]).relu() * weights[:, :, None]).sum(1)


def test_sm120_fp8_mqa_logits() -> None:
    torch.manual_seed(0)
    num_rows, num_tokens = 5, 193
    q = torch.randn((num_rows, NUM_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    k = torch.randn((num_tokens, HEAD_DIM), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scales = torch.rand(num_tokens, device="cuda") * 0.02
    weights = torch.randn((num_rows, NUM_HEADS), device="cuda")
    starts = torch.tensor([0, 3, 61, 100, 192], dtype=torch.int32, device="cuda")
    ends = torch.tensor([193, 129, 162, 193, 193], dtype=torch.int32, device="cuda")

    actual = sm120_fp8_mqa_logits(q, k, scales, weights, starts, ends)
    expected = _reference(q, k, scales, weights)
    positions = torch.arange(num_tokens, device="cuda")[None, :]
    valid = (positions >= starts[:, None]) & (positions < ends[:, None])
    expected.masked_fill_(~valid, -torch.inf)

    torch.testing.assert_close(actual[valid], expected[valid], rtol=2e-2, atol=2e-2)
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))


@pytest.mark.parametrize("next_n", [1, 2, 7])
def test_sm120_fp8_paged_mqa_logits(next_n: int) -> None:
    torch.manual_seed(1)
    batch_size, num_pages = 2, 5
    max_model_len = 192
    q = torch.randn(
        (batch_size, next_n, NUM_HEADS, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    keys = torch.randn((num_pages, PAGE_SIZE, HEAD_DIM), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scales = torch.rand((num_pages, PAGE_SIZE), device="cuda") * 0.02
    cache = torch.empty(
        (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
        dtype=torch.uint8,
        device="cuda",
    )
    # Match indexer_k_quant_and_cache's planar page layout: all key rows,
    # followed by all per-token scales.
    page_bytes = cache.view(num_pages, -1)
    page_bytes[:, : PAGE_SIZE * HEAD_DIM].copy_(keys.view(torch.uint8).reshape(num_pages, -1))
    page_bytes[:, PAGE_SIZE * HEAD_DIM :].copy_(scales.view(torch.uint8).reshape(num_pages, -1))
    weights = torch.randn((batch_size * next_n, NUM_HEADS), device="cuda")
    context_lens = torch.tensor(
        [[129 + token for token in range(next_n)], [70 + token for token in range(next_n)]],
        dtype=torch.int32,
        device="cuda",
    )
    block_tables = torch.tensor([[2, 0, 4], [3, 1, 0]], dtype=torch.int32, device="cuda")

    actual = sm120_fp8_paged_mqa_logits(q, cache, weights, context_lens, block_tables, max_model_len)
    expected = torch.full_like(actual, -torch.inf)
    for batch in range(batch_size):
        pages = block_tables[batch].long()
        logical_k = keys[pages].reshape(-1, HEAD_DIM)
        logical_scales = scales[pages].reshape(-1)
        for token in range(next_n):
            row = batch * next_n + token
            length = int(context_lens[batch, token])
            expected[row, :length] = _reference(
                q[batch, token].unsqueeze(0),
                logical_k[:length],
                logical_scales[:length],
                weights[row].unsqueeze(0),
            )[0]

    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], rtol=2e-2, atol=2e-2)
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
