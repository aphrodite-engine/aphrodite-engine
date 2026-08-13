# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for FlashInfer sparse MLA workspace sizing."""

from aphrodite.v1.attention.backends.mla.flashinfer_mla_sparse import (
    _DEFAULT_WORKSPACE_BUFFER_SIZE,
    compute_trtllm_sparse_mla_workspace_bytes,
)


def test_dcp_workspace_covers_softmax_stats() -> None:
    size = compute_trtllm_sparse_mla_workspace_bytes(
        base_workspace_bytes=_DEFAULT_WORKSPACE_BUFFER_SIZE,
        dcp_world_size=8,
        num_heads_per_rank=8,
        max_num_batched_tokens=16384,
    )

    expected_softmax_bytes = 8 * (8 * 8) * 16384 * 256 + 1024 * 1024
    assert size == _DEFAULT_WORKSPACE_BUFFER_SIZE + expected_softmax_bytes


def test_non_dcp_workspace_keeps_default_size() -> None:
    size = compute_trtllm_sparse_mla_workspace_bytes(
        base_workspace_bytes=_DEFAULT_WORKSPACE_BUFFER_SIZE,
        dcp_world_size=1,
        num_heads_per_rank=128,
        max_num_batched_tokens=65536,
    )

    assert size == _DEFAULT_WORKSPACE_BUFFER_SIZE
