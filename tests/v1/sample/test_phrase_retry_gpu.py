# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from aphrodite.v1.phrase_guard.v2 import RetryMask


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_retry_mask_only_changes_checkpoint_rows_and_clears_reused_slots():
    mask = RetryMask(3, torch.device("cuda"))
    mask.add_request(2, 12, (3, [4, 7]))
    mask.add_request(0, 10, None)
    mask.apply_staged_writes()
    mapping = torch.tensor([2, 2, 0, 2], device="cuda")
    positions = torch.tensor([14, 15, 14, 16], device="cuda")
    logits = torch.zeros((4, 16), device="cuda")
    mask.apply(logits, mapping, np.array([2, 0]), positions)
    expected = torch.zeros_like(logits)
    expected[0, [4, 7]] = -torch.inf
    torch.testing.assert_close(logits, expected)
    mask.add_request(2, 12, None)
    mask.apply_staged_writes()
    logits.zero_()
    mask.apply(logits, mapping, np.array([2, 0]), positions)
    assert not logits.count_nonzero().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("vocab_size", [16, 151936, 256128])
def test_exhausted_retry_returns_sentinel_without_changing_other_rows(vocab_size):
    mask = RetryMask(2, torch.device("cuda"))
    mask.add_request(0, 12, (0, [4, 7]))
    mask.add_request(1, 12, (0, [4, 7]))
    mask.apply_staged_writes()
    mapping = torch.tensor([0, 1], device="cuda")
    positions = torch.tensor([11, 11], device="cuda")
    logits = torch.full((2, vocab_size), -torch.inf, device="cuda")
    logits[0, 7] = 2.0
    logits[1, -1] = 3.0
    expected = logits.clone()
    expected[0, 7] = -torch.inf
    expected[0, 4] = 0.0
    mask.apply(logits, mapping, np.array([0, 1]), positions)
    torch.testing.assert_close(logits, expected)
