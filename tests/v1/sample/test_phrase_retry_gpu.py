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
