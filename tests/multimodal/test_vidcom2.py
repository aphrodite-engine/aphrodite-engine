# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project

import pytest
import torch

from aphrodite.multimodal.vidcom2 import (
    compute_retained_tokens_count,
    compute_retention_mask,
)


def _fake_video_embeds(num_frames: int, rows: int, cols: int, hidden: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(num_frames * rows * cols, hidden, generator=generator)


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("num_frames", [1, 4, 16])
def test_retention_mask(q: float, num_frames: int) -> None:
    merge = 2
    rows, cols = 6, 8
    mask = compute_retention_mask(
        _fake_video_embeds(num_frames, rows, cols),
        (num_frames, rows * merge, cols * merge),
        spatial_merge_size=merge,
        q=q,
    )
    expected = compute_retained_tokens_count(rows * cols, num_frames, q)
    assert mask.dtype == torch.bool
    assert mask.shape == (num_frames * rows * cols,)
    assert int(mask.sum()) == expected


def test_retained_count_floors_at_one_token_per_frame() -> None:
    assert compute_retained_tokens_count(48, 4, 0.999) == 4
    assert compute_retained_tokens_count(48, 4, 0.0) == 192


def test_empty_input_safe() -> None:
    embeds = torch.zeros(0, 32)
    mask = compute_retention_mask(embeds, (0, 0, 0), 2, 0.25)
    assert mask.numel() == 0
