# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from aphrodite.v1.worker.block_table import get_block_table_width


def test_block_table_width_matches_dcp_alignment() -> None:
    # 80K tokens / (64-token blocks * DCP4) needs 313 local blocks. The
    # runtime table aligns this to a 128-token boundary, hence 314 entries.
    assert get_block_table_width(313, 64) == 314


def test_block_table_width_accounts_for_kernel_block_splitting() -> None:
    assert get_block_table_width(7, 32, 16) == 16


def test_block_table_width_rejects_incompatible_kernel_size() -> None:
    with pytest.raises(ValueError, match="must divide"):
        get_block_table_width(8, 24, 16)
