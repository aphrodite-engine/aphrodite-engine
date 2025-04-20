# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional

import torch


def _create_causal_mask(
    position_ids,
    target_length,
    sliding_window: Optional[int] = None,
):
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
    """
    if sliding_window is not None:
        query_indices = position_ids.unsqueeze(-1)
        kv_indices = torch.arange(target_length).view(1, -1)
        # --- Rolling buffer ---
        pos_max = position_ids.max(1, keepdim=True).values
        kv_start = (pos_max // target_length) * target_length
        kv_indices_high = kv_indices + kv_start
        kv_indices_low = torch.where(
            kv_indices_high < target_length,
            kv_indices,
            kv_indices_high - target_length,
        )
        kv_indices = torch.where(
            kv_indices_high > pos_max, kv_indices_low, kv_indices_high
        )
        kv_indices = kv_indices.unsqueeze(1)
        # ------
        causal_mask = kv_indices > query_indices
        attention_mask = causal_mask

        window_indices = query_indices - sliding_window + 1
        window_mask = kv_indices < window_indices
        attention_mask = attention_mask | window_mask
        attention_mask = attention_mask.unsqueeze(1)
    else:
        query_indices = position_ids.unsqueeze(-1)
        kv_indices = torch.arange(target_length).view(1, 1, -1)
        attention_mask = kv_indices > query_indices
        attention_mask = attention_mask.unsqueeze(1)

    return attention_mask
