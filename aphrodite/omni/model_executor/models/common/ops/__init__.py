# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common operators shared across models and hardware backends.

This module contains fused operators that can be reused across multiple models
and compiled for different hardware targets (CUDA, ROCm, etc.).
"""

from aphrodite.omni.model_executor.models.common.ops.fused_adaptive_group_norm_silu import (
    fused_adaptive_group_norm_silu,
)
from aphrodite.omni.model_executor.models.common.ops.fused_group_norm_silu import (
    fused_group_norm_silu,
)

__all__ = [
    "fused_group_norm_silu",
    "fused_adaptive_group_norm_silu",
]
