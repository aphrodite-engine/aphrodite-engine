# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Compatibility shim.

The MammothModa2 DiT implementation lives under `aphrodite.omni.diffusion` to align
with other ARDiT structured models. We keep this module path so existing
OmniModelRegistry entries (and downstream code) keep working.
"""

from aphrodite.omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (  # noqa: F401
    MammothModa2DiTPipeline,
)

__all__ = [
    "MammothModa2DiTPipeline",
]
