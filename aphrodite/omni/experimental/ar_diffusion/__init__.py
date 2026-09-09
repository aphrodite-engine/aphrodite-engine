# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AR-Diffusion Engine (AR-Diffusion).

The AR-Diffusion engine: a ``DiffusionEngine`` subclass that adds engine-level KV
cache management for autoregressive / chunked diffusion models. Selected via
``OmniDiffusionConfig.engine_backend = "ar_diffusion"``.
"""

from aphrodite.omni.experimental.ar_diffusion.capability import (
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
    SupportsARDiffusionPipeline,
    SupportsARDiffusionWarmup,
)
from aphrodite.omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer
from aphrodite.omni.experimental.ar_diffusion.engine import ARDiffusionEngine
from aphrodite.omni.experimental.ar_diffusion.session import (
    ARDiffusionSession,
    ARDiffusionSessionManager,
    ARDiffusionWorkerLifecycle,
)
from aphrodite.omni.experimental.ar_diffusion.tick_protocol import (
    AR_DIFFUSION_TICK_KEY,
    ARDiffusionChunkMetadata,
    ARDiffusionControlInput,
    ARDiffusionTickRequest,
)

__all__ = [
    "ARDiffusionKVBranchSpec",
    "ARDiffusionCrossAttentionKVSpec",
    "ARDiffusionEngine",
    "ARDiffusionOmniTickConsumer",
    "ARDiffusionSession",
    "ARDiffusionSessionManager",
    "ARDiffusionWorkerLifecycle",
    "AR_DIFFUSION_TICK_KEY",
    "ARDiffusionChunkMetadata",
    "ARDiffusionControlInput",
    "ARDiffusionTickRequest",
    "ARDiffusionKVCacheSpec",
    "SupportsARDiffusionPipeline",
    "SupportsARDiffusionWarmup",
]
