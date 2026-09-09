# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Public API for the Cache-DiT diffusion cache backend."""

from aphrodite.omni.diffusion.cache.cachedit.backend import (
    CUSTOM_DIT_ENABLERS,
    CacheDiTBackend,
    CacheDiTEnableResult,
    cache_summary,
    enable_cache_for_dit,
)
from aphrodite.omni.diffusion.cache.cachedit.config import (
    CacheDiTAdapterConfig,
    CacheDiTConfig,
)
from aphrodite.omni.diffusion.cache.cachedit.model_specific import (
    BagelCachedAdapter,
    SensenovaCachedAdapter,
)
from aphrodite.omni.diffusion.cache.cachedit.model_specific import (
    register_custom_dit_enablers as _register_custom_dit_enablers,
)
from aphrodite.omni.diffusion.cache.cachedit.runtime import (
    CacheDiTRequestSpec,
    RequestScopedCacheDiTRuntime,
)

_register_custom_dit_enablers()

__all__ = [
    "BagelCachedAdapter",
    "CUSTOM_DIT_ENABLERS",
    "CacheDiTAdapterConfig",
    "CacheDiTBackend",
    "CacheDiTEnableResult",
    "CacheDiTConfig",
    "CacheDiTRequestSpec",
    "RequestScopedCacheDiTRuntime",
    "SensenovaCachedAdapter",
    "cache_summary",
    "enable_cache_for_dit",
]
