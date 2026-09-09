# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from aphrodite.config import AphroditeConfig
from aphrodite.omni.diffusion.aphrodite_config import create_diffusion_aphrodite_config
from aphrodite.omni.diffusion.data import OmniDiffusionConfig
from aphrodite.omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from aphrodite.omni.platforms import current_omni_platform
from aphrodite.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    resolve_kv_cache_block_sizes,
)
from aphrodite.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

if TYPE_CHECKING:
    from aphrodite.omni.diffusion.executor.abstract import DiffusionExecutor
    from aphrodite.omni.diffusion.request import OmniDiffusionRequest


def build_native_kv_cache_configs(
    aphrodite_config: AphroditeConfig,
    worker_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> tuple[list[KVCacheConfig], KVCacheConfig]:
    """Build rank-local and Scheduler-native configs using Sonar utilities."""

    from aphrodite.v1.core import single_type_kv_cache_manager

    single_type_kv_cache_manager.register_all_kvcache_specs(aphrodite_config)
    worker_configs = get_kv_cache_configs(aphrodite_config, worker_specs, available_memory)
    return worker_configs, generate_scheduler_kv_cache_config(worker_configs)


def initialize_diffusion_kv_control_plane(
    executor: DiffusionExecutor,
    od_config: OmniDiffusionConfig,
    *,
    profile_requests: list[OmniDiffusionRequest] | None = None,
    device: torch.device | None = None,
) -> tuple[KVCacheConfig, int, int, AphroditeConfig] | None:
    """Run the native Worker-spec to Scheduler-config initialization chain."""

    if (
        getattr(od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
        is not DiffusionKVCacheMode.PAGED_SCHEDULER
    ):
        return None
    if not profile_requests:
        raise ValueError("paged_scheduler Diffusion KV initialization requires prepared memory profile requests")
    if device is None:
        device = current_omni_platform.get_torch_device(0)
    aphrodite_config = create_diffusion_aphrodite_config(device, od_config)
    worker_specs = executor.get_kv_cache_specs()
    available_memory = executor.determine_available_kv_memory(profile_requests)
    if len(worker_specs) != od_config.num_gpus or len(available_memory) != od_config.num_gpus:
        raise ValueError(
            "Diffusion KV initialization rank count mismatch: "
            f"expected={od_config.num_gpus}, specs={len(worker_specs)}, memory={len(available_memory)}"
        )
    worker_configs, scheduler_kv_cache_config = build_native_kv_cache_configs(
        aphrodite_config,
        worker_specs,
        available_memory,
    )
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        scheduler_kv_cache_config,
        aphrodite_config,
    )
    # Native sizing may auto-fit ``max_model_len=-1`` against the profiled
    # memory budget.  Forward that final value so every Worker sizes its
    # physical BlockTables from the same bound as the Scheduler.
    resolved_max_model_len = aphrodite_config.model_config.max_model_len
    if type(resolved_max_model_len) is not int or resolved_max_model_len <= 0:
        raise ValueError("native Diffusion KV sizing must resolve a positive max_model_len")
    executor.set_kv_cache_configs(worker_configs, resolved_max_model_len)
    return (
        scheduler_kv_cache_config,
        scheduler_block_size,
        hash_block_size,
        aphrodite_config,
    )
