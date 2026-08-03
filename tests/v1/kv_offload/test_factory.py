# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for OffloadingSpecFactory.

These tests verify:
1. Pre-registration integrity — registered module paths can actually import
   and yield correct OffloadingSpec subclasses (CI sentinel against file moves).
2. End-to-end factory → spec construction with real configs.
3. Downstream collaboration — build_metric_definitions delegation.
4. Error paths — unregistered specs, missing config, duplicate registration.
"""

from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from aphrodite.config import KVTransferConfig
from aphrodite.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from aphrodite.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from aphrodite.v1.kv_offload.base import OffloadingHistogramMetadata, OffloadingSpec
from aphrodite.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from aphrodite.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from aphrodite.v1.kv_offload.factory import OffloadingSpecFactory
from aphrodite.v1.kv_offload.tiering.spec import TieringOffloadingSpec

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_registry():
    """Save and restore OffloadingSpecFactory._registry between tests."""
    original = dict(OffloadingSpecFactory._registry)
    yield
    OffloadingSpecFactory._registry = original


def _make_aphrodite_config(
    spec_name: str | None = "CPUOffloadingSpec",
    cpu_bytes_to_use: int | None = None,
    store_threshold: int = 0,
    extra_config: dict | None = None,
):
    """Build a real AphroditeConfig with kv_transfer_config set for offloading."""
    from aphrodite.config import (
        AphroditeConfig,
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        SchedulerConfig,
    )

    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=10000,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )

    cfg = extra_config or {}
    if cpu_bytes_to_use is not None:
        cfg["cpu_bytes_to_use"] = cpu_bytes_to_use
    cfg["spec_name"] = spec_name
    if store_threshold > 0:
        cfg["store_threshold"] = store_threshold

    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=cfg,
    )
    return AphroditeConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )


def _make_kv_cache_config():
    """Build a minimal KVCacheConfig with one KV cache tensor."""
    num_blocks = 16
    num_kv_heads = 1
    head_size = 1
    dtype = torch.float32
    page_size = 2 * num_kv_heads * head_size * torch.finfo(dtype).bits // 8
    kv_tensor = KVCacheTensor(size=num_blocks * page_size, shared_by=["layer"], block_stride=0)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[kv_tensor],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                ),
            )
        ],
    )


def _make_hybrid_kv_cache_config():
    """Build a minimal KVCacheConfig with heterogeneous KV cache groups."""
    num_blocks = 16
    num_kv_heads = 1
    head_size = 1
    dtype = torch.float32
    page_size = 2 * num_kv_heads * head_size * torch.finfo(dtype).bits // 8
    kv_tensor = KVCacheTensor(
        size=num_blocks * page_size * 2,
        shared_by=["attention_layer", "sliding_layer"],
        block_stride=0,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[kv_tensor],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention_layer"],
                FullAttentionSpec(
                    block_size=12,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                ),
            ),
            KVCacheGroupSpec(
                ["sliding_layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                ),
            ),
        ],
    )


def _make_mamba_hybrid_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=16,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Pre-registration integrity (CI sentinel)
# ---------------------------------------------------------------------------


def test_pre_registered_specs_can_be_imported():
    """If someone moves cpu/spec.py but forgets to update factory.py, CI fails."""
    for name in OffloadingSpecFactory._registry:
        cls = OffloadingSpecFactory._registry[name]()
        assert issubclass(cls, OffloadingSpec)


def test_cpu_spec_registered():
    """CPUOffloadingSpec is registered and importable."""
    cls = OffloadingSpecFactory._registry["CPUOffloadingSpec"]()
    assert cls is CPUOffloadingSpec


def test_tiering_spec_registered():
    """TieringOffloadingSpec is registered and importable."""
    cls = OffloadingSpecFactory._registry["TieringOffloadingSpec"]()
    assert cls is TieringOffloadingSpec


# ---------------------------------------------------------------------------
# Normal path — get_spec_cls
# ---------------------------------------------------------------------------


def test_get_spec_cls_returns_registered_class():
    """Registered spec_name returns correct class."""
    config = _make_aphrodite_config(spec_name="CPUOffloadingSpec")
    spec_cls = OffloadingSpecFactory.get_spec_cls(config.kv_transfer_config.kv_connector_extra_config)
    assert spec_cls is CPUOffloadingSpec


def test_get_spec_cls_default_to_cpu():
    """Default spec_name (absent from config) resolves to CPUOffloadingSpec."""
    config = _make_aphrodite_config(spec_name=None)
    config.kv_transfer_config.kv_connector_extra_config.pop("spec_name", None)
    spec_cls = OffloadingSpecFactory.get_spec_cls(config.kv_transfer_config.kv_connector_extra_config)
    assert spec_cls is CPUOffloadingSpec


# ---------------------------------------------------------------------------
# End-to-end — create_spec
# ---------------------------------------------------------------------------


def test_create_cpu_offloading_spec_end_to_end():
    """Full factory → spec construction with real AphroditeConfig/KVCacheConfig.

    Verifies:
    - cpu_bytes_to_use validation and num_blocks calculation
    - block_size % tokens_per_hash assertion
    - spec instance is CPUOffloadingSpec
    """
    config = _make_aphrodite_config(cpu_bytes_to_use=65536)
    kv_cache_config = _make_kv_cache_config()
    spec = OffloadingSpecFactory.create_spec(build_offloading_config(config, kv_cache_config))
    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.num_blocks > 0


def test_cpu_spec_sizes_use_shared_region_alignment():
    alignment = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    config = _make_aphrodite_config(cpu_bytes_to_use=alignment * 3)
    spec = OffloadingSpecFactory.create_spec(build_offloading_config(config, _make_kv_cache_config()))

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.cpu_page_size_per_worker == 8
    assert spec.kv_bytes_per_chunk == alignment
    assert spec.num_blocks == 3


def _create_cpu_spec(
    *,
    cpu_bytes_to_use: int,
    worker_kv_bytes_per_block: int,
    world_size: int,
    replicated_layout: bool,
) -> CPUOffloadingSpec:
    config = _make_aphrodite_config(cpu_bytes_to_use=cpu_bytes_to_use)
    offloading_config = build_offloading_config(config, _make_kv_cache_config())
    offloading_config = replace(
        offloading_config,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        parallel=replace(offloading_config.parallel, world_size=world_size),
        replicated_layout=replicated_layout,
    )
    return CPUOffloadingSpec(offloading_config)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_cpu_spec_replicated_sizing_on_shared_region(monkeypatch, world_size: int):
    import aphrodite.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    worker_bytes = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_cpu_spec(
        cpu_bytes_to_use=worker_bytes * 8,
        worker_kv_bytes_per_block=worker_bytes,
        world_size=world_size,
        replicated_layout=True,
    )

    assert spec.replicated_layout is True
    assert spec.cpu_page_size_per_worker == worker_bytes
    assert spec.kv_bytes_per_chunk == worker_bytes
    assert spec.num_blocks == 8


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_cpu_spec_replicated_disabled_without_shared_region(monkeypatch, world_size: int):
    import aphrodite.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: False)
    worker_bytes = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_cpu_spec(
        cpu_bytes_to_use=worker_bytes * world_size * 2,
        worker_kv_bytes_per_block=worker_bytes,
        world_size=world_size,
        replicated_layout=True,
    )

    assert spec.replicated_layout is False
    assert spec.cpu_page_size_per_worker == worker_bytes
    assert spec.kv_bytes_per_chunk == worker_bytes * world_size
    assert spec.num_blocks == 2


@pytest.mark.parametrize("config_replicated", [True, False])
@pytest.mark.parametrize("cuda_alike", [True, False])
def test_cpu_spec_replicated_layout_truth_matrix(monkeypatch, cuda_alike: bool, config_replicated: bool):
    import aphrodite.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: cuda_alike)
    worker_bytes = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_cpu_spec(
        cpu_bytes_to_use=worker_bytes * 8,
        worker_kv_bytes_per_block=worker_bytes,
        world_size=4,
        replicated_layout=config_replicated,
    )

    assert spec.replicated_layout is (cuda_alike and config_replicated)


def test_cpu_spec_create_worker_uses_mmap_on_cuda_alike(monkeypatch):
    import aphrodite.v1.kv_offload.cpu.spec as cpu_spec_module

    alignment = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    config = _make_aphrodite_config(cpu_bytes_to_use=alignment * 8)
    config.parallel_config.tensor_parallel_size = 4
    spec = OffloadingSpecFactory.create_spec(build_offloading_config(config, _make_kv_cache_config()))
    assert isinstance(spec, CPUOffloadingSpec)

    region = MagicMock()
    region_calls: list[dict[str, Any]] = []
    worker_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return region

    def fake_worker_ctor(**kwargs):
        worker_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(cpu_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(cpu_spec_module, "CPUOffloadingWorker", fake_worker_ctor)
    monkeypatch.setattr(cpu_spec_module.torch.accelerator, "current_device_index", lambda: 5)

    kv_caches = MagicMock()
    spec.create_worker(kv_caches)

    assert region_calls[0]["rank"] == 1
    assert region_calls[0]["engine_id"] == spec.config.engine_id
    assert region_calls[0]["kv_bytes_per_block"] == spec.kv_bytes_per_chunk
    assert worker_calls[0]["kv_caches"] is kv_caches
    assert worker_calls[0]["mmap_region"] is region


def test_cpu_spec_create_worker_uses_tensor_path_off_cuda_alike(monkeypatch):
    import aphrodite.v1.kv_offload.cpu.spec as cpu_spec_module

    config = _make_aphrodite_config(cpu_bytes_to_use=65536)
    spec = OffloadingSpecFactory.create_spec(build_offloading_config(config, _make_kv_cache_config()))
    assert isinstance(spec, CPUOffloadingSpec)

    region_calls: list[dict[str, Any]] = []
    worker_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: False)
    monkeypatch.setattr(
        cpu_spec_module,
        "SharedOffloadRegion",
        lambda **kwargs: region_calls.append(kwargs),
    )
    monkeypatch.setattr(
        cpu_spec_module,
        "CPUOffloadingWorker",
        lambda **kwargs: worker_calls.append(kwargs),
    )

    spec.create_worker(MagicMock())

    assert region_calls == []
    assert worker_calls[0]["mmap_region"] is None


def test_cpu_spec_create_worker_skips_mmap_for_empty_cache(monkeypatch):
    import aphrodite.v1.kv_offload.cpu.spec as cpu_spec_module

    config = _make_aphrodite_config(cpu_bytes_to_use=65536)
    spec = OffloadingSpecFactory.create_spec(build_offloading_config(config, _make_kv_cache_config()))
    assert isinstance(spec, CPUOffloadingSpec)
    spec.num_blocks = 0

    region_calls: list[dict[str, Any]] = []
    worker_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(
        cpu_spec_module,
        "SharedOffloadRegion",
        lambda **kwargs: region_calls.append(kwargs),
    )
    monkeypatch.setattr(
        cpu_spec_module,
        "CPUOffloadingWorker",
        lambda **kwargs: worker_calls.append(kwargs),
    )

    spec.create_worker(MagicMock())

    assert region_calls == []
    assert worker_calls[0]["mmap_region"] is None


@pytest.mark.parametrize(
    ("replicated_layout", "device_index", "world_size", "expected_rank"),
    [
        (True, 5, 4, 0),
        (True, 0, 4, 0),
        (False, 5, 4, 1),
        (False, 7, 4, 3),
    ],
)
def test_cpu_spec_create_worker_rank_assignment(
    monkeypatch, replicated_layout, device_index, world_size, expected_rank
):
    import aphrodite.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    worker_bytes = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_cpu_spec(
        cpu_bytes_to_use=worker_bytes * 8,
        worker_kv_bytes_per_block=worker_bytes,
        world_size=world_size,
        replicated_layout=replicated_layout,
    )
    region_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(cpu_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(cpu_spec_module, "CPUOffloadingWorker", MagicMock())
    monkeypatch.setattr(
        cpu_spec_module.torch.accelerator,
        "current_device_index",
        lambda: device_index,
    )

    spec.create_worker(MagicMock())

    assert region_calls[0]["rank"] == expected_rank


# ---------------------------------------------------------------------------
# Dynamic import via spec_module_path
# ---------------------------------------------------------------------------


def test_dynamic_load_via_spec_module_path():
    """External spec loaded via spec_module_path.

    This is how external projects (e.g., llm-d-kv-cache SharedStorageOffloadingSpec)
    integrate with Aphrodite without being pre-registered in the factory.
    The fallback path: registry miss → spec_module_path → importlib.import_module.
    """
    config = _make_aphrodite_config(spec_name="CPUOffloadingSpec")
    # Delete from registry to force the dynamic import path
    del OffloadingSpecFactory._registry["CPUOffloadingSpec"]
    # spec_name not in registry → falls through to spec_module_path
    config.kv_transfer_config.kv_connector_extra_config["spec_module_path"] = "aphrodite.v1.kv_offload.cpu.spec"
    spec_cls = OffloadingSpecFactory.get_spec_cls(config.kv_transfer_config.kv_connector_extra_config)
    assert spec_cls is CPUOffloadingSpec


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unregistered_spec_without_module_path_raises():
    """spec_name not in registry + no spec_module_path → ValueError."""
    config = _make_aphrodite_config(spec_name="NonexistentSpec")
    with pytest.raises(ValueError, match="Unsupported spec type"):
        OffloadingSpecFactory.get_spec_cls(config.kv_transfer_config.kv_connector_extra_config)

    # create_spec should also fail (calls get_spec_cls internally)
    kv_cache_config = _make_kv_cache_config()
    with pytest.raises(ValueError, match="Unsupported spec type"):
        OffloadingSpecFactory.create_spec(build_offloading_config(config, kv_cache_config))


def test_cpu_spec_missing_cpu_bytes_to_use_raises():
    """CPUOffloadingSpec requires cpu_bytes_to_use → Exception."""
    config = _make_aphrodite_config(cpu_bytes_to_use=None)
    config.kv_transfer_config.kv_connector_extra_config.pop("cpu_bytes_to_use", None)
    kv_cache_config = _make_kv_cache_config()
    with pytest.raises(Exception, match="cpu_bytes_to_use must be specified"):
        OffloadingSpecFactory.create_spec(build_offloading_config(config, kv_cache_config))


def test_duplicate_registration_raises():
    """register_spec with existing name → ValueError."""
    with pytest.raises(ValueError, match="is already registered"):
        OffloadingSpecFactory.register_spec("CPUOffloadingSpec", "some.module", "SomeClass")


# ---------------------------------------------------------------------------
# Downstream collaboration — build_metric_definitions
# ---------------------------------------------------------------------------


def test_build_metric_definitions_below_threshold():
    """store_threshold < 2 keeps stores_skipped disabled."""
    from aphrodite.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    config = _make_aphrodite_config(store_threshold=1)
    spec_cls = OffloadingSpecFactory.get_spec_cls(config.kv_transfer_config.kv_connector_extra_config)
    metrics = spec_cls.build_metric_definitions(config.kv_transfer_config.kv_connector_extra_config)
    assert CPUOffloadingMetrics.STORES_SKIPPED not in metrics
    assert CPUOffloadingMetrics.CPU_ALLOCATION_SIZE in metrics


def test_build_metric_definitions_allocation_size_histogram():
    """CPU allocation size is always reported as a histogram."""
    from aphrodite.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    config = _make_aphrodite_config(store_threshold=0)
    spec_cls = OffloadingSpecFactory.get_spec_cls(config.kv_transfer_config.kv_connector_extra_config)
    metrics = spec_cls.build_metric_definitions(config.kv_transfer_config.kv_connector_extra_config)
    metadata = metrics[CPUOffloadingMetrics.CPU_ALLOCATION_SIZE]
    assert isinstance(metadata, OffloadingHistogramMetadata)
    assert metadata.buckets == (
        1,
        4,
        16,
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
    )


def test_build_metric_definitions_returns_counter_at_threshold():
    """store_threshold >= 2 → returns stores_skipped counter definition."""
    from aphrodite.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    config = _make_aphrodite_config(store_threshold=2)
    spec_cls = OffloadingSpecFactory.get_spec_cls(config.kv_transfer_config.kv_connector_extra_config)
    metrics = spec_cls.build_metric_definitions(config.kv_transfer_config.kv_connector_extra_config)
    assert CPUOffloadingMetrics.STORES_SKIPPED in metrics


def test_offloading_spec_accepts_blocks_per_chunk_for_heterogeneous_groups():
    config = _make_aphrodite_config(
        cpu_bytes_to_use=65536,
        extra_config={"blocks_per_chunk": 2},
    )

    spec = OffloadingSpecFactory.create_spec(build_offloading_config(config, _make_hybrid_kv_cache_config()))

    assert spec.tokens_per_block == (12, 16)
    assert spec.blocks_per_chunk == 2


def test_dcp_scales_attention_but_not_mamba_group_blocks():
    config = _make_aphrodite_config(cpu_bytes_to_use=65536)
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.decode_context_parallel_size = 2

    offloading_config = build_offloading_config(config, _make_mamba_hybrid_kv_cache_config())

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (
        32,
        16,
    )


def test_block_size_and_blocks_per_chunk_are_mutually_exclusive():
    config = _make_aphrodite_config(
        cpu_bytes_to_use=65536,
        extra_config={
            "block_size": 64,
            "blocks_per_chunk": 2,
        },
    )

    with pytest.raises(ValueError, match="Specify only one"):
        OffloadingSpecFactory.create_spec(build_offloading_config(config, _make_kv_cache_config()))


def test_blocks_per_chunk_must_be_positive():
    config = _make_aphrodite_config(
        cpu_bytes_to_use=65536,
        extra_config={
            "blocks_per_chunk": 0,
        },
    )

    with pytest.raises(ValueError, match="greater than 0"):
        OffloadingSpecFactory.create_spec(build_offloading_config(config, _make_kv_cache_config()))
