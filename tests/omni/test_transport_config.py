# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys


def test_transport_config_fields_and_deferred_defaults():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import copy
from dataclasses import fields
from aphrodite.config import CacheConfig, SchedulerConfig, ParallelConfig
from aphrodite.omni.config.omni_config import (
    OmniStageCacheConfig, OmniStageSchedulerConfig, OmniStageParallelConfig,
    OmniStageDiffusionParallelConfig, _CACHE_CONFIG_ENGINE_FIELD_MAP,
    _SCHEDULER_CONFIG_ENGINE_FIELD_MAP, _PARALLEL_CONFIG_ENGINE_FIELD_MAP,
)
from aphrodite.omni.engine.stage_init_utils import _project_upstream_config_fields

for transport, core, field_map in (
    (OmniStageCacheConfig, CacheConfig, _CACHE_CONFIG_ENGINE_FIELD_MAP),
    (OmniStageSchedulerConfig, SchedulerConfig, _SCHEDULER_CONFIG_ENGINE_FIELD_MAP),
    (OmniStageParallelConfig, ParallelConfig, _PARALLEL_CONFIG_ENGINE_FIELD_MAP),
):
    assert {f.name for f in fields(core)} <= {f.name for f in fields(transport)}
    assert _project_upstream_config_fields(transport(), field_map) == {}

cache = OmniStageCacheConfig(gpu_memory_utilization=0.6, enable_prefix_caching=False)
assert _project_upstream_config_fields(copy.deepcopy(cache), _CACHE_CONFIG_ENGINE_FIELD_MAP) == {
    'gpu_memory_utilization': 0.6, 'enable_prefix_caching': False,
}
assert OmniStageCacheConfig().gpu_memory_utilization is None
assert OmniStageCacheConfig().mamba_ssm_cache_dtype is None
assert OmniStageCacheConfig(mamba_block_size=128).user_specified_mamba_block_size
scheduler = OmniStageSchedulerConfig()
assert scheduler.max_num_seqs is None and scheduler.max_num_batched_tokens is None
assert scheduler.encoder_cache_size is None
assert OmniStageSchedulerConfig(max_num_batched_tokens=4096).encoder_cache_size == 4096
parallel = OmniStageParallelConfig(data_parallel_size=3, data_parallel_rank=2)
assert parallel.data_parallel_rank == 2 and parallel.data_parallel_size_local is None
assert parallel.worker_cls is None and parallel.data_parallel_rpc_port is None
assert _project_upstream_config_fields(parallel, _PARALLEL_CONFIG_ENGINE_FIELD_MAP) == {
    'data_parallel_size': 3, 'data_parallel_rank': 2,
}
assert OmniStageDiffusionParallelConfig(data_parallel_size=3, data_parallel_rank=2).data_parallel_index == 2
for cls, kwargs in (
    (OmniStageCacheConfig, {'gpu_memory_utilization': 2.0}),
    (OmniStageSchedulerConfig, {'max_num_seqs': 4, 'max_num_batched_tokens': 2}),
    (OmniStageParallelConfig, {'tensor_parallel_size': 3, 'decode_context_parallel_size': 2}),
):
    try:
        cls(**kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError((cls, kwargs))
""",
        ],
        check=True,
    )
