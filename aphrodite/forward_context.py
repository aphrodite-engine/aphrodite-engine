
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from loguru import logger

import aphrodite.common.envs as envs
from aphrodite.common.config import AphroditeConfig
from aphrodite.distributed.kv_transfer import (get_kv_transfer_group,
                                               has_kv_transfer_group,
                                               is_v1_kv_transfer_group)
from aphrodite.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1)

if TYPE_CHECKING:
    from aphrodite.attention.backends.abstract import AttentionMetadata


track_batchsize: bool = envs.APHRODITE_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.APHRODITE_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)


@dataclass
class DPMetadata:
    cu_tokens_across_dp_cpu: torch.Tensor


@dataclass
class ForwardContext:
    # copy from aphrodite_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    # TODO: extend to support per-layer dynamic forward context
    attn_metadata: "AttentionMetadata"  # set dynamically for each forward pass
    # TODO: remove after making all virtual_engines share the same kv cache
    virtual_engine: int  # set dynamically for each forward pass
    # set dynamically for each forward pass
    dp_metadata: Optional[DPMetadata] = None


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context.")
    return _forward_context


@contextmanager
def set_forward_context(attn_metadata: Any,
                        aphrodite_config: AphroditeConfig,
                        virtual_engine: int = 0,
                        num_tokens: int = 0):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()
    dp_metadata: Optional[DPMetadata] = None
    if aphrodite_config.parallel_config.data_parallel_size > 1:
        dp_size = aphrodite_config.parallel_config.data_parallel_size
        dp_rank = aphrodite_config.parallel_config.data_parallel_rank
        if attn_metadata is not None and hasattr(attn_metadata,
                                                 "num_prefill_tokens"):
            # for v0 attention backends
            batchsize = attn_metadata.num_prefill_tokens + \
                attn_metadata.num_decode_tokens
        else:
            # for v1 attention backends or no attn_metadata
            batchsize = num_tokens
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = batchsize
        num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                         device="cpu",
                                         dtype=torch.int32)
        from aphrodite.distributed.parallel_state import get_dp_group
        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_tensor, dim=0)
        dp_metadata = DPMetadata(cu_tokens_across_dp_cpu)

    global _forward_context
    prev_context = _forward_context
    _forward_context = ForwardContext(
        no_compile_layers=aphrodite_config.compilation_config.
        static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata)

    # KVConnector: trigger (possibly async) load before forward.
    # Each attn layer will block until the reading is complete.
    trigger_kv_transfer = (attn_metadata is not None
                           and has_kv_transfer_group()
                           and is_v1_kv_transfer_group())
    if trigger_kv_transfer:
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase_V1)
        kv_connector.start_load_kv(_forward_context)

    try:
        yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends
                batchsize = num_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            torch.cuda.synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append(
                (now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(("Batchsize forward time stats "
                                 "(batchsize, count, median_time(ms)): {}"),
                                forward_stats)

        # KVConnector: each attn layer triggers (possibly async) save.
        # Ensure all those operations complete before forward() is done.
        if trigger_kv_transfer:
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase_V1)
            kv_connector.wait_for_save()

        _forward_context = prev_context
