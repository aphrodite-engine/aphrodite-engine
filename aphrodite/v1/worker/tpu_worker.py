"""A TPU worker class."""
import os
from typing import Optional

import torch
import torch.distributed
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.runtime as xr
from loguru import logger

import aphrodite.common.envs as envs
from aphrodite.common.config import AphroditeConfig, ParallelConfig
from aphrodite.common.utils import STR_DTYPE_TO_TORCH_DTYPE
from aphrodite.distributed import (ensure_model_parallel_initialized,
                                   init_distributed_environment)
from aphrodite.modeling import set_random_seed
from aphrodite.v1.core.sched.output import SchedulerOutput
from aphrodite.v1.kv_cache_interface import (AttentionSpec, KVCacheConfig,
                                             KVCacheSpec)
from aphrodite.v1.outputs import ModelRunnerOutput
from aphrodite.v1.utils import bind_kv_cache, report_usage_stats
from aphrodite.v1.worker.tpu_model_runner import TPUModelRunner


class TPUWorker:

    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        self.is_driver_worker = is_driver_worker
        self.aphrodite_config = aphrodite_config
        self.model_config = aphrodite_config.model_config
        self.cache_config = aphrodite_config.cache_config
        self.lora_config = aphrodite_config.lora_config
        self.load_config = aphrodite_config.load_config
        self.parallel_config = aphrodite_config.parallel_config
        self.scheduler_config = aphrodite_config.scheduler_config
        self.device_config = aphrodite_config.device_config
        self.speculative_config = aphrodite_config.speculative_config
        self.prompt_adapter_config = aphrodite_config.prompt_adapter_config
        self.observability_config = aphrodite_config.observability_config

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from aphrodite.common.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Delay profiler initialization to the start of the profiling.
        # This is because in Aphrodite V1, MP runtime is initialized before the
        # TPU Worker is initialized. The profiler server needs to start after
        # MP runtime is initialized.
        self.profiler = None
        self.profile_dir = None
        if envs.APHRODITE_TORCH_PROFILER_DIR and self.rank < 1:
            # For TPU, we can only have 1 active profiler session for 1 profiler
            # server. So we only profile on rank0.
            self.profile_dir = envs.APHRODITE_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: {}",
                        self.profile_dir)

        if self.model_config.seed is None:
            self.model_config.seed = 0

    def init_device(self):
        os.environ["PJRT_DEVICE"] = "TPU"
        # Note: Currently the XLA compiler wrongly uses 2D ring strategy on 1D
        # ring, the xla tpu compiler flag
        # `xla_tpu_force_1d_allreduce_at_chunk_count` is a temporary solution to
        # fix this. It will be removed after the bug in XLA compiler is fixed.
        os.environ["LIBTPU_INIT_ARGS"] = (
            "--xla_tpu_force_1d_allreduce_at_chunk_count=1")
        torch.set_grad_enabled(False)
        torch.set_default_dtype(self.model_config.dtype)

        # Initialize the distributed environment.
        init_tpu_worker_distributed_environment(self.parallel_config,
                                                self.rank,
                                                self.distributed_init_method,
                                                self.local_rank)

        # Device initialization should happen after initializing
        # the distributed runtime.
        self.device = xm.xla_device()
        self.device_config.device = self.device

        # Set random seed.
        set_random_seed(self.model_config.seed)
        if self.model_config.seed is not None:
            xm.set_rng_state(self.model_config.seed, self.device)

        # Increase the cache size limit, which is the maximum number of
        # dynamo graphs that can be compiled.
        # TODO (NickLucche) On gsm we compile 80+ graphs.
        # Re-evaluate limit, with MM we may get close to this limit.
        torch._dynamo.config.cache_size_limit = 128
        # Use persistent cache to avoid XLA recompilation.
        # NOTE: Set per-rank cache path since different ranks
        # can have slightly different XLA graphs.
        world_size = self.parallel_config.world_size
        rank = xr.global_ordinal()
        # The PyTorch/XLA compilation cache uses the Torch IR to generate keys.
        # Consequently, changes in optimization flags, which affect compilation
        # results, don't change the cache key. This can result in the wrong
        # compilation being used. To prevent this, disabling the XLA compilation
        # cache during development is recommended.We can disable it by
        # `export APHRODITE_XLA_CACHE_PATH=`
        if envs.APHRODITE_XLA_CACHE_PATH:
            per_rank_path = os.path.join(envs.APHRODITE_XLA_CACHE_PATH,
                                         f"tp{world_size}_rank{rank}")
            xr.initialize_cache(per_rank_path, readonly=False)

        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner = TPUModelRunner(self.aphrodite_config, self.device)

        if rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.aphrodite_config)

    def determine_available_memory(self) -> int:
        kv_caches: dict[str, torch.Tensor] = {}
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, AttentionSpec):
                dtype = layer_spec.dtype

                # Use an empty tensor instead of `None`` to force Dynamo to pass
                # it by reference, rather by specializing on the value ``None``.
                tpu_kv_cache = torch.tensor([],
                                            dtype=dtype,
                                            device=self.device)
                kv_caches[layer_name] = tpu_kv_cache
            else:
                raise NotImplementedError(
                    f"Unsupported KV cache spec '{type(layer_spec)}'")

        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(
            kv_caches,
            self.aphrodite_config.compilation_config.static_forward_context,
            runner_kv_caches)

        # `max_num_tokens >= max_num_batched_tokens` due to padding.
        self.model_runner.profile_run(self.model_runner.max_num_tokens)

        # Synchronize before measuring the memory usage.
        xm.wait_device_ops()

        # During the profiling run, the model runs without KV cache. After
        # the profiling run, the model always runs with KV cache. Here we clear
        # the dynamo cache and cached bytecode to ensure the model always has
        # one compiled bytecode. Having one FX graph/cached bytecode per
        # compiled model is required for `support_torch_compile` decorator to
        # skip dynamo guard.
        self.model_runner.reset_dynamo_cache()

        # Get the maximum amount of memory used by the model weights and
        # intermediate activations.
        m = xm.get_memory_info(self.device)
        total_memory_size = m["bytes_limit"]
        current_mem = m["bytes_used"]
        # Ideally we would use profiled = m["peak_bytes_used"] to
        # get weights + activations. But there is memory used during
        # compilation / weight loading that impacts the peak and
        # there is no way to reset peak memory in XLA, So we
        # use the heuristic of 2% of weights.
        profiled = current_mem * 1.02

        # Calculate the TPU KV cache size based on profiling.
        usable_memory_size = int(total_memory_size *
                                 self.cache_config.gpu_memory_utilization)
        tpu_kv_cache_bytes = max(usable_memory_size - profiled, 0)

        return int(tpu_kv_cache_bytes)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.is_driver_worker else None

    def profile(self, is_start: bool = True):
        if self.rank < 1:
            if self.profile_dir is None:
                raise RuntimeError("Profiler is not enabled.")
            if is_start:
                if self.profiler is None:
                    self.profiler = xp.start_server(9012)
                xp.start_trace(self.profile_dir)
            else:
                xp.stop_trace()

    def load_model(self) -> None:
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return


def init_tpu_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""

    # NOTE: This is just to initialize the TP group and broadcast
    # the input objects on CPU. The all-reduce and all-gather ops on TPU
    # are invoked by `xm.all_reduce` and `xm.all_gather` which use their
    # own context.
    init_distributed_environment(
        world_size=parallel_config.world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
        backend="gloo",
    )
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)
