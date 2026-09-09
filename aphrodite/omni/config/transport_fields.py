# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import field
from typing import ClassVar

from pydantic import Field

from aphrodite.config import cache as cache_types
from aphrodite.config import parallel as parallel_types
from aphrodite.config import scheduler as scheduler_types
from aphrodite.config.utils import config


@config(kw_only=True)
class CacheConfigTransportFields:
    DEFAULT_BLOCK_SIZE: ClassVar[int] = 16
    block_size: int | None = Field(default=None, gt=0)
    """Size of a contiguous cache block in number of tokens."""
    user_specified_block_size: bool = field(default=False, init=False)
    """Whether block_size was explicitly provided."""
    user_specified_mamba_block_size: bool = field(default=False, init=False)
    """Whether mamba_block_size was explicitly provided."""
    kv_cache_layout: str | None = field(default=None, init=False)
    """Resolved physical KV cache layout name (a ``KVCacheLayout`` member)."""
    prefix_match_unit: int | None = Field(default=None, gt=0)
    """The finest token boundary (in tokens) a prefix-cache hit can land on."""
    cache_dtype: cache_types.CacheDType = "auto"
    """Data type for kv cache storage."""
    is_attention_free: bool = False
    """Whether the model is attention-free."""
    num_gpu_blocks_override: int | None = None
    """Number of GPU blocks to use."""
    sliding_window: int | None = None
    """Sliding window size for the KV cache."""
    prefix_caching_hash_algo: cache_types.PrefixCachingHashAlgo = "sha256"
    """Set the hash algorithm for prefix caching: - "sha256" uses Pickle for object serialization before hashing."""
    prefix_cache_retention_interval: int | None = Field(
        default_factory=cache_types._get_prefix_cache_retention_interval, ge=0
    )
    """Token interval between retained sliding-window and Mamba prefix-cache checkpoints."""
    kv_cache_dtype_skip_layers: list[str] = field(default_factory=list)
    """Layer patterns to skip KV cache quantization."""
    mamba_page_size_padded: int | None = None
    """Mamba page size override used to align state pages with attention pages."""
    skip_page_size_padded: int | None = None
    """Page size override that aligns unquantized skip layers with quantized KV cache pages."""
    mamba_block_size: int | None = Field(default=None, gt=0)
    """Size of a contiguous cache block in number of tokens for mamba cache."""
    mamba_cache_dtype: cache_types.MambaDType = "auto"
    """The data type to use for the Mamba cache (both the conv as well as the ssm state)."""
    mamba_cache_mode: cache_types.MambaCacheMode = "none"
    """The cache strategy for Mamba layers."""
    replayssm_buffer_len: int = Field(default=16, gt=0)
    """ReplaySSM logical history length B for Mamba2."""
    use_replayssm: bool = False
    """Use the ReplaySSM Mamba2 decode kernel: cache recent SSM inputs and skip the per-step full-state store, writing the checkpoint back only on flush."""
    use_kda_recoverssm: bool = field(default=False, init=False)
    """Whether Kimi-K3 KDA uses RecoverSSM speculative decode."""
    num_gpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for GPU memory."""
    num_cpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for CPU memory."""
    kv_cache_size_tokens: int | None = field(default=None, init=False)
    """Per-DP-engine KV cache capacity in tokens (group-aware)."""
    kv_cache_max_concurrency: float | None = field(default=None, init=False)
    """Per-DP-engine maximum concurrency at max_model_len tokens."""
    kv_sharing_fast_prefill: bool = False
    """In some KV sharing setups, e.g."""
    kv_offloading_size: float | None = None
    """Size of the KV cache offloading buffer in GiB."""
    kv_offloading_backend: cache_types.KVOffloadingBackend = "native"
    """The backend to use for KV cache offloading."""
    _block_size_resolved: bool = field(default=False, init=False)
    """Guard against pydantic re-running _apply_block_size_default."""


@config(kw_only=True)
class SchedulerConfigTransportFields:
    DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048
    DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP: ClassVar[int] = 256
    DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128
    runner_type: scheduler_types.RunnerType = "generate"
    """The runner type to launch for the model."""
    max_num_scheduled_tokens: int | None = Field(default=None, ge=0)
    """Maximum number of tokens that the scheduler may issue in a single iteration."""
    long_prefill_token_threshold: int = Field(default=0, ge=0)
    """For chunked prefill, a request is considered long if the prompt is longer than this number of tokens."""
    max_num_queued_reqs: int | None = Field(default=None, ge=0)
    """Maximum number of requests that can be in-flight (waiting or running) at the same time, or None for no limit."""
    max_num_queued_tokens: int | None = Field(default=None, ge=0)
    """Maximum total prompt tokens of requests currently in the prefill phase, or None for no limit."""
    is_multimodal_model: bool = False
    """True if the model is multimodal."""
    policy: scheduler_types.SchedulerPolicy = "fcfs"
    """The policy type for expert parallel load balancing (EPLB)."""
    disable_chunked_mm_input: bool = False
    """If set to true and chunked prefill is enabled, we do not want to partially schedule a multimodal item."""
    scheduler_cls: str | type[object] | None = None
    """The scheduler class to use."""
    disable_hybrid_kv_cache_manager: bool | None = None
    """If set to True, KV cache manager will allocate the same size of KV cache for all attention layers even if there are multiple type of attention layers like full attention and sliding window attention."""
    scheduler_reserve_full_isl: bool = True
    """If True, the scheduler checks whether the full input sequence length fits in the KV cache before admitting a new request, rather than only checking the first chunk."""
    watermark: float = Field(default=0.0, ge=0.0, lt=1.0)
    """Fraction of total KV cache blocks to keep free (the watermark) when admitting waiting or preempted requests into the running queue."""
    prefill_schedule_interval: int = Field(default=1, ge=1)
    """For data-parallel deployments, only admit new prefill requests once every N engine steps, aligned across DP ranks, to better balance per-step forward-pass times."""
    stream_interval: int = Field(default=1, ge=1)
    """The interval (or buffer size) for streaming in terms of token length."""


@config(kw_only=True)
class ParallelConfigTransportFields:
    pipeline_parallel_size: int = Field(default=1, ge=1)
    """Number of pipeline parallel groups."""
    tensor_parallel_size: int = Field(default=1, ge=1)
    """Number of tensor parallel groups."""
    prefill_context_parallel_size: int = Field(default=1, ge=1)
    """Number of ranks that split prefill sequence computation."""
    data_parallel_size: int = Field(default=1, ge=1)
    """Number of data parallel groups."""
    data_parallel_rank_local: int | None = None
    """Local rank of the data parallel group, set only in SPMD mode."""
    data_parallel_master_ip: str = "127.0.0.1"
    """IP of the data parallel master."""
    dp_sync_interval: int = Field(default=16, ge=1)
    """Steps between DP finish-sync all-reduces. Use the same value on all DP ranks."""
    data_parallel_master_port: int = 29500
    """Port of the data parallel master."""
    data_parallel_backend: parallel_types.DataParallelBackend = "mp"
    """Backend to use for data parallel, either "mp" or "ray"."""
    data_parallel_external_lb: bool = False
    """Whether to use "external" DP LB mode."""
    data_parallel_hybrid_lb: bool = False
    """Whether to use "hybrid" DP LB mode."""
    is_moe_model: bool | None = None
    """Whether the deployed model is MoE (if known)."""
    enable_expert_parallel: bool = False
    """Use expert parallelism instead of tensor parallelism for MoE layers."""
    enable_batch_sharded_sampling: bool | None = None
    """Use sharded sampling across tensor parallel ranks."""
    enable_ep_weight_filter: bool = False
    """Skip non-local expert weights during model loading when expert parallelism is active."""
    enable_eplb: bool = False
    """Enable expert parallelism load balancing for MoE layers."""
    eplb_config: parallel_types.EPLBConfig = Field(default_factory=parallel_types.EPLBConfig)
    """Expert parallelism configuration."""
    expert_placement_strategy: parallel_types.ExpertPlacementStrategy = "linear"
    """The expert placement strategy for MoE layers: - "linear": Experts are placed in a contiguous manner."""
    all2all_backend: parallel_types.All2AllBackend = "allgather_reducescatter"
    """All2All backend for MoE expert parallel communication."""
    max_parallel_loading_workers: int | None = Field(default=None, ge=1)
    """Maximum number of parallel loading workers when loading model sequentially in multiple batches."""
    disable_custom_all_reduce: bool = False
    """Disable the custom all-reduce kernel and fall back to NCCL."""
    enable_elastic_ep: bool = False
    """Enable elastic expert parallelism with stateless NCCL groups for DP/EP."""
    enable_dbo: bool = False
    """Enable dual batch overlap for the model executor."""
    ubatch_size: int = Field(default=0, ge=0)
    """Number of ubatch size."""
    dbo_decode_token_threshold: int = Field(default=32, ge=0)
    """The threshold for dual batch overlap for batches only containing decodes."""
    dbo_prefill_token_threshold: int = Field(default=512, ge=0)
    """The threshold for dual batch overlap for batches that contain one or more prefills."""
    disable_nccl_for_dp_synchronization: bool | None = None
    """Forces the dp synchronization logic in aphrodite/v1/worker/dp_utils.py to use Gloo instead of NCCL for its all reduce."""
    ray_workers_use_nsight: bool = False
    """Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler."""
    ray_runtime_env: parallel_types.RuntimeEnv | None = None
    """Ray runtime environment to pass to distributed workers."""
    placement_group: parallel_types.PlacementGroup | None = None
    """ray distributed model workers placement group."""
    distributed_executor_backend: (
        str | parallel_types.DistributedExecutorBackend | type[parallel_types.Executor] | None
    ) = None
    """Backend to use for distributed model workers, either "ray" or "mp" (multiprocessing)."""
    sd_worker_cls: str = "auto"
    """The full name of the worker class to use for speculative decoding."""
    worker_extension_cls: str = ""
    """The full name of the worker extension class to use."""
    master_addr: str = "127.0.0.1"
    """distributed master address for multi-node distributed inference when distributed_executor_backend is mp."""
    master_port: int = 29501
    """distributed master port for multi-node distributed inference when distributed_executor_backend is mp."""
    node_rank: int = Field(default=0, ge=0)
    """distributed node rank for multi-node distributed inference when distributed_executor_backend is mp."""
    nnodes: int = Field(default=1, ge=1)
    """num of nodes for multi-node distributed inference when distributed_executor_backend is mp."""
    numa_bind: bool = False
    """Enable NUMA binding for GPU worker subprocesses."""
    numa_bind_nodes: list[int] | None = None
    """NUMA node to bind each GPU worker to."""
    numa_bind_cpus: list[str] | None = None
    """Optional CPU lists to bind each GPU worker to."""
    assigned_physical_gpu_ids: list[int] | None = None
    """Mapping from Aphrodite-local logical GPU IDs to physical GPU IDs."""
    distributed_timeout_seconds: int | None = None
    """Timeout in seconds for distributed operations (e.g., init_process_group)."""
    cpu_distributed_timeout_seconds: int | None = None
    """Timeout (in seconds) for cpu communication groups."""
    world_size: int = Field(init=False)
    """world_size is TPxPP, it affects the number of workers we create."""
    rank: int = 0
    """Global rank in distributed setup."""
    _data_parallel_master_port_list: list[int] = Field(default_factory=list)
    """List of open port auto-queried for data parallel messaging."""
    _coord_store_port: int = 0
    """Port of the coordination TCPStore."""
    decode_context_parallel_size: int = Field(default=1, ge=1)
    """Number of ranks that shard the decode KV cache."""
    dcp_kv_cache_interleave_size: int = 1
    """Interleave size of kv_cache storage while using DCP."""
    dcp_comm_backend: parallel_types.DCPCommBackend | None = None
    """Communication backend for Decode Context Parallel (DCP)."""
    dcp_q_replicate: bool | None = None
    """Replicate the MLA query projection within each DCP group so decode can skip the query all-gather."""
    cp_kv_cache_interleave_size: int = 1
    """Interleave size of kv_cache storage while using DCP."""
    _api_process_count: int = Field(default=1, gt=0)
    """The number of API processes initialized."""
    _api_process_rank: int = Field(default=0, ge=-1)
    """The rank of this API process, or `-1` for engine core processes under API server scale-out."""
    enable_fault_tolerance: bool = False
    """Enable fault tolerance for detailed error recovery, such as scaling down fault DPEngineCore."""
    fault_tolerance_config: parallel_types.FaultToleranceConfig = Field(
        default_factory=parallel_types.FaultToleranceConfig
    )
    """The configurations for fault tolerance."""
