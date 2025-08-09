import hashlib
import os
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    APHRODITE_HOST_IP: str = ""
    APHRODITE_PORT: Optional[int] = None
    APHRODITE_RPC_BASE_PATH: str = tempfile.gettempdir()
    APHRODITE_USE_MODELSCOPE: bool = False
    APHRODITE_RINGBUFFER_WARNING_INTERVAL: int = 60
    APHRODITE_NCCL_SO_PATH: Optional[str] = None
    LD_LIBRARY_PATH: Optional[str] = None
    APHRODITE_USE_TRITON_FLASH_ATTN: bool = True
    APHRODITE_V1_USE_PREFILL_DECODE_ATTENTION: bool = False
    APHRODITE_USE_AITER_UNIFIED_ATTENTION: bool = False
    APHRODITE_FLASH_ATTN_VERSION: Optional[int] = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    APHRODITE_ENGINE_ITERATION_TIMEOUT_S: int = 60
    APHRODITE_API_KEY: Optional[str] = None
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None
    APHRODITE_MODEL_REDIRECT_PATH: Optional[str] = None
    APHRODITE_CACHE_ROOT: str = os.path.expanduser("~/.cache/aphrodite")
    APHRODITE_CONFIG_ROOT: str = os.path.expanduser("~/.config/aphrodite")
    APHRODITE_USAGE_STATS_SERVER: str = ""
    APHRODITE_NO_USAGE_STATS: bool = False
    APHRODITE_DO_NOT_TRACK: bool = False
    APHRODITE_USAGE_SOURCE: str = ""
    APHRODITE_CONFIGURE_LOGGING: int = 1
    APHRODITE_LOGGING_LEVEL: str = "INFO"
    APHRODITE_LOGGING_PREFIX: str = ""
    APHRODITE_LOGGING_CONFIG_PATH: Optional[str] = None
    APHRODITE_LOGITS_PROCESSOR_THREADS: Optional[int] = None
    APHRODITE_TRACE_FUNCTION: int = 0
    APHRODITE_ATTENTION_BACKEND: Optional[str] = None
    APHRODITE_USE_FLASHINFER_SAMPLER: Optional[bool] = None
    APHRODITE_FLASHINFER_FORCE_TENSOR_CORES: bool = False
    APHRODITE_PP_LAYER_PARTITION: Optional[str] = None
    APHRODITE_CPU_KVCACHE_SPACE: Optional[int] = 0
    APHRODITE_CPU_OMP_THREADS_BIND: str = ""
    APHRODITE_CPU_NUM_OF_RESERVED_CPU: Optional[int] = None
    APHRODITE_CPU_MOE_PREPACK: bool = True
    APHRODITE_CPU_SGL_KERNEL: bool = False
    APHRODITE_XLA_CACHE_PATH: str = os.path.join(APHRODITE_CACHE_ROOT, "xla_cache")
    APHRODITE_XLA_CHECK_RECOMPILATION: bool = False
    APHRODITE_FUSED_MOE_CHUNK_SIZE: int = 64 * 1024
    APHRODITE_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING: bool = True
    APHRODITE_USE_RAY_SPMD_WORKER: bool = False
    APHRODITE_USE_RAY_COMPILED_DAG: bool = False
    APHRODITE_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: str = "auto"
    APHRODITE_USE_RAY_COMPILED_DAG_OVERLAP_COMM: bool = False
    APHRODITE_USE_RAY_WRAPPED_PP_COMM: bool = True
    APHRODITE_XLA_USE_SPMD: bool = False
    APHRODITE_WORKER_MULTIPROC_METHOD: str = "fork"
    APHRODITE_ASSETS_CACHE: str = os.path.join(APHRODITE_CACHE_ROOT, "assets")
    APHRODITE_IMAGE_FETCH_TIMEOUT: int = 5
    APHRODITE_VIDEO_FETCH_TIMEOUT: int = 30
    APHRODITE_AUDIO_FETCH_TIMEOUT: int = 10
    APHRODITE_MAX_AUDIO_CLIP_FILESIZE_MB: int = 25
    APHRODITE_VIDEO_LOADER_BACKEND: str = "opencv"
    APHRODITE_MM_INPUT_CACHE_GIB: int = 8
    APHRODITE_TARGET_DEVICE: str = "cuda"
    MAX_JOBS: Optional[str] = None
    NVCC_THREADS: Optional[str] = None
    APHRODITE_USE_PRECOMPILED: bool = False
    APHRODITE_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL: bool = False
    APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH: bool = False
    CMAKE_BUILD_TYPE: Optional[str] = None
    VERBOSE: bool = False
    APHRODITE_ALLOW_LONG_MAX_MODEL_LEN: bool = False
    APHRODITE_RPC_TIMEOUT: int = 10000  # ms
    APHRODITE_HTTP_TIMEOUT_KEEP_ALIVE: int = 5  # seconds
    APHRODITE_PLUGINS: Optional[list[str]] = None
    APHRODITE_LORA_RESOLVER_CACHE_DIR: Optional[str] = None
    APHRODITE_TORCH_PROFILER_DIR: Optional[str] = None
    APHRODITE_TORCH_PROFILER_RECORD_SHAPES: bool = False
    APHRODITE_TORCH_PROFILER_WITH_PROFILE_MEMORY: bool = False
    APHRODITE_TORCH_PROFILER_WITH_STACK: bool = True
    APHRODITE_TORCH_PROFILER_WITH_FLOPS: bool = False
    APHRODITE_USE_TRITON_AWQ: bool = False
    APHRODITE_ALLOW_RUNTIME_LORA_UPDATING: bool = False
    APHRODITE_SKIP_P2P_CHECK: bool = False
    APHRODITE_DISABLED_KERNELS: list[str] = []
    APHRODITE_USE_V1: bool = True
    APHRODITE_ROCM_USE_AITER: bool = False
    APHRODITE_ROCM_USE_AITER_PAGED_ATTN: bool = False
    APHRODITE_ROCM_USE_AITER_LINEAR: bool = True
    APHRODITE_ROCM_USE_AITER_MOE: bool = True
    APHRODITE_ROCM_USE_AITER_RMSNORM: bool = True
    APHRODITE_ROCM_USE_AITER_MLA: bool = True
    APHRODITE_ROCM_USE_AITER_MHA: bool = True
    APHRODITE_ROCM_USE_SKINNY_GEMM: bool = True
    APHRODITE_ROCM_FP8_PADDING: bool = True
    APHRODITE_ROCM_MOE_PADDING: bool = True
    APHRODITE_ROCM_CUSTOM_PAGED_ATTN: bool = True
    APHRODITE_ENABLE_V1_MULTIPROCESSING: bool = True
    APHRODITE_LOG_BATCHSIZE_INTERVAL: float = -1
    APHRODITE_DISABLE_COMPILE_CACHE: bool = False
    Q_SCALE_CONSTANT: int = 200
    K_SCALE_CONSTANT: int = 200
    V_SCALE_CONSTANT: int = 100
    APHRODITE_SERVER_DEV_MODE: bool = False
    APHRODITE_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128
    APHRODITE_MLA_DISABLE: bool = False
    APHRODITE_RAY_PER_WORKER_GPUS: float = 1.0
    APHRODITE_RAY_BUNDLE_INDICES: str = ""
    APHRODITE_CUDART_SO_PATH: Optional[str] = None
    APHRODITE_DP_RANK: int = 0
    APHRODITE_DP_RANK_LOCAL: int = -1
    APHRODITE_DP_SIZE: int = 1
    APHRODITE_DP_MASTER_IP: str = ""
    APHRODITE_DP_MASTER_PORT: int = 0
    APHRODITE_MOE_DP_CHUNK_SIZE: int = 256
    APHRODITE_RANDOMIZE_DP_DUMMY_INPUTS: bool = False
    APHRODITE_MARLIN_USE_ATOMIC_ADD: bool = False
    APHRODITE_V0_USE_OUTLINES_CACHE: bool = False
    APHRODITE_V1_USE_OUTLINES_CACHE: bool = False
    APHRODITE_TPU_BUCKET_PADDING_GAP: int = 0
    APHRODITE_TPU_MOST_MODEL_LEN: Optional[int] = None
    APHRODITE_TPU_USING_PATHWAYS: bool = False
    APHRODITE_USE_DEEP_GEMM: bool = False
    APHRODITE_SKIP_DEEP_GEMM_WARMUP: bool = False
    APHRODITE_USE_FLASHINFER_MOE_FP8: bool = False
    APHRODITE_USE_FLASHINFER_MOE_FP4: bool = False
    APHRODITE_XGRAMMAR_CACHE_MB: int = 0
    APHRODITE_MSGPACK_ZERO_COPY_THRESHOLD: int = 256
    APHRODITE_ALLOW_INSECURE_SERIALIZATION: bool = False
    APHRODITE_NIXL_SIDE_CHANNEL_HOST: str = "localhost"
    APHRODITE_NIXL_SIDE_CHANNEL_PORT: int = 5557
    APHRODITE_ALL2ALL_BACKEND: str = "naive"
    APHRODITE_MAX_TOKENS_PER_EXPERT_FP4_MOE: int = 163840
    APHRODITE_TOOL_PARSE_REGEX_TIMEOUT_SECONDS: int = 1
    APHRODITE_SLEEP_WHEN_IDLE: bool = False
    APHRODITE_MQ_MAX_CHUNK_BYTES_MB: int = 16
    APHRODITE_EXECUTE_MODEL_TIMEOUT_SECONDS: int = 300
    APHRODITE_KV_CACHE_LAYOUT: Optional[str] = None
    APHRODITE_COMPUTE_NANS_IN_LOGITS: bool = False
    APHRODITE_USE_NVFP4_CT_EMULATIONS: bool = False
    APHRODITE_ROCM_QUICK_REDUCE_QUANTIZATION: str = "NONE"
    APHRODITE_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16: bool = True
    APHRODITE_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB: Optional[int] = None
    APHRODITE_NIXL_ABORT_REQUEST_TIMEOUT: int = 120
    APHRODITE_USE_CUDNN_PREFILL: bool = False
    APHRODITE_ENABLE_CUDAGRAPH_GC: bool = False
    APHRODITE_LOOPBACK_IP: str = ""
    APHRODITE_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: bool = False
    APHRODITE_ENABLE_RESPONSES_API_STORE: bool = False
    APHRODITE_USE_TRTLLM_CONTEXT_ATTENTION: bool = False
    APHRODITE_USE_TRTLLM_DECODE_ATTENTION: bool = False
    APHRODITE_KOBOLD_API: bool = False
    APHRODITE_REQUEST_LEVEL_METRICS: bool = False
    APHRODITE_USE_SAMPLING_KERNELS: bool = False


def get_default_cache_root():
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root():
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def get_aphrodite_port() -> Optional[int]:
    """Get the port from APHRODITE_PORT environment variable.
    Returns:
        The port number as an integer if APHRODITE_PORT is set, None otherwise.
    Raises:
        ValueError: If APHRODITE_PORT is a URI, suggest k8s service discovery
        issue.
    """
    if 'APHRODITE_PORT' not in os.environ:
        return None

    port = os.getenv('APHRODITE_PORT', '0')

    try:
        return int(port)
    except ValueError as err:
        from urllib.parse import urlparse
        parsed = urlparse(port)
        if parsed.scheme:
            raise ValueError(
                f"APHRODITE_PORT '{port}' appears to be a URI. "
                "This may be caused by a Kubernetes service discovery issue."
            ) from None
        raise ValueError(
            f"APHRODITE_PORT '{port}' must be a valid integer") from err


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

environment_variables: dict[str, Callable[[], Any]] = {

    # ================== Installation Time Env Vars ==================

    # Target device of Aphrodite, supporting [cuda (by default),
    # rocm, neuron, cpu]
    "APHRODITE_TARGET_DEVICE":
    lambda: os.getenv("APHRODITE_TARGET_DEVICE", "cuda").lower(),

    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),

    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS":
    lambda: os.getenv("NVCC_THREADS", None),

    # If set, aphrodite will use precompiled binaries (*.so)
    "APHRODITE_USE_PRECOMPILED":
    lambda: bool(os.environ.get("APHRODITE_USE_PRECOMPILED")) or bool(
        os.environ.get("APHRODITE_PRECOMPILED_WHEEL_LOCATION")),

    # Whether to force using nightly wheel in python build.
    # This is used for testing the nightly wheel in python build.
    "APHRODITE_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL":
    lambda: bool(int(os.getenv("APHRODITE_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL", "0"))
                 ),

    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),

    # If set, aphrodite will print verbose logs during installation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),

    # Root directory for Aphrodite configuration files
    # Defaults to `~/.config/aphrodite` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how aphrodite finds its configuration files
    # during runtime, but also affects how aphrodite installs its configuration
    # files during **installation**.
    "APHRODITE_CONFIG_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "APHRODITE_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "aphrodite"),
        )),

    # ================== Runtime Env Vars ==================

    # Root directory for Aphrodite cache files
    # Defaults to `~/.cache/aphrodite` unless `XDG_CACHE_HOME` is set
    "APHRODITE_CACHE_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "APHRODITE_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "aphrodite"),
        )),

    # used in distributed environment to determine the ip address
    # of the current node, when the node has multiple network interfaces.
    # If you are using multi-node inference, you should set this differently
    # on each node.
    'APHRODITE_HOST_IP':
    lambda: os.getenv('APHRODITE_HOST_IP', ""),

    # used in distributed environment to manually set the communication port
    # Note: if APHRODITE_PORT is set, and some code asks for multiple ports, the
    # APHRODITE_PORT will be used as the first port, and the rest will be generated
    # by incrementing the APHRODITE_PORT value.
    'APHRODITE_PORT':
    get_aphrodite_port,

    # path used for ipc when the frontend api server is running in
    # multi-processing mode to communicate with the backend engine process.
    'APHRODITE_RPC_BASE_PATH':
    lambda: os.getenv('APHRODITE_RPC_BASE_PATH', tempfile.gettempdir()),

    # If true, will load models from ModelScope instead of Hugging Face Hub.
    # note that the value is true or false, not numbers
    "APHRODITE_USE_MODELSCOPE":
    lambda: os.environ.get("APHRODITE_USE_MODELSCOPE", "False").lower() == "true",

    # Interval in seconds to log a warning message when the ring buffer is full
    "APHRODITE_RINGBUFFER_WARNING_INTERVAL":
    lambda: int(os.environ.get("APHRODITE_RINGBUFFER_WARNING_INTERVAL", "60")),

    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME":
    lambda: os.environ.get("CUDA_HOME", None),

    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "APHRODITE_NCCL_SO_PATH":
    lambda: os.environ.get("APHRODITE_NCCL_SO_PATH", None),

    # when `APHRODITE_NCCL_SO_PATH` is not set, aphrodite will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH":
    lambda: os.environ.get("LD_LIBRARY_PATH", None),

    # flag to control if aphrodite should use triton flash attention
    "APHRODITE_USE_TRITON_FLASH_ATTN":
    lambda: (os.environ.get("APHRODITE_USE_TRITON_FLASH_ATTN", "True").lower() in
             ("true", "1")),

    # Use separate prefill and decode kernels for V1 attention instead of
    # the unified triton kernel.
    "APHRODITE_V1_USE_PREFILL_DECODE_ATTENTION":
    lambda:
    (os.getenv("APHRODITE_V1_USE_PREFILL_DECODE_ATTENTION", "False").lower() in
     ("true", "1")),

    # Use AITER triton unified attention for V1 attention
    "APHRODITE_USE_AITER_UNIFIED_ATTENTION":
    lambda:
    (os.getenv("APHRODITE_USE_AITER_UNIFIED_ATTENTION", "False").lower() in
     ("true", "1")),

    # Force aphrodite to use a specific flash-attention version (2 or 3), only valid
    # when using the flash-attention backend.
    "APHRODITE_FLASH_ATTN_VERSION":
    lambda: maybe_convert_int(os.environ.get("APHRODITE_FLASH_ATTN_VERSION", None)),

    # Internal flag to enable Dynamo fullgraph capture
    "APHRODITE_TEST_DYNAMO_FULLGRAPH_CAPTURE":
    lambda: bool(
        os.environ.get("APHRODITE_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1") != "0"),

    # Feature flag to enable/disable Inductor standalone compile.
    # In torch <= 2.7 we ignore this flag; in torch >= 2.8 this is
    # enabled by default.
    "APHRODITE_USE_STANDALONE_COMPILE":
    lambda: os.environ.get("APHRODITE_USE_STANDALONE_COMPILE", "1") == "1",

    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK":
    lambda: int(os.environ.get("LOCAL_RANK", "0")),

    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES":
    lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),

    # timeout for each iteration in the engine
    "APHRODITE_ENGINE_ITERATION_TIMEOUT_S":
    lambda: int(os.environ.get("APHRODITE_ENGINE_ITERATION_TIMEOUT_S", "60")),

    # API key for Aphrodite API server
    "APHRODITE_API_KEY":
    lambda: os.environ.get("APHRODITE_API_KEY", None),

    # Whether to log responses from API Server for debugging
    "APHRODITE_DEBUG_LOG_API_SERVER_RESPONSE":
    lambda: os.environ.get("APHRODITE_DEBUG_LOG_API_SERVER_RESPONSE", "False"
                           ).lower() == "true",

    # S3 access information, used for tensorizer to load model from S3
    "S3_ACCESS_KEY_ID":
    lambda: os.environ.get("S3_ACCESS_KEY_ID", None),
    "S3_SECRET_ACCESS_KEY":
    lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),
    "S3_ENDPOINT_URL":
    lambda: os.environ.get("S3_ENDPOINT_URL", None),

    # Usage stats collection
    "APHRODITE_USAGE_STATS_SERVER":
    lambda: os.environ.get("APHRODITE_USAGE_STATS_SERVER", "https://stats.aphrodite.ai"),
    "APHRODITE_NO_USAGE_STATS":
    lambda: os.environ.get("APHRODITE_NO_USAGE_STATS", "0") == "1",
    "APHRODITE_DO_NOT_TRACK":
    lambda: (os.environ.get("APHRODITE_DO_NOT_TRACK", None) or os.environ.get(
        "DO_NOT_TRACK", None) or "0") == "1",
    "APHRODITE_USAGE_SOURCE":
    lambda: os.environ.get("APHRODITE_USAGE_SOURCE", "production"),

    # Logging configuration
    # If set to 0, aphrodite will not configure logging
    # If set to 1, aphrodite will configure logging using the default configuration
    #    or the configuration file specified by APHRODITE_LOGGING_CONFIG_PATH
    "APHRODITE_CONFIGURE_LOGGING":
    lambda: int(os.getenv("APHRODITE_CONFIGURE_LOGGING", "1")),
    "APHRODITE_LOGGING_CONFIG_PATH":
    lambda: os.getenv("APHRODITE_LOGGING_CONFIG_PATH"),

    # this is used for configuring the default logging level
    "APHRODITE_LOGGING_LEVEL":
    lambda: os.getenv("APHRODITE_LOGGING_LEVEL", "INFO").upper(),

    # if set, APHRODITE_LOGGING_PREFIX will be prepended to all log messages
    "APHRODITE_LOGGING_PREFIX":
    lambda: os.getenv("APHRODITE_LOGGING_PREFIX", ""),

    # if set, aphrodite will call logits processors in a thread pool with this many
    # threads. This is useful when using custom logits processors that either
    # (a) launch additional CUDA kernels or (b) do significant CPU-bound work
    # while not holding the python GIL, or both.
    "APHRODITE_LOGITS_PROCESSOR_THREADS":
    lambda: int(os.getenv("APHRODITE_LOGITS_PROCESSOR_THREADS", "0"))
    if "APHRODITE_LOGITS_PROCESSOR_THREADS" in os.environ else None,

    # Trace function calls
    # If set to 1, aphrodite will trace function calls
    # Useful for debugging
    "APHRODITE_TRACE_FUNCTION":
    lambda: int(os.getenv("APHRODITE_TRACE_FUNCTION", "0")),

    # Backend for attention computation
    # Available options:
    # - "TORCH_SDPA": use torch.nn.MultiheadAttention
    # - "FLASH_ATTN": use FlashAttention
    # - "XFORMERS": use XFormers
    # - "ROCM_FLASH": use ROCmFlashAttention
    # - "FLASHINFER": use flashinfer
    # - "FLASHMLA": use FlashMLA
    "APHRODITE_ATTENTION_BACKEND":
    lambda: os.getenv("APHRODITE_ATTENTION_BACKEND", None),

    # If set, aphrodite will use flashinfer sampler
    "APHRODITE_USE_FLASHINFER_SAMPLER":
    lambda: bool(int(os.environ["APHRODITE_USE_FLASHINFER_SAMPLER"]))
    if "APHRODITE_USE_FLASHINFER_SAMPLER" in os.environ else None,

    # If set, aphrodite will force flashinfer to use tensor cores;
    # otherwise will use heuristic based on model architecture.
    "APHRODITE_FLASHINFER_FORCE_TENSOR_CORES":
    lambda: bool(int(os.getenv("APHRODITE_FLASHINFER_FORCE_TENSOR_CORES", "0"))),

    # Pipeline stage partition strategy
    "APHRODITE_PP_LAYER_PARTITION":
    lambda: os.getenv("APHRODITE_PP_LAYER_PARTITION", None),

    # (CPU backend only) CPU key-value cache space.
    # default is None and will be set as 4 GB
    "APHRODITE_CPU_KVCACHE_SPACE":
    lambda: int(os.getenv("APHRODITE_CPU_KVCACHE_SPACE", "0"))
    if "APHRODITE_CPU_KVCACHE_SPACE" in os.environ else None,

    # (CPU backend only) CPU core ids bound by OpenMP threads, e.g., "0-31",
    # "0,1,2", "0-31,33". CPU cores of different ranks are separated by '|'.
    "APHRODITE_CPU_OMP_THREADS_BIND":
    lambda: os.getenv("APHRODITE_CPU_OMP_THREADS_BIND", "auto"),

    # (CPU backend only) CPU cores not used by OMP threads .
    # Those CPU cores will not be used by OMP threads of a rank.
    "APHRODITE_CPU_NUM_OF_RESERVED_CPU":
    lambda: int(os.getenv("APHRODITE_CPU_NUM_OF_RESERVED_CPU", "0"))
    if "APHRODITE_CPU_NUM_OF_RESERVED_CPU" in os.environ else None,

    # (CPU backend only) whether to use prepack for MoE layer. This will be
    # passed to ipex.llm.modules.GatedMLPMOE. On unsupported CPUs, you might
    # need to set this to "0" (False).
    "APHRODITE_CPU_MOE_PREPACK":
    lambda: bool(int(os.getenv("APHRODITE_CPU_MOE_PREPACK", "1"))),

    # (CPU backend only) whether to use SGL kernels, optimized for small batch.
    "APHRODITE_CPU_SGL_KERNEL":
    lambda: bool(int(os.getenv("APHRODITE_CPU_SGL_KERNEL", "0"))),

    # If the env var is set, then all workers will execute as separate
    # processes from the engine, and we use the same mechanism to trigger
    # execution on all workers.
    # Run Aphrodite with APHRODITE_USE_RAY_SPMD_WORKER=1 to enable it.
    "APHRODITE_USE_RAY_SPMD_WORKER":
    lambda: bool(int(os.getenv("APHRODITE_USE_RAY_SPMD_WORKER", "0"))),

    # If the env var is set, it uses the Ray's Compiled Graph
    # (previously known as ADAG) API which optimizes the
    # control plane overhead.
    # Run Aphrodite with APHRODITE_USE_RAY_COMPILED_DAG=1 to enable it.
    # Note that this variable is set to 1 in V1 by default
    # when ray distributed executor is used.
    "APHRODITE_USE_RAY_COMPILED_DAG":
    lambda: bool(int(os.getenv("APHRODITE_USE_RAY_COMPILED_DAG", "0"))),

    # If the env var is set, Ray Compiled Graph uses the specified
    # channel type to communicate between workers belonging to
    # different pipeline-parallel stages.
    # Available options:
    # - "auto": use the default channel type
    # - "nccl": use NCCL for communication
    # - "shm": use shared memory and gRPC for communication
    # This flag is ignored if APHRODITE_USE_RAY_COMPILED_DAG is not set.
    "APHRODITE_USE_RAY_COMPILED_DAG_CHANNEL_TYPE":
    lambda: os.getenv("APHRODITE_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "auto"),

    # If the env var is set, it enables GPU communication overlap
    # (experimental feature) in Ray's Compiled Graph. This flag is ignored if
    # APHRODITE_USE_RAY_COMPILED_DAG is not set.
    "APHRODITE_USE_RAY_COMPILED_DAG_OVERLAP_COMM":
    lambda: bool(int(os.getenv("APHRODITE_USE_RAY_COMPILED_DAG_OVERLAP_COMM", "0"))
                 ),

    # If the env var is set, it uses a Ray Communicator wrapping
    # Aphrodite's pipeline parallelism communicator to interact with Ray's
    # Compiled Graph. Otherwise, it uses Ray's NCCL communicator.
    # This flag is ignored if APHRODITE_USE_RAY_COMPILED_DAG is not set.
    "APHRODITE_USE_RAY_WRAPPED_PP_COMM":
    lambda: bool(int(os.getenv("APHRODITE_USE_RAY_WRAPPED_PP_COMM", "1"))),

    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "APHRODITE_WORKER_MULTIPROC_METHOD":
    lambda: os.getenv("APHRODITE_WORKER_MULTIPROC_METHOD", "fork"),

    # Path to the cache for storing downloaded assets
    "APHRODITE_ASSETS_CACHE":
    lambda: os.path.expanduser(
        os.getenv(
            "APHRODITE_ASSETS_CACHE",
            os.path.join(get_default_cache_root(), "aphrodite", "assets"),
        )),

    # Timeout for fetching images when serving multimodal models
    # Default is 5 seconds
    "APHRODITE_IMAGE_FETCH_TIMEOUT":
    lambda: int(os.getenv("APHRODITE_IMAGE_FETCH_TIMEOUT", "5")),

    # Timeout for fetching videos when serving multimodal models
    # Default is 30 seconds
    "APHRODITE_VIDEO_FETCH_TIMEOUT":
    lambda: int(os.getenv("APHRODITE_VIDEO_FETCH_TIMEOUT", "30")),

    # Timeout for fetching audio when serving multimodal models
    # Default is 10 seconds
    "APHRODITE_AUDIO_FETCH_TIMEOUT":
    lambda: int(os.getenv("APHRODITE_AUDIO_FETCH_TIMEOUT", "10")),

    # Maximum filesize in MB for a single audio file when processing
    # speech-to-text requests. Files larger than this will be rejected.
    # Default is 25 MB
    "APHRODITE_MAX_AUDIO_CLIP_FILESIZE_MB":
    lambda: int(os.getenv("APHRODITE_MAX_AUDIO_CLIP_FILESIZE_MB", "25")),

    # Backend for Video IO
    # - "opencv": Default backend that uses OpenCV stream buffered backend.
    #
    # Custom backend implementations can be registered
    # via `@VIDEO_LOADER_REGISTRY.register("my_custom_video_loader")` and
    # imported at runtime.
    # If a non-existing backend is used, an AssertionError will be thrown.
    "APHRODITE_VIDEO_LOADER_BACKEND":
    lambda: os.getenv("APHRODITE_VIDEO_LOADER_BACKEND", "opencv"),

    # Cache size (in GiB) for multimodal input cache
    # Default is 4 GiB
    "APHRODITE_MM_INPUT_CACHE_GIB":
    lambda: int(os.getenv("APHRODITE_MM_INPUT_CACHE_GIB", "4")),

    # Path to the XLA persistent cache directory.
    # Only used for XLA devices such as TPUs.
    "APHRODITE_XLA_CACHE_PATH":
    lambda: os.path.expanduser(
        os.getenv(
            "APHRODITE_XLA_CACHE_PATH",
            os.path.join(get_default_cache_root(), "aphrodite", "xla_cache"),
        )),

    # If set, assert on XLA recompilation after each execution step.
    "APHRODITE_XLA_CHECK_RECOMPILATION":
    lambda: bool(int(os.getenv("APHRODITE_XLA_CHECK_RECOMPILATION", "0"))),

    # Enable SPMD mode for TPU backend.
    "APHRODITE_XLA_USE_SPMD":
    lambda: bool(int(os.getenv("APHRODITE_XLA_USE_SPMD", "0"))),
    "APHRODITE_FUSED_MOE_CHUNK_SIZE":
    lambda: int(os.getenv("APHRODITE_FUSED_MOE_CHUNK_SIZE", "32768")),
    # Control whether to use fused MoE activation chunking. Current chunking
    # logic is incompatible with torch.compile and causes IMA. See issue
    # https://github.com/aphrodite-project/aphrodite/issues/19631.
    "APHRODITE_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING":
    lambda: bool(
        int(os.getenv("APHRODITE_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING", "1"))),

    # If set, the OpenAI API server will stay alive even after the underlying
    # AsyncLLMEngine errors and stops serving requests
    "APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH":
    lambda: bool(os.getenv("APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH", 0)),

    # If the env var APHRODITE_ALLOW_LONG_MAX_MODEL_LEN is set, it allows
    # the user to specify a max sequence length greater than
    # the max length derived from the model's config.json.
    # To enable this, set APHRODITE_ALLOW_LONG_MAX_MODEL_LEN=1.
    "APHRODITE_ALLOW_LONG_MAX_MODEL_LEN":
    lambda:
    (os.environ.get("APHRODITE_ALLOW_LONG_MAX_MODEL_LEN", "0").strip().lower() in
     ("1", "true")),

    # If set, forces FP8 Marlin to be used for FP8 quantization regardless
    # of the hardware support for FP8 compute.
    "APHRODITE_TEST_FORCE_FP8_MARLIN":
    lambda:
    (os.environ.get("APHRODITE_TEST_FORCE_FP8_MARLIN", "0").strip().lower() in
     ("1", "true")),
    "APHRODITE_TEST_FORCE_LOAD_FORMAT":
    lambda: os.getenv("APHRODITE_TEST_FORCE_LOAD_FORMAT", "dummy"),

    # Time in ms for the zmq client to wait for a response from the backend
    # server for simple data operations
    "APHRODITE_RPC_TIMEOUT":
    lambda: int(os.getenv("APHRODITE_RPC_TIMEOUT", "10000")),

    # Timeout in seconds for keeping HTTP connections alive in API server
    "APHRODITE_HTTP_TIMEOUT_KEEP_ALIVE":
    lambda: int(os.environ.get("APHRODITE_HTTP_TIMEOUT_KEEP_ALIVE", "5")),

    # a list of plugin names to load, separated by commas.
    # if this is not set, it means all plugins will be loaded
    # if this is set to an empty string, no plugins will be loaded
    "APHRODITE_PLUGINS":
    lambda: None if "APHRODITE_PLUGINS" not in os.environ else os.environ[
        "APHRODITE_PLUGINS"].split(","),

    # a local directory to look in for unrecognized LoRA adapters.
    # only works if plugins are enabled and
    # APHRODITE_ALLOW_RUNTIME_LORA_UPDATING is enabled.
    "APHRODITE_LORA_RESOLVER_CACHE_DIR":
    lambda: os.getenv("APHRODITE_LORA_RESOLVER_CACHE_DIR", None),

    # Enables torch profiler if set. Path to the directory where torch profiler
    # traces are saved. Note that it must be an absolute path.
    "APHRODITE_TORCH_PROFILER_DIR":
    lambda: (None if os.getenv("APHRODITE_TORCH_PROFILER_DIR", None) is None else os
             .path.expanduser(os.getenv("APHRODITE_TORCH_PROFILER_DIR", "."))),

    # Enable torch profiler to record shapes if set
    # APHRODITE_TORCH_PROFILER_RECORD_SHAPES=1. If not set, torch profiler will
    # not record shapes.
    "APHRODITE_TORCH_PROFILER_RECORD_SHAPES":
    lambda: bool(os.getenv("APHRODITE_TORCH_PROFILER_RECORD_SHAPES", "0") != "0"),

    # Enable torch profiler to profile memory if set
    # APHRODITE_TORCH_PROFILER_WITH_PROFILE_MEMORY=1. If not set, torch profiler
    # will not profile memory.
    "APHRODITE_TORCH_PROFILER_WITH_PROFILE_MEMORY":
    lambda: bool(
        os.getenv("APHRODITE_TORCH_PROFILER_WITH_PROFILE_MEMORY", "0") != "0"),

    # Enable torch profiler to profile stack if set
    # APHRODITE_TORCH_PROFILER_WITH_STACK=1. If not set, torch profiler WILL
    # profile stack by default.
    "APHRODITE_TORCH_PROFILER_WITH_STACK":
    lambda: bool(os.getenv("APHRODITE_TORCH_PROFILER_WITH_STACK", "1") != "0"),

    # Enable torch profiler to profile flops if set
    # APHRODITE_TORCH_PROFILER_WITH_FLOPS=1. If not set, torch profiler will
    # not profile flops.
    "APHRODITE_TORCH_PROFILER_WITH_FLOPS":
    lambda: bool(os.getenv("APHRODITE_TORCH_PROFILER_WITH_FLOPS", "0") != "0"),

    # If set, Aphrodite will use Triton implementations of AWQ.
    "APHRODITE_USE_TRITON_AWQ":
    lambda: bool(int(os.getenv("APHRODITE_USE_TRITON_AWQ", "0"))),

    # If set, allow loading or unloading lora adapters in runtime,
    "APHRODITE_ALLOW_RUNTIME_LORA_UPDATING":
    lambda:
    (os.environ.get("APHRODITE_ALLOW_RUNTIME_LORA_UPDATING", "0").strip().lower() in
     ("1", "true")),

    # We assume drivers can report p2p status correctly.
    # If the program hangs when using custom allreduce,
    # potantially caused by a bug in the driver (535 series),
    # if might be helpful to set APHRODITE_SKIP_P2P_CHECK=0
    # so that Aphrodite can verify if p2p is actually working.
    # See https://github.com/aphrodite-project/aphrodite/blob/a9b15c606fea67a072416ea0ea115261a2756058/aphrodite/distributed/device_communicators/custom_all_reduce_utils.py#L101-L108 for details. # noqa
    "APHRODITE_SKIP_P2P_CHECK":
    lambda: os.getenv("APHRODITE_SKIP_P2P_CHECK", "1") == "1",

    # List of quantization kernels that should be disabled, used for testing
    # and performance comparisons. Currently only affects MPLinearKernel
    # selection
    # (kernels: MacheteLinearKernel, MarlinLinearKernel, ExllamaLinearKernel)
    "APHRODITE_DISABLED_KERNELS":
    lambda: [] if "APHRODITE_DISABLED_KERNELS" not in os.environ else os.environ[
        "APHRODITE_DISABLED_KERNELS"].split(","),

    # If set, use the V1 code path.
    "APHRODITE_USE_V1":
    lambda: bool(int(os.getenv("APHRODITE_USE_V1", "1"))),

    # Disable aiter ops unless specifically enabled.
    # Acts as a parent switch to enable the rest of the other operations.
    "APHRODITE_ROCM_USE_AITER":
    lambda: (os.getenv("APHRODITE_ROCM_USE_AITER", "False").lower() in
             ("true", "1")),

    # Whether to use aiter paged attention.
    # By default is disabled.
    "APHRODITE_ROCM_USE_AITER_PAGED_ATTN":
    lambda: (os.getenv("APHRODITE_ROCM_USE_AITER_PAGED_ATTN", "False").lower() in
             ("true", "1")),

    # use aiter linear op if aiter ops are enabled
    # The following list of related ops
    # - scaled_mm (per-tensor / rowwise)
    "APHRODITE_ROCM_USE_AITER_LINEAR":
    lambda: (os.getenv("APHRODITE_ROCM_USE_AITER_LINEAR", "True").lower() in
             ("true", "1")),

    # Whether to use aiter moe ops.
    # By default is enabled.
    "APHRODITE_ROCM_USE_AITER_MOE":
    lambda: (os.getenv("APHRODITE_ROCM_USE_AITER_MOE", "True").lower() in
             ("true", "1")),

    # use aiter rms norm op if aiter ops are enabled.
    "APHRODITE_ROCM_USE_AITER_RMSNORM":
    lambda: (os.getenv("APHRODITE_ROCM_USE_AITER_RMSNORM", "True").lower() in
             ("true", "1")),

    # Whether to use aiter mla ops.
    # By default is enabled.
    "APHRODITE_ROCM_USE_AITER_MLA":
    lambda: (os.getenv("APHRODITE_ROCM_USE_AITER_MLA", "True").lower() in
             ("true", "1")),

    # Whether to use aiter mha ops.
    # By default is enabled.
    "APHRODITE_ROCM_USE_AITER_MHA":
    lambda: (os.getenv("APHRODITE_ROCM_USE_AITER_MHA", "True").lower() in
             ("true", "1")),

    # use rocm skinny gemms
    "APHRODITE_ROCM_USE_SKINNY_GEMM":
    lambda: (os.getenv("APHRODITE_ROCM_USE_SKINNY_GEMM", "True").lower() in
             ("true", "1")),

    # Pad the fp8 weights to 256 bytes for ROCm
    "APHRODITE_ROCM_FP8_PADDING":
    lambda: bool(int(os.getenv("APHRODITE_ROCM_FP8_PADDING", "1"))),

    # Pad the weights for the moe kernel
    "APHRODITE_ROCM_MOE_PADDING":
    lambda: bool(int(os.getenv("APHRODITE_ROCM_MOE_PADDING", "1"))),

    # custom paged attention kernel for MI3* cards
    "APHRODITE_ROCM_CUSTOM_PAGED_ATTN":
    lambda: (os.getenv("APHRODITE_ROCM_CUSTOM_PAGED_ATTN", "True").lower() in
             ("true", "1")),

    # Custom quick allreduce kernel for MI3* cards
    # Choice of quantization level: FP, INT8, INT6, INT4 or NONE
    # Recommended for large models to get allreduce
    "APHRODITE_ROCM_QUICK_REDUCE_QUANTIZATION":
    lambda: os.getenv("APHRODITE_ROCM_QUICK_REDUCE_QUANTIZATION", "NONE").upper(),

    # Custom quick allreduce kernel for MI3* cards
    # Due to the lack of the bfloat16 asm instruction, bfloat16
    # kernels are slower than fp16,
    # If environment variable is set to 1, the input is converted to fp16
    "APHRODITE_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16":
    lambda:
    (os.getenv("APHRODITE_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "True").lower() in
     ("true", "1")),

    # Custom quick allreduce kernel for MI3* cards.
    # Controls the maximum allowed number of data bytes(MB) for custom quick
    # allreduce communication.
    # Default: 2048 MB.
    # Data exceeding this size will use either custom allreduce or RCCL
    # communication.
    "APHRODITE_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB":
    lambda: maybe_convert_int(
        os.environ.get("APHRODITE_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", None)),

    # Divisor for dynamic query scale factor calculation for FP8 KV Cache
    "Q_SCALE_CONSTANT":
    lambda: int(os.getenv("Q_SCALE_CONSTANT", "200")),
    # Divisor for dynamic key scale factor calculation for FP8 KV Cache
    "K_SCALE_CONSTANT":
    lambda: int(os.getenv("K_SCALE_CONSTANT", "200")),
    # Divisor for dynamic value scale factor calculation for FP8 KV Cache
    "V_SCALE_CONSTANT":
    lambda: int(os.getenv("V_SCALE_CONSTANT", "100")),

    # If set, enable multiprocessing in LLM for the V1 code path.
    "APHRODITE_ENABLE_V1_MULTIPROCESSING":
    lambda: bool(int(os.getenv("APHRODITE_ENABLE_V1_MULTIPROCESSING", "1"))),
    "APHRODITE_LOG_BATCHSIZE_INTERVAL":
    lambda: float(os.getenv("APHRODITE_LOG_BATCHSIZE_INTERVAL", "-1")),
    "APHRODITE_DISABLE_COMPILE_CACHE":
    lambda: bool(int(os.getenv("APHRODITE_DISABLE_COMPILE_CACHE", "0"))),

    # If set, aphrodite will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "APHRODITE_SERVER_DEV_MODE":
    lambda: bool(int(os.getenv("APHRODITE_SERVER_DEV_MODE", "0"))),

    # Controls the maximum number of requests to handle in a
    # single asyncio task when processing per-token outputs in the
    # V1 AsyncLLM interface. It is applicable when handling a high
    # concurrency of streaming requests.
    # Setting this too high can result in a higher variance of
    # inter-message latencies. Setting it too low can negatively impact
    # TTFT and overall throughput.
    "APHRODITE_V1_OUTPUT_PROC_CHUNK_SIZE":
    lambda: int(os.getenv("APHRODITE_V1_OUTPUT_PROC_CHUNK_SIZE", "128")),

    # If set, Aphrodite will disable the MLA attention optimizations.
    "APHRODITE_MLA_DISABLE":
    lambda: bool(int(os.getenv("APHRODITE_MLA_DISABLE", "0"))),

    # Number of GPUs per worker in Ray, if it is set to be a fraction,
    # it allows ray to schedule multiple actors on a single GPU,
    # so that users can colocate other actors on the same GPUs as Aphrodite.
    "APHRODITE_RAY_PER_WORKER_GPUS":
    lambda: float(os.getenv("APHRODITE_RAY_PER_WORKER_GPUS", "1.0")),

    # Bundle indices for Ray, if it is set, it can control precisely
    # which indices are used for the Ray bundle, for every worker.
    # Format: comma-separated list of integers, e.g. "0,1,2,3"
    "APHRODITE_RAY_BUNDLE_INDICES":
    lambda: os.getenv("APHRODITE_RAY_BUNDLE_INDICES", ""),

    # In some system, find_loaded_library() may not work. So we allow users to
    # specify the path through environment variable APHRODITE_CUDART_SO_PATH.
    "APHRODITE_CUDART_SO_PATH":
    lambda: os.getenv("APHRODITE_CUDART_SO_PATH", None),

    # Rank of the process in the data parallel setting
    "APHRODITE_DP_RANK":
    lambda: int(os.getenv("APHRODITE_DP_RANK", "0")),

    # Rank of the process in the data parallel setting.
    # Defaults to APHRODITE_DP_RANK when not set.
    "APHRODITE_DP_RANK_LOCAL":
    lambda: int(
        os.getenv("APHRODITE_DP_RANK_LOCAL", sys.modules[__name__].APHRODITE_DP_RANK)),

    # World size of the data parallel setting
    "APHRODITE_DP_SIZE":
    lambda: int(os.getenv("APHRODITE_DP_SIZE", "1")),

    # IP address of the master node in the data parallel setting
    "APHRODITE_DP_MASTER_IP":
    lambda: os.getenv("APHRODITE_DP_MASTER_IP", "127.0.0.1"),

    # Port of the master node in the data parallel setting
    "APHRODITE_DP_MASTER_PORT":
    lambda: int(os.getenv("APHRODITE_DP_MASTER_PORT", "0")),

    # In the context of executing MoE models with Data-Parallel, Expert-Parallel
    # and Batched All-to-All dispatch/combine kernels, APHRODITE_MOE_DP_CHUNK_SIZE
    # dictates the quantum of tokens that can be dispatched from a DP
    # rank. All DP ranks process the activations in APHRODITE_MOE_DP_CHUNK_SIZE
    # units.
    "APHRODITE_MOE_DP_CHUNK_SIZE":
    lambda: int(os.getenv("APHRODITE_MOE_DP_CHUNK_SIZE", "256")),

    # Randomize inputs during dummy runs when using Data Parallel
    "APHRODITE_RANDOMIZE_DP_DUMMY_INPUTS":
    lambda: os.environ.get("APHRODITE_RANDOMIZE_DP_DUMMY_INPUTS", "0") == "1",

    # Whether to use S3 path for model loading in CI via RunAI Streamer
    "APHRODITE_CI_USE_S3":
    lambda: os.environ.get("APHRODITE_CI_USE_S3", "0") == "1",

    # Use model_redirect to redirect the model name to a local folder.
    # `model_redirect` can be a json file mapping the model between
    # repo_id and local folder:
    # {"meta-llama/Llama-3.2-1B": "/tmp/Llama-3.2-1B"}
    # or a space separated values table file:
    # meta-llama/Llama-3.2-1B   /tmp/Llama-3.2-1B
    "APHRODITE_MODEL_REDIRECT_PATH":
    lambda: os.environ.get("APHRODITE_MODEL_REDIRECT_PATH", None),

    # Whether to use atomicAdd reduce in gptq/awq marlin kernel.
    "APHRODITE_MARLIN_USE_ATOMIC_ADD":
    lambda: os.environ.get("APHRODITE_MARLIN_USE_ATOMIC_ADD", "0") == "1",

    # Whether to turn on the outlines cache for V0
    # This cache is unbounded and on disk, so it's not safe to use in
    # an environment with potentially malicious users.
    "APHRODITE_V0_USE_OUTLINES_CACHE":
    lambda: os.environ.get("APHRODITE_V0_USE_OUTLINES_CACHE", "0") == "1",

    # Whether to turn on the outlines cache for V1
    # This cache is unbounded and on disk, so it's not safe to use in
    # an environment with potentially malicious users.
    "APHRODITE_V1_USE_OUTLINES_CACHE":
    lambda: os.environ.get("APHRODITE_V1_USE_OUTLINES_CACHE", "0") == "1",

    # Gap between padding buckets for the forward pass. So we have
    # 8, we will run forward pass with [16, 24, 32, ...].
    "APHRODITE_TPU_BUCKET_PADDING_GAP":
    lambda: int(os.environ["APHRODITE_TPU_BUCKET_PADDING_GAP"])
    if "APHRODITE_TPU_BUCKET_PADDING_GAP" in os.environ else 0,
    "APHRODITE_TPU_MOST_MODEL_LEN":
    lambda: maybe_convert_int(os.environ.get("APHRODITE_TPU_MOST_MODEL_LEN", None)),

    # Whether using Pathways
    "APHRODITE_TPU_USING_PATHWAYS":
    lambda: bool("proxy" in os.getenv("JAX_PLATFORMS", "").lower()),

    # Allow use of DeepGemm kernels for fused moe ops.
    "APHRODITE_USE_DEEP_GEMM":
    lambda: bool(int(os.getenv("APHRODITE_USE_DEEP_GEMM", "0"))),

    # DeepGemm JITs the kernels on-demand. The warmup attempts to make DeepGemm
    # JIT all the required kernels before model execution so there is no
    # JIT'ing in the hot-path. However, this warmup increases the engine
    # startup time by a couple of minutes.
    # Set `APHRODITE_SKIP_DEEP_GEMM_WARMUP` to disable the warmup.
    "APHRODITE_SKIP_DEEP_GEMM_WARMUP":
    lambda: bool(int(os.getenv("APHRODITE_SKIP_DEEP_GEMM_WARMUP", "0"))),

    # Allow use of FlashInfer MoE kernels for fused moe ops.
    "APHRODITE_USE_FLASHINFER_MOE_FP8":
    lambda: bool(int(os.getenv("APHRODITE_USE_FLASHINFER_MOE_FP8", "0"))),

    # Allow use of FlashInfer CUTLASS kernels for fused moe ops.
    "APHRODITE_USE_FLASHINFER_MOE_FP4":
    lambda: bool(int(os.getenv("APHRODITE_USE_FLASHINFER_MOE_FP4", "0"))),

    # Control the cache sized used by the xgrammar compiler. The default
    # of 512 MB should be enough for roughly 1000 JSON schemas.
    # It can be changed with this variable if needed for some reason.
    "APHRODITE_XGRAMMAR_CACHE_MB":
    lambda: int(os.getenv("APHRODITE_XGRAMMAR_CACHE_MB", "512")),

    # Control the threshold for msgspec to use 'zero copy' for
    # serialization/deserialization of tensors. Tensors below
    # this limit will be encoded into the msgpack buffer, and
    # tensors above will instead be sent via a separate message.
    # While the sending side still actually copies the tensor
    # in all cases, on the receiving side, tensors above this
    # limit will actually be zero-copy decoded.
    "APHRODITE_MSGPACK_ZERO_COPY_THRESHOLD":
    lambda: int(os.getenv("APHRODITE_MSGPACK_ZERO_COPY_THRESHOLD", "256")),

    # If set, allow insecure serialization using pickle.
    # This is useful for environments where it is deemed safe to use the
    # insecure method and it is needed for some reason.
    "APHRODITE_ALLOW_INSECURE_SERIALIZATION":
    lambda: bool(int(os.getenv("APHRODITE_ALLOW_INSECURE_SERIALIZATION", "0"))),

    # IP address used for NIXL handshake between remote agents.
    "APHRODITE_NIXL_SIDE_CHANNEL_HOST":
    lambda: os.getenv("APHRODITE_NIXL_SIDE_CHANNEL_HOST", "localhost"),

    # Port used for NIXL handshake between remote agents.
    "APHRODITE_NIXL_SIDE_CHANNEL_PORT":
    lambda: int(os.getenv("APHRODITE_NIXL_SIDE_CHANNEL_PORT", "5557")),

    # all2all backend for aphrodite's expert parallel communication
    # Available options:
    # - "naive": naive all2all implementation using all-reduce
    # - "pplx": use pplx kernels
    # - "deepep_high_throughput", use deepep high-throughput kernels
    # - "deepep_low_latency", use deepep low-latency kernels
    "APHRODITE_ALL2ALL_BACKEND":
    lambda: os.getenv("APHRODITE_ALL2ALL_BACKEND", "naive"),

    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE CUTLASS Kernel. This value is used to create a buffer for
    # the blockscale tensor of activations NVFP4 Quantization.
    # This is used to prevent the kernel from running out of memory.
    "APHRODITE_MAX_TOKENS_PER_EXPERT_FP4_MOE":
    lambda: int(os.getenv("APHRODITE_MAX_TOKENS_PER_EXPERT_FP4_MOE", "163840")),

    # Regex timeout for use by the Aphrodite tool parsing plugins.
    "APHRODITE_TOOL_PARSE_REGEX_TIMEOUT_SECONDS":
    lambda: int(os.getenv("APHRODITE_TOOL_PARSE_REGEX_TIMEOUT_SECONDS", "1")),

    # Reduce CPU usage when Aphrodite is idle. Enabling this will incur small
    # latency penalty when a request eventually comes.
    "APHRODITE_SLEEP_WHEN_IDLE":
    lambda: bool(int(os.getenv("APHRODITE_SLEEP_WHEN_IDLE", "0"))),

    # Control the max chunk bytes (in MB) for the rpc message queue.
    # Object larger than this threshold will be broadcast to worker
    # processes via zmq.
    "APHRODITE_MQ_MAX_CHUNK_BYTES_MB":
    lambda: int(os.getenv("APHRODITE_MQ_MAX_CHUNK_BYTES_MB", "16")),

    # Timeout in seconds for execute_model RPC calls in multiprocessing
    # executor (only applies when TP > 1).
    "APHRODITE_EXECUTE_MODEL_TIMEOUT_SECONDS":
    lambda: int(os.getenv("APHRODITE_EXECUTE_MODEL_TIMEOUT_SECONDS", "300")),

    # KV Cache layout used throughout aphrodite.
    # Some common values are:
    # - NHD
    # - HND
    # Where N=num_blocks, H=num_heads and D=head_size. The default value will
    # leave the layout choice to the backend. Mind that backends may only
    # implement and support a subset of all possible layouts.
    "APHRODITE_KV_CACHE_LAYOUT":
    lambda: os.getenv("APHRODITE_KV_CACHE_LAYOUT", None),

    # Enable checking whether the generated logits contain NaNs,
    # indicating corrupted output. Useful for debugging low level bugs
    # or bad hardware but it may add compute overhead.
    "APHRODITE_COMPUTE_NANS_IN_LOGITS":
    lambda: bool(int(os.getenv("APHRODITE_COMPUTE_NANS_IN_LOGITS", "0"))),

    # Controls whether or not emulations are used for NVFP4
    # generations on machines < 100 for compressed-tensors
    # models
    "APHRODITE_USE_NVFP4_CT_EMULATIONS":
    lambda: bool(int(os.getenv("APHRODITE_USE_NVFP4_CT_EMULATIONS", "0"))),

    # Time (in seconds) after which the KV cache on the producer side is
    # automatically cleared if no READ notification is received from the
    # consumer. This is only applicable when using NixlConnector in a
    # disaggregated decode-prefill setup.
    "APHRODITE_NIXL_ABORT_REQUEST_TIMEOUT":
    lambda: int(os.getenv("APHRODITE_NIXL_ABORT_REQUEST_TIMEOUT", "120")),

    # Controls whether or not to use cudnn prefill
    "APHRODITE_USE_CUDNN_PREFILL":
    lambda: bool(int(os.getenv("APHRODITE_USE_CUDNN_PREFILL", "0"))),

    # If set to 1, use the TRTLLM Context Attention backend in flashinfer.
    "APHRODITE_USE_TRTLLM_CONTEXT_ATTENTION":
    lambda: bool(int(os.getenv("APHRODITE_USE_TRTLLM_CONTEXT_ATTENTION", "0"))),

    # If set to 1, use the TRTLLM Decode Attention backend in flashinfer.
    "APHRODITE_USE_TRTLLM_DECODE_ATTENTION":
    lambda: bool(int(os.getenv("APHRODITE_USE_TRTLLM_DECODE_ATTENTION", "0"))),

    # Controls garbage collection during CUDA graph capture.
    # If set to 0 (default), enables GC freezing to speed up capture time.
    # If set to 1, allows GC to run during capture.
    "APHRODITE_ENABLE_CUDAGRAPH_GC":
    lambda: bool(int(os.getenv("APHRODITE_ENABLE_CUDAGRAPH_GC", "0"))),

    # Used to force set up loopback IP
    "APHRODITE_LOOPBACK_IP":
    lambda: os.getenv("APHRODITE_LOOPBACK_IP", ""),

    # Used to set the process name prefix for Aphrodite processes.
    # This is useful for debugging and monitoring purposes.
    # The default value is "APHRODITE".
    "APHRODITE_PROCESS_NAME_PREFIX":
    lambda: os.getenv("APHRODITE_PROCESS_NAME_PREFIX", "APHRODITE"),

    # Allow chunked local attention with hybrid kv cache manager.
    # Currently using the Hybrid KV cache manager with chunked local attention
    # in the Llama4 models (the only models currently using chunked local attn)
    # causes a latency regression. For this reason, we disable it by default.
    # This flag is used to allow users to enable it if they want to (to save on
    # kv-cache memory usage and enable longer contexts)
    # TODO(lucas): Remove this flag once latency regression is resolved.
    "APHRODITE_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE":
    lambda: bool(int(os.getenv(\
            "APHRODITE_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE", "0"))),

    # Enables support for the "store" option in the OpenAI Responses API.
    # When set to 1, Aphrodite's OpenAI server will retain the input and output
    # messages for those requests in memory. By default, this is disabled (0),
    # and the "store" option is ignored.
    # NOTE/WARNING:
    # 1. Messages are kept in memory only (not persisted to disk) and will be
    #    lost when the Aphrodite server shuts down.
    # 2. Enabling this option will cause a memory leak, as stored messages are
    #    never removed from memory until the server terminates.
    "APHRODITE_ENABLE_RESPONSES_API_STORE":
    lambda: bool(int(os.getenv("APHRODITE_ENABLE_RESPONSES_API_STORE", "0"))),

    # If set, enables the KoboldAI API routes in the API server
    "APHRODITE_KOBOLD_API":
    lambda: bool(int(os.getenv("APHRODITE_KOBOLD_API", "0"))),

    # If set, Aphrodite will use request-level metrics instead of
    # interval-based metrics.
    "APHRODITE_REQUEST_LEVEL_METRICS":
    lambda: bool(int(os.getenv("APHRODITE_REQUEST_LEVEL_METRICS", "0"))),

    # If set, aphrodite will use custom sampling kernels
    "APHRODITE_USE_SAMPLING_KERNELS":
    lambda: bool(int(os.getenv("APHRODITE_USE_SAMPLING_KERNELS", "0"))),
}

# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def set_aphrodite_use_v1(use_v1: bool):
    if is_set("APHRODITE_USE_V1"):
        raise ValueError(
            "Should not call set_aphrodite_use_v1() if APHRODITE_USE_V1 is set "
            "explicitly by the user. Please raise this as a Github "
            "Issue and explicitly set APHRODITE_USE_V1=0 or 1.")
    os.environ["APHRODITE_USE_V1"] = "1" if use_v1 else "0"


def compute_hash() -> str:
    """
    WARNING: Whenever a new key is added to this environment
    variables, ensure that it is included in the factors list if
    it affects the computation graph. For example, different values
    of APHRODITE_PP_LAYER_PARTITION will generate different computation
    graphs, so it is included in the factors list. The env vars that
    affect the choice of different kernels or attention backends should
    also be included in the factors list.
    """
    factors: list[Any] = []

    # summarize environment variables
    def factorize(name: str):
        if __getattr__(name):
            factors.append(__getattr__(name))
        else:
            factors.append("None")

    # The values of envs may affects the computation graph.
    # TODO: hash all environment variables?
    # for key in environment_variables:
    #     factorize(key)
    environment_variables_to_hash = [
        "APHRODITE_PP_LAYER_PARTITION",
        "APHRODITE_MLA_DISABLE",
        "APHRODITE_USE_TRITON_FLASH_ATTN",
        "APHRODITE_USE_TRITON_AWQ",
        "APHRODITE_DP_RANK",
        "APHRODITE_DP_SIZE",
        "APHRODITE_USE_STANDALONE_COMPILE",
        "APHRODITE_FUSED_MOE_CHUNK_SIZE",
    ]
    for key in environment_variables_to_hash:
        if key in environment_variables:
            factorize(key)

    hash_str = hashlib.md5(str(factors).encode(),
                           usedforsecurity=False).hexdigest()

    return hash_str
