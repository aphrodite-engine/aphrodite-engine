import os
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    APHRODITE_HOST_IP: str = ""
    APHRODITE_PORT: Optional[int] = None
    APHRODITE_RPC_BASE_PATH: str = tempfile.gettempdir()
    APHRODITE_USE_MODELSCOPE: bool = False
    APHRODITE_RINGBUFFER_WARNING_INTERVAL: int = 60
    APHRODITE_INSTANCE_ID: Optional[str] = None
    APHRODITE_NCCL_SO_PATH: Optional[str] = None
    LD_LIBRARY_PATH: Optional[str] = None
    APHRODITE_USE_TRITON_FLASH_ATTN: bool = False
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    APHRODITE_ENGINE_ITERATION_TIMEOUT_S: int = 60
    APHRODITE_API_KEY: Optional[str] = None
    APHRODITE_ADMIN_KEY: Optional[str] = None
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None
    APHRODITE_CACHE_ROOT: str = os.path.expanduser("~/.cache/aphrodite")
    APHRODITE_CONFIG_ROOT: str = os.path.expanduser("~/.config/aphrodite")
    APHRODITE_CONFIGURE_LOGGING: int = 1
    APHRODITE_LOGGING_LEVEL: str = "INFO"
    APHRODITE_LOGGING_CONFIG_PATH: Optional[str] = None
    APHRODITE_TRACE_FUNCTION: int = 0
    APHRODITE_ATTENTION_BACKEND: Optional[str] = None
    APHRODITE_USE_SAMPLING_KERNELS: bool = False
    APHRODITE_PP_LAYER_PARTITION: Optional[str] = None
    APHRODITE_CPU_KVCACHE_SPACE: int = 0
    APHRODITE_CPU_OMP_THREADS_BIND: str = ""
    APHRODITE_OPENVINO_KVCACHE_SPACE: int = 0
    APHRODITE_OPENVINO_CPU_KV_CACHE_PRECISION: Optional[str] = None
    APHRODITE_OPENVINO_ENABLE_QUANTIZED_WEIGHTS: bool = False
    APHRODITE_XLA_CACHE_PATH: str = os.path.join(APHRODITE_CACHE_ROOT, "xla_cache")  # noqa: E501
    APHRODITE_FUSED_MOE_CHUNK_SIZE: int = 64 * 1024
    APHRODITE_USE_RAY_SPMD_WORKER: bool = False
    APHRODITE_USE_RAY_COMPILED_DAG: bool = False
    APHRODITE_USE_RAY_COMPILED_DAG_NCCL_CHANNEL: bool = True
    APHRODITE_WORKER_MULTIPROC_METHOD: str = "fork"
    APHRODITE_ASSETS_CACHE: str = os.path.join(APHRODITE_CACHE_ROOT, "assets")
    APHRODITE_IMAGE_FETCH_TIMEOUT: int = 5
    APHRODITE_AUDIO_FETCH_TIMEOUT: int = 5
    APHRODITE_TARGET_DEVICE: str = "cuda"
    MAX_JOBS: Optional[str] = None
    NVCC_THREADS: Optional[str] = None
    APHRODITE_USE_PRECOMPILED: bool = False
    APHRODITE_NO_DEPRECATION_WARNING: bool = False
    APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH: bool = False
    CMAKE_BUILD_TYPE: Optional[str] = None
    VERBOSE: bool = False
    APHRODITE_DYNAMIC_ROPE_SCALING: bool = False
    APHRODITE_TEST_FORCE_FP8_MARLIN: bool = False
    APHRODITE_PLUGINS: Optional[List[str]] = None
    APHRODITE_RPC_TIMEOUT: int = 20000
    APHRODITE_FORCE_SINGLE_USER_PREFIX_CACHE: bool = False
    APHRODITE_TEST_DYNAMO_GRAPH_CAPTURE: int = 0
    APHRODITE_TEST_DYNAMO_FULLGRAPH_CAPTURE: int = 0
    APHRODITE_USE_TRITON_AWQ: bool = False
    APHRODITE_DYNAMO_USE_CUSTOM_DISPATCHER: bool = False
    APHRODITE_USE_TRITON_BACKEND: bool = False
    APHRODITE_FORCE_P2P: bool = False
    APHRODITE_TEST_ENABLE_ARTIFICIAL_PREEMPT: bool = False
    APHRODITE_REQUEST_LEVEL_METRICS: bool = False


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


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

environment_variables: Dict[str, Callable[[], Any]] = {

    # ================== Installation Time Env Vars ==================

    # Target device of Aphrodite, supporting [cuda (by default),
    # rocm, neuron, cpu, openvino]
    "APHRODITE_TARGET_DEVICE":
    lambda: os.getenv("APHRODITE_TARGET_DEVICE", "cuda"),

    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),

    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS":
    lambda: os.getenv("NVCC_THREADS", None),

    # If set, Aphrodite will use precompiled binaries (*.so)
    "APHRODITE_USE_PRECOMPILED":
    lambda: bool(os.environ.get("APHRODITE_USE_PRECOMPILED")),

    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),

    # If set, Aphrodite will print verbose logs during installation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),

    # Root directory for APHRODITE configuration files
    # Defaults to `~/.config/aphrodite` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how aphrodite finds its configuration
    # files during runtime, but also affects how aphrodite installs its
    # configuration files during **installation**.
    "APHRODITE_CONFIG_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "APHRODITE_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "aphrodite"),
        )),

    # ================== Runtime Env Vars ==================

    # Root directory for APHRODITE cache files
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
    lambda: os.getenv('APHRODITE_HOST_IP', "") or os.getenv("HOST_IP", ""),

    # used in distributed environment to manually set the communication port
    # Note: if APHRODITE_PORT is set, and some code asks for multiple ports, the
    # APHRODITE_PORT will be used as the first port, and the rest will be
    # generated by incrementing the APHRODITE_PORT value.
    # '0' is used to make mypy happy
    'APHRODITE_PORT':
    lambda: int(os.getenv('APHRODITE_PORT', '0'))
    if 'APHRODITE_PORT' in os.environ else None,

    # path used for ipc when the frontend api server is running in
    # multi-processing mode to communicate with the backend engine process.
    'APHRODITE_RPC_BASE_PATH':
    lambda: os.getenv('APHRODITE_RPC_BASE_PATH', tempfile.gettempdir()),

    # If true, will load models from ModelScope instead of Hugging Face Hub.
    # note that the value is true or false, not numbers
    "APHRODITE_USE_MODELSCOPE":
    lambda: os.environ.get(
        "APHRODITE_USE_MODELSCOPE", "False").lower() == "true",

    # Instance id represents an instance of the APHRODITE. All processes in the
    # same instance should have the same instance id.
    "APHRODITE_INSTANCE_ID":
    lambda: os.environ.get("APHRODITE_INSTANCE_ID", None),

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

    # when `APHRODITE_NCCL_SO_PATH` is not set, aphrodite will try to find the
    # nccl library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH":
    lambda: os.environ.get("LD_LIBRARY_PATH", None),

    # flag to control if aphrodite should use triton flash attention
    "APHRODITE_USE_TRITON_FLASH_ATTN":
    lambda: (os.environ.get(
        "APHRODITE_USE_TRITON_FLASH_ATTN", "True").lower() in ("true", "1")),

    # Internal flag to enable Dynamo graph capture
    "APHRODITE_TEST_DYNAMO_GRAPH_CAPTURE":
    lambda: int(os.environ.get("APHRODITE_TEST_DYNAMO_GRAPH_CAPTURE", "0")),
    "APHRODITE_DYNAMO_USE_CUSTOM_DISPATCHER":
    lambda:
    (os.environ.get("APHRODITE_DYNAMO_USE_CUSTOM_DISPATCHER", "True").lower() in
     ("true", "1")),

    # Internal flag to enable Dynamo fullgraph capture
    "APHRODITE_TEST_DYNAMO_FULLGRAPH_CAPTURE":
    lambda: bool(
        os.environ.get("APHRODITE_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1") != "0"),

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

    # API key for APHRODITE API server
    "APHRODITE_API_KEY":
    lambda: os.environ.get("APHRODITE_API_KEY", None),

    # Admin API key for APHRODITE API server
    "APHRODITE_ADMIN_KEY":
    lambda: os.environ.get("APHRODITE_ADMIN_KEY", None),

    # S3 access information, used for tensorizer to load model from S3
    "S3_ACCESS_KEY_ID":
    lambda: os.environ.get("S3_ACCESS_KEY_ID", None),
    "S3_SECRET_ACCESS_KEY":
    lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),
    "S3_ENDPOINT_URL":
    lambda: os.environ.get("S3_ENDPOINT_URL", None),

    # Logging configuration
    # If set to 0, aphrodite will not configure logging
    # If set to 1, aphrodite will configure logging using the default
    # configuration or the configuration file specified by
    # APHRODITE_LOGGING_CONFIG_PATH
    "APHRODITE_CONFIGURE_LOGGING":
    lambda: int(os.getenv("APHRODITE_CONFIGURE_LOGGING", "1")),
    "APHRODITE_LOGGING_CONFIG_PATH":
    lambda: os.getenv("APHRODITE_LOGGING_CONFIG_PATH"),

    # this is used for configuring the default logging level
    "APHRODITE_LOGGING_LEVEL":
    lambda: os.getenv("APHRODITE_LOGGING_LEVEL", "INFO"),

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
    "APHRODITE_ATTENTION_BACKEND":
    lambda: os.getenv("APHRODITE_ATTENTION_BACKEND", None),

    # If set, aphrodite will use custom sampling kernels
    "APHRODITE_USE_SAMPLING_KERNELS":
    lambda: bool(int(os.getenv("APHRODITE_USE_SAMPLING_KERNELS", "0"))),

    # Pipeline stage partition strategy
    "APHRODITE_PP_LAYER_PARTITION":
    lambda: os.getenv("APHRODITE_PP_LAYER_PARTITION", None),

    # (CPU backend only) CPU key-value cache space.
    # default is 4GB
    "APHRODITE_CPU_KVCACHE_SPACE":
    lambda: int(os.getenv("APHRODITE_CPU_KVCACHE_SPACE", "0")),

    # (CPU backend only) CPU core ids bound by OpenMP threads, e.g., "0-31",
    # "0,1,2", "0-31,33". CPU cores of different ranks are separated by '|'.
    "APHRODITE_CPU_OMP_THREADS_BIND":
    lambda: os.getenv("APHRODITE_CPU_OMP_THREADS_BIND", "all"),

    # OpenVINO key-value cache space
    # default is 4GB
    "APHRODITE_OPENVINO_KVCACHE_SPACE":
    lambda: int(os.getenv("APHRODITE_OPENVINO_KVCACHE_SPACE", "0")),

    # OpenVINO KV cache precision
    # default is bf16 if natively supported by platform, otherwise f16
    # To enable KV cache compression, please, explicitly specify u8
    "APHRODITE_OPENVINO_CPU_KV_CACHE_PRECISION":
    lambda: os.getenv("APHRODITE_OPENVINO_CPU_KV_CACHE_PRECISION", None),

    # Enables weights compression during model export via HF Optimum
    # default is False
    "APHRODITE_OPENVINO_ENABLE_QUANTIZED_WEIGHTS":
    lambda: bool(os.getenv(
        "APHRODITE_OPENVINO_ENABLE_QUANTIZED_WEIGHTS", False)),

    # If the env var is set, then all workers will execute as separate
    # processes from the engine, and we use the same mechanism to trigger
    # execution on all workers.
    # Run aphrodite with APHRODITE_USE_RAY_SPMD_WORKER=1 to enable it.
    "APHRODITE_USE_RAY_SPMD_WORKER":
    lambda: bool(int(os.getenv("APHRODITE_USE_RAY_SPMD_WORKER", "0"))),

    # If the env var is set, it uses the Ray's compiled DAG API
    # which optimizes the control plane overhead.
    # Run aphrodite with APHRODITE_USE_RAY_COMPILED_DAG=1 to enable it.
    "APHRODITE_USE_RAY_COMPILED_DAG":
    lambda: bool(int(os.getenv("APHRODITE_USE_RAY_COMPILED_DAG", "0"))),

    # If the env var is set, it uses NCCL for communication in
    # Ray's compiled DAG. This flag is ignored if
    # APHRODITE_USE_RAY_COMPILED_DAG is not set.
    "APHRODITE_USE_RAY_COMPILED_DAG_NCCL_CHANNEL":
    lambda: bool(int(
        os.getenv("APHRODITE_USE_RAY_COMPILED_DAG_NCCL_CHANNEL", "1"))),

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

    # Timeout for fetching audio when serving multimodal models
    # Default is 5 seconds
    "APHRODITE_AUDIO_FETCH_TIMEOUT":
    lambda: int(os.getenv("APHRODITE_AUDIO_FETCH_TIMEOUT", "5")),

    # Path to the XLA persistent cache directory.
    # Only used for XLA devices such as TPUs.
    "APHRODITE_XLA_CACHE_PATH":
    lambda: os.path.expanduser(
        os.getenv(
            "APHRODITE_XLA_CACHE_PATH",
            os.path.join(get_default_cache_root(), "aphrodite", "xla_cache"),
        )),
    "APHRODITE_FUSED_MOE_CHUNK_SIZE":
    lambda: int(os.getenv("APHRODITE_FUSED_MOE_CHUNK_SIZE", "65536")),

    # If set, aphrodite will skip the deprecation warnings.
    "APHRODITE_NO_DEPRECATION_WARNING":
    lambda: bool(int(os.getenv("APHRODITE_NO_DEPRECATION_WARNING", "0"))),

    # If set, the OpenAI API server will stay alive even after the underlying
    # AsyncLLMEngine errors and stops serving requests
    "APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH":
    lambda: bool(os.getenv("APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH", 0)),

    # If the env var APHRODITE_DYNAMIC_ROPE_SCALING is set, it allows
    # the user to specify a max sequence length greater than
    # the max length derived from the model's config.json.
    # To enable this, set APHRODITE_DYNAMIC_ROPE_SCALING=1.
    "APHRODITE_DYNAMIC_ROPE_SCALING":
    lambda:
    (os.environ.get(
        "APHRODITE_DYNAMIC_ROPE_SCALING",
        "0").strip().lower() in ("1", "true")),

    # If set, forces FP8 Marlin to be used for FP8 quantization regardless
    # of the hardware support for FP8 compute.
    "APHRODITE_TEST_FORCE_FP8_MARLIN":
    lambda:
    (os.environ.get("APHRODITE_TEST_FORCE_FP8_MARLIN", "0").strip().lower() in
     ("1", "true")),

    # Time in ms for the zmq client to wait for a response from the backend
    # server for simple data operations
    "APHRODITE_RPC_TIMEOUT":
    lambda: int(os.getenv("APHRODITE_RPC_TIMEOUT", "20000")),

    # a list of plugin names to load, separated by commas.
    # if this is not set, it means all plugins will be loaded
    # if this is set to an empty string, no plugins will be loaded
    "APHRODITE_PLUGINS":
    lambda: None if "APHRODITE_PLUGINS" not in os.environ else os.environ[
        "APHRODITE_PLUGINS"].split(","),

    # If set, forces prefix cache in single user mode
    "APHRODITE_FORCE_SINGLE_USER_PREFIX_CACHE":
    lambda: bool(int(os.getenv("APHRODITE_FORCE_SINGLE_USER_PREFIX_CACHE",
                               "0"))),

    # If set, Aphrodite will use Triton implementations of AWQ.
    "APHRODITE_USE_TRITON_AWQ":
    lambda: bool(int(os.getenv("APHRODITE_USE_TRITON_AWQ", "0"))),

    # If set, Aphrodite will use Triton implementations of layernorm.
    "APHRODITE_USE_TRITON_BACKEND":
    lambda: bool(int(os.getenv("APHRODITE_USE_TRITON_BACKEND", "0"))),

    # If set, Aphrodite will skip the P2P check and assume that P2P is
    # available. Used for custom all-reduce kernels.
    "APHRODITE_FORCE_P2P":
    lambda: bool(int(os.getenv("APHRODITE_FORCE_P2P", "0"))),

    # If set, Aphrodite will use artificial preemption.
    "APHRODITE_TEST_ENABLE_ARTIFICIAL_PREEMPT":
    lambda: bool(int(
        os.getenv("APHRODITE_TEST_ENABLE_ARTIFICIAL_PREEMPT", "0"))),

    # If set, Aphrodite will use request-level metrics instead of
    # interval-based metrics.
    "APHRODITE_REQUEST_LEVEL_METRICS":
    lambda: bool(int(os.getenv("APHRODITE_REQUEST_LEVEL_METRICS", "0"))),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
