import time

from aphrodite.config import AphroditeConfig, CompilationConfig, CompilationMode
from aphrodite.logger import init_logger

logger = init_logger(__name__)

context_manager = None
torch_compile_start_time: float = 0.0
dynamo_progress_task = None


def start_monitoring_torch_compile(aphrodite_config: AphroditeConfig):
    global torch_compile_start_time
    torch_compile_start_time = time.time()

    compilation_config: CompilationConfig = aphrodite_config.compilation_config
    path = aphrodite_config.compile_debug_dump_path()
    if compilation_config.mode == CompilationMode.APHRODITE_COMPILE and path:
        import depyf

        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Dumping depyf output to %s", path)
        global context_manager
        context_manager = depyf.prepare_debug(path.as_posix())
        context_manager.__enter__()

    from aphrodite.distributed.parallel_state import is_global_first_rank

    if is_global_first_rank():
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        from aphrodite.utils import get_progress_log_prefix

        global dynamo_progress_task

        log_prefix = get_progress_log_prefix()

        progress = Progress(
            TextColumn(log_prefix),
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
        )
        progress.start()
        dynamo_progress_task = (progress, progress.add_task("Analyzing model for compilation (Dynamo)..."))


def end_monitoring_torch_compile(aphrodite_config: AphroditeConfig):
    compilation_config: CompilationConfig = aphrodite_config.compilation_config
    if compilation_config.mode == CompilationMode.APHRODITE_COMPILE:
        logger.debug_once(
            "torch.compile takes %.2f s in total",
            compilation_config.compilation_time,
            scope="local",
        )
        global context_manager
        if context_manager is not None:
            context_manager.__exit__(None, None, None)
            context_manager = None


cudagraph_capturing_enabled: bool = True


def validate_cudagraph_capturing_enabled():
    # used to monitor whether a cudagraph capturing is legal at runtime.
    # should be called before any cudagraph capturing.
    # if an illegal cudagraph capturing happens, raise an error.
    global cudagraph_capturing_enabled
    if not cudagraph_capturing_enabled:
        raise RuntimeError(
            "CUDA graph capturing detected at an inappropriate time. This operation is currently disabled."
        )


def set_cudagraph_capturing_enabled(enabled: bool):
    global cudagraph_capturing_enabled
    cudagraph_capturing_enabled = enabled
