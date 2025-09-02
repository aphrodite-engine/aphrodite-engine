import os
import time

from aphrodite.config import (AphroditeConfig, CompilationConfig,
                                     CompilationLevel)

context_manager = None
torch_compile_start_time: float = 0.0


def start_monitoring_torch_compile(aphrodite_config: AphroditeConfig):
    global torch_compile_start_time
    torch_compile_start_time = time.time()

    compilation_config: CompilationConfig = aphrodite_config.compilation_config
    if compilation_config.level == CompilationLevel.PIECEWISE and \
        compilation_config.debug_dump_path:
        import depyf
        path = os.path.join(compilation_config.debug_dump_path,
                            f"rank_{aphrodite_config.parallel_config.rank}")
        global context_manager
        context_manager = depyf.prepare_debug(path)
        context_manager.__enter__()


def end_monitoring_torch_compile(aphrodite_config: AphroditeConfig):
    compilation_config: CompilationConfig = aphrodite_config.compilation_config
    if compilation_config.level == CompilationLevel.PIECEWISE:
        global context_manager
        if context_manager is not None:
            context_manager.__exit__(None, None, None)
            context_manager = None


cudagraph_capturing_enabled: bool = True


def validate_cudagraph_capturing_enabled():
    # used to monitor whether an cudagraph capturing is legal at runtime.
    # should be called before any cudagraph capturing.
    # if an illegal cudagraph capturing happens, raise an error.
    global cudagraph_capturing_enabled
    if not cudagraph_capturing_enabled:
        raise RuntimeError("CUDA graph capturing detected at an inappropriate "
                           "time. This operation is currently disabled.")


def set_cudagraph_capturing_enabled(enabled: bool):
    global cudagraph_capturing_enabled
    cudagraph_capturing_enabled = enabled
