from __future__ import annotations

import contextlib
import multiprocessing
import os
import signal
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TextIO

import psutil

import aphrodite.envs as envs
from aphrodite.logger import init_logger
from aphrodite.ray.lazy_utils import is_in_ray_actor

from .platform_utils import cuda_is_initialized, xpu_is_initialized

logger = init_logger(__name__)

CYAN = "\033[1;36m"
RESET = "\033[0;0m"


# Environment variable utilities


def update_environment_variables(envs_dict: dict[str, str]):
    """Update multiple environment variables with logging."""
    for k, v in envs_dict.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s from '%s' to '%s'",
                k,
                os.environ[k],
                v,
            )
        os.environ[k] = v


@contextlib.contextmanager
def set_env_var(key: str, value: str) -> Iterator[None]:
    """Temporarily set an environment variable."""
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


@contextlib.contextmanager
def suppress_c_lib_output():
    """
    Suppress stdout/stderr from C libraries at the file descriptor level.

    Example:
        with suppress_c_lib_output():
            # C library calls that would normally print to stdout/stderr
            torch.distributed.new_group(ranks, backend="gloo")
    """
    if not envs.APHRODITE_SUPPRESS_C_LIB_OUTPUT:
        yield
        return

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    stdout_dup = os.dup(stdout_fd)
    stderr_dup = os.dup(stderr_fd)

    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_dup, stdout_fd)
        os.dup2(stderr_dup, stderr_fd)
        os.close(stdout_dup)
        os.close(stderr_dup)
        os.close(devnull_fd)


# File path utilities


def unique_filepath(fn: Callable[[int], Path]) -> Path:
    """Generate a unique file path by trying incrementing integers.

    Note: This function has a TOCTOU race condition.
    Caller should use atomic operations (e.g., open with 'x' mode)
    when creating the file to ensure thread safety.
    """
    i = 0
    while True:
        p = fn(i)
        if not p.exists():
            return p
        i += 1


# Process management utilities


def _maybe_force_spawn():
    """Check if we need to force the use of the `spawn` multiprocessing start
    method.
    """
    if os.environ.get("APHRODITE_WORKER_MULTIPROC_METHOD") == "spawn":
        return

    reasons = []
    if is_in_ray_actor():
        # even if we choose to spawn, we need to pass the ray address
        # to the subprocess so that it knows how to connect to the ray cluster.
        # env vars are inherited by subprocesses, even if we use spawn.
        import ray

        os.environ["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
        reasons.append("In a Ray actor and can only be spawned")

    if cuda_is_initialized():
        reasons.append("CUDA is initialized")
    elif xpu_is_initialized():
        reasons.append("XPU is initialized")

    if reasons:
        logger.warning(
            "We must use the `spawn` multiprocessing start method. "
            "Overriding APHRODITE_WORKER_MULTIPROC_METHOD to 'spawn'. "
            "See https://docs.aphrodite.ai/en/latest/usage/"
            "troubleshooting.html#python-multiprocessing "
            "for more information. Reasons: %s",
            "; ".join(reasons),
        )
        os.environ["APHRODITE_WORKER_MULTIPROC_METHOD"] = "spawn"


def get_mp_context():
    """Get a multiprocessing context with a particular method (spawn or fork).
    By default we follow the value of the APHRODITE_WORKER_MULTIPROC_METHOD to
    determine the multiprocessing method (default is fork). However, under
    certain conditions, we may enforce spawn and override the value of
    APHRODITE_WORKER_MULTIPROC_METHOD.
    """
    _maybe_force_spawn()
    mp_method = envs.APHRODITE_WORKER_MULTIPROC_METHOD
    return multiprocessing.get_context(mp_method)


def set_process_title(
    name: str,
    suffix: str = "",
    prefix: str = envs.APHRODITE_PROCESS_NAME_PREFIX,
) -> None:
    """Set the current process title with optional suffix."""
    try:
        import setproctitle
    except ImportError:
        return

    if suffix:
        name = f"{name}_{suffix}"

    setproctitle.setproctitle(f"{prefix}::{name}")


def _simplify_process_name(process_name: str) -> str:
    """Simplify process names to match the desired format.

    Examples:
        EngineCore -> Engine
        EngineCore_DP0 -> Engine (DP0)
        Worker_PP0 -> Worker (PP0)
        Worker_TP1 -> Worker (TP1)
        APIServer -> API
        APIServer_0 -> API (0)
    """
    if process_name.startswith("EngineCore"):
        if "_" in process_name:
            suffix = process_name.split("_", 1)[1]
            return f"Engine ({suffix})"
        return "Engine"

    if process_name.startswith("Worker"):
        if "_" in process_name:
            suffix = process_name.split("_", 1)[1]
            return f"Worker ({suffix})"
        return "Worker"

    if process_name.startswith("APIServer"):
        if "_" in process_name:
            suffix = process_name.split("_", 1)[1]
            return f"API ({suffix})"
        return "API"

    return process_name


def _add_prefix(file: TextIO, worker_name: str, pid: int) -> None:
    """Add colored prefix to file output for log decoration."""
    simplified_name = _simplify_process_name(worker_name)
    prefix = f"{CYAN}({simplified_name}){RESET} "
    file_write = file.write

    def write_with_prefix(s: str):
        if not s:
            return
        if file.start_new_line:  # type: ignore[attr-defined]
            file_write(prefix)
        idx = 0
        while (next_idx := s.find("\n", idx)) != -1:
            next_idx += 1
            file_write(s[idx:next_idx])
            if next_idx == len(s):
                file.start_new_line = True  # type: ignore[attr-defined]
                return
            file_write(prefix)
            idx = next_idx
        file_write(s[idx:])
        file.start_new_line = False  # type: ignore[attr-defined]

    file.start_new_line = True  # type: ignore[attr-defined]
    file.write = write_with_prefix  # type: ignore[method-assign]


def decorate_logs(process_name: str | None = None) -> None:
    """Decorate stdout/stderr with process name and PID prefix."""
    if os.environ.get("APHRODITE_DECORATE_LOGS", "0") not in ("1", "true", "True"):
        return

    if process_name is None:
        process_name = get_mp_context().current_process().name

    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)


def kill_process_tree(pid: int):
    """
    Kills all descendant processes of the given pid by sending SIGKILL.

    Args:
        pid (int): Process ID of the parent process
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


# Resource utilities


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L630
def set_ulimit(target_soft_limit: int = 65535):
    if sys.platform.startswith("win"):
        logger.info("Windows detected, skipping ulimit adjustment.")
        return

    import resource

    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase "
                "with error %s. This can cause fd limit errors like "
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n",
                current_soft,
                e,
            )


def get_kernels_install_command() -> str:
    """Get the install command for aphrodite-kernels based on the current environment."""
    base_url = "https://downloads.pygmalion.chat/whl"
    install_page = "https://aphrodite.pygmalion.chat/installation/installation/"

    try:
        import torch

        # Check for CPU build
        if torch.version.cuda is None and torch.version.hip is None:
            extra_index_url = f"{base_url}/cpu"
            return f"pip install --extra-index-url {extra_index_url} aphrodite-kernels==0.0.1"

        # Check for ROCm
        if torch.version.hip is not None:
            extra_index_url = f"{base_url}/rocm"
            return f"pip install --extra-index-url {extra_index_url} aphrodite-kernels==0.0.1"

        # Check for CUDA
        if torch.version.cuda:
            cuda_major, cuda_minor = torch.version.cuda.split(".")
            cuda_version_str = f"{cuda_major}{cuda_minor}"

            # Support CUDA 12.8 and 12.9
            if cuda_version_str == "129":
                extra_index_url = f"{base_url}/cu129"
                return f"pip install --extra-index-url {extra_index_url} aphrodite-kernels==0.0.1"
            elif cuda_version_str == "128":
                # Default (12.8) uses base URL without /cu128 suffix
                extra_index_url = base_url
                return f"pip install --extra-index-url {extra_index_url} aphrodite-kernels==0.0.1"

            # Try to detect from nvcc if available
            try:
                from torch.utils.cpp_extension import CUDA_HOME

                if CUDA_HOME:
                    import subprocess

                    nvcc_output = subprocess.check_output(
                        [f"{CUDA_HOME}/bin/nvcc", "-V"], universal_newlines=True, stderr=subprocess.DEVNULL
                    )
                    import regex as re

                    version_match = re.search(r"release (\d+\.\d+)", nvcc_output)
                    if version_match:
                        nvcc_version = version_match.group(1)
                        nvcc_major, nvcc_minor = nvcc_version.split(".")
                        nvcc_version_str = f"{nvcc_major}{nvcc_minor}"
                        if nvcc_version_str == "129":
                            extra_index_url = f"{base_url}/cu129"
                            return f"pip install --extra-index-url {extra_index_url} aphrodite-kernels==0.0.1"
                        elif nvcc_version_str == "128":
                            extra_index_url = base_url
                            return f"pip install --extra-index-url {extra_index_url} aphrodite-kernels==0.0.1"
            except Exception:
                pass

            # Unsupported CUDA version - direct to build from source
            return (
                f"Your CUDA version ({torch.version.cuda}) is not supported by pre-built wheels.\n"
                f"Please build aphrodite-kernels from source. See: {install_page}"
            )
    except ImportError:
        pass

    # Fallback - default to 12.8
    extra_index_url = base_url
    return f"pip install --extra-index-url {extra_index_url} aphrodite-kernels==0.0.1"
