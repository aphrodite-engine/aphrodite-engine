import enum
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, Optional, Type

import torch
from loguru import logger

import aphrodite.common.envs as envs
from aphrodite.attention.backends.abstract import AttentionBackend
from aphrodite.common.utils import (STR_BACKEND_ENV_VAR, is_cpu, is_hip,
                                    is_openvino, is_xpu)
from aphrodite.platforms import current_platform

APHRODITE_ATTENTION_BACKEND = envs.APHRODITE_ATTENTION_BACKEND


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    FLASH_ATTN_APHRODITE_V1 = enum.auto()
    XFORMERS = enum.auto()
    TRITON_FLASH = enum.auto()
    TORCH_SDPA = enum.auto()
    OPENVINO = enum.auto()
    FLASHINFER = enum.auto()
    PALLAS = enum.auto()
    IPEX = enum.auto()
    NO_ATTENTION = enum.auto()


def backend_name_to_enum(backend_name: str) -> _Backend:
    assert backend_name is not None

    backend_members = _Backend.__members__
    if backend_name not in backend_members:
        raise ValueError(f"Invalid attention backend '{backend_name}'. "
                         f"Available backends: {', '.join(backend_members)} "
                         "(case-sensitive).")

    return _Backend[backend_name]


def get_env_variable_attn_backend() -> Optional[_Backend]:
    '''
    Get the backend override specified by the Aphrodite attention
    backend environment variable, if one is specified.

    Returns:

    * _Backend enum value if an override is specified
    * None otherwise
    '''
    backend_name = os.environ.get(STR_BACKEND_ENV_VAR)
    return (None
            if backend_name is None else backend_name_to_enum(backend_name))


# Global state allows a particular choice of backend
# to be forced, overriding the logic which auto-selects
# a backend based on system & workload configuration
# (default behavior if this variable is None)
#
# THIS SELECTION TAKES PRECEDENCE OVER THE
# APHRODITE ATTENTION BACKEND ENVIRONMENT VARIABLE
forced_attn_backend: Optional[_Backend] = None


def global_force_attn_backend(attn_backend: Optional[_Backend]) -> None:
    '''
    Force all attention operations to use a specified backend.

    Passing `None` for the argument re-enables automatic
    backend selection.,

    Arguments:

    * attn_backend: backend selection (None to revert to auto)
    '''
    global forced_attn_backend
    forced_attn_backend = attn_backend


def get_global_forced_attn_backend() -> Optional[_Backend]:
    '''
    Get the currently-forced choice of attention backend,
    or None if auto-selection is currently enabled.
    '''
    return forced_attn_backend


@lru_cache(maxsize=None)
def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    is_attention_free: bool,
    is_blocksparse: bool = False,
) -> Type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    if is_blocksparse:
        logger.info("Using BlocksparseFlashAttention backend.")
        from aphrodite.attention.backends.blocksparse_attn import (
            BlocksparseFlashAttentionBackend)
        return BlocksparseFlashAttentionBackend

    backend = which_attn_to_use(head_size, dtype, kv_cache_dtype, block_size,
                                is_attention_free)
    if backend == _Backend.FLASH_ATTN:
        from aphrodite.attention.backends.flash_attn import (  # noqa: F401
            FlashAttentionBackend)
        return FlashAttentionBackend
    elif backend == _Backend.FLASH_ATTN_APHRODITE_V1:
        from aphrodite.v1.attention.backends.flash_attn import (  # noqa: F401
            FlashAttentionBackend)
        return FlashAttentionBackend
    if backend == _Backend.XFORMERS:
        logger.info("Using XFormers backend.")
        from aphrodite.attention.backends.xformers import (  # noqa: F401
            XFormersBackend)
        return XFormersBackend
    elif backend == _Backend.TRITON_FLASH:
        logger.info("Using TritonFlashAttention backend.")
        from aphrodite.attention.backends.triton_flash_attn import (  # noqa: F401
            TritonFlashAttentionBackend)
        return TritonFlashAttentionBackend
    elif backend == _Backend.TORCH_SDPA:
        assert is_cpu(), RuntimeError(
            "Torch SDPA backend is only used for the CPU device.")
        logger.info("Using Torch SDPA backend.")
        from aphrodite.attention.backends.torch_sdpa import TorchSDPABackend
        return TorchSDPABackend
    elif backend == _Backend.OPENVINO:
        logger.info("Using OpenVINO Attention backend.")
        from aphrodite.attention.backends.openvino import (
            OpenVINOAttentionBackend)
        return OpenVINOAttentionBackend
    elif backend == _Backend.IPEX:
        assert is_xpu(), RuntimeError(
            "IPEX attention backend is only used for the XPU device.")
        logger.info("Using IPEX attention backend.")
        from aphrodite.attention.backends.ipex_attn import IpexAttnBackend
        return IpexAttnBackend
    elif backend == _Backend.FLASHINFER:
        logger.info("Using Flashinfer backend.")
        from aphrodite.attention.backends.flashinfer import FlashInferBackend
        return FlashInferBackend
    elif backend == _Backend.PALLAS:
        logger.info("Using Pallas backend.")
        from aphrodite.attention.backends.pallas import PallasAttentionBackend
        return PallasAttentionBackend
    elif backend == _Backend.NO_ATTENTION:
        from aphrodite.attention.backends.placeholder_attn import (
            PlaceholderAttentionBackend)
        return PlaceholderAttentionBackend
    else:
        raise ValueError("Invalid attention backend.")


def which_attn_to_use(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    is_attention_free: bool,
) -> _Backend:
    """Returns which flash attention backend to use."""
    # Default case.
    selected_backend = _Backend.FLASH_ATTN

    # If there are no attention layers (e.g. we are running Mamba),
    # use the placeholder NO_ATTENTION
    if is_attention_free:
        return _Backend.NO_ATTENTION

    # Check whether a particular choice of backend was
    # previously forced.
    #
    # THIS SELECTION OVERRIDES THE APHRODITE_ATTENTION_BACKEND
    # ENVIRONMENT VARIABLE.
    backend_by_global_setting: Optional[_Backend] = (
        get_global_forced_attn_backend())
    if backend_by_global_setting is not None:
        selected_backend = backend_by_global_setting
    else:
        # Check the environment variable and override if specified
        backend_by_env_var: Optional[str] = APHRODITE_ATTENTION_BACKEND
        if backend_by_env_var is not None:
            selected_backend = backend_name_to_enum(backend_by_env_var)

    if is_cpu():
        if selected_backend != _Backend.TORCH_SDPA:
            logger.info(f"Cannot use {selected_backend} backend on CPU.")
        return _Backend.TORCH_SDPA

    if is_openvino():
        if selected_backend != _Backend.OPENVINO:
            logger.info(f"Cannot use {selected_backend} backend on OpenVINO.")
        return _Backend.OPENVINO

    if is_xpu():
        if selected_backend != _Backend.IPEX:
            logger.info(f"Cannot use {selected_backend} backend on XPU.")
        return _Backend.IPEX

    if current_platform.is_tpu():
        if selected_backend != _Backend.PALLAS:
            logger.info(f"Cannot use {selected_backend} backend on TPU.")
        return _Backend.PALLAS

    if is_hip():
        # AMD GPUs.
        selected_backend = (_Backend.TRITON_FLASH if selected_backend
                            == _Backend.FLASH_ATTN else selected_backend)
        if selected_backend == _Backend.TRITON_FLASH:
            if current_platform.get_device_capability()[0] != 9:
                # not Instinct series GPUs.
                logger.info("flash_attn is not supported on NAVI GPUs.")
        else:
            logger.info(f"{selected_backend} is not supported in AMD GPUs.")
        return _Backend.TRITON_FLASH

    if envs.APHRODITE_USE_V1:
        return _Backend.FLASH_ATTN_APHRODITE_V1

    # FlashAttn in NVIDIA GPUs.
    if selected_backend == _Backend.FLASH_ATTN:
        if current_platform.get_device_capability()[0] < 8:
            # Volta and Turing NVIDIA GPUs.
            logger.info(
                "Cannot use FlashAttention-2 backend for Volta and Turing "
                "GPUs.")
            selected_backend = _Backend.XFORMERS
        elif dtype not in (torch.float16, torch.bfloat16):
            logger.info(
                "Cannot use FlashAttention-2 backend for dtype other than "
                "torch.float16 or torch.bfloat16.")
            selected_backend = _Backend.XFORMERS
        elif kv_cache_dtype is not None and kv_cache_dtype.startswith("fp8"):
            logger.info(
                "Cannot use FlashAttention-2 backend for FP8 KV cache.")
            logger.warning(
                "Please use FlashInfer backend with FP8 KV Cache for "
                "better performance by setting the environment "
                "variable APHRODITE_ATTENTION_BACKEND=FLASHINFER")
            selected_backend = _Backend.XFORMERS
        elif block_size % 16 != 0:
            logger.info(
                "Cannot use FlashAttention-2 backend for block size not "
                "divisible by 16.")
            selected_backend = _Backend.XFORMERS

    # FlashAttn is valid for the model, checking if the package is installed.
    if selected_backend == _Backend.FLASH_ATTN:
        try:
            import aphrodite.attention.ops.aphrodite_flash_attn  # noqa: F401
            from aphrodite.attention.backends.flash_attn import (  # noqa: F401
                FlashAttentionBackend)

            supported_sizes = FlashAttentionBackend.get_supported_head_sizes()
            if head_size not in supported_sizes:
                logger.info(
                    "Cannot use FlashAttention-2 backend for head size "
                    f"{head_size}")
                selected_backend = _Backend.XFORMERS
        except ImportError:
            logger.info(
                "Cannot use FlashAttention-2 backend because the "
                "aphrodite._aphrodite_flash_attn_C object is not found. "
                "This is built by default on supported hardware.")
            selected_backend = _Backend.XFORMERS

    return selected_backend


@contextmanager
def global_force_attn_backend_context_manager(
        attn_backend: _Backend) -> Generator[None, None, None]:
    '''
    Globally force a Aphrodite attention backend override within a
    context manager, reverting the global attention backend
    override to its prior state upon exiting the context
    manager.

    Arguments:

    * attn_backend: attention backend to force

    Returns:

    * Generator
    '''

    # Save the current state of the global backend override (if any)
    original_value = get_global_forced_attn_backend()

    # Globally force the new backend override
    global_force_attn_backend(attn_backend)

    # Yield control back to the enclosed code block
    try:
        yield
    finally:
        # Revert the original global backend override, if any
        global_force_attn_backend(original_value)
