from typing import Optional

from loguru import logger

from aphrodite.common import envs
from aphrodite.common.logger import log_once
from aphrodite.platforms import current_platform

if current_platform.is_cuda():
    from aphrodite import _custom_ops as ops
    reshape_and_cache_flash = ops.reshape_and_cache_flash
    from aphrodite.aphrodite_flash_attn import (flash_attn_varlen_func,
                                                get_scheduler_metadata)
elif current_platform.is_xpu():
    from aphrodite._ipex_ops import ipex_ops as ops
    reshape_and_cache_flash = ops.reshape_and_cache_flash
    flash_attn_varlen_func = ops.flash_attn_varlen_func
    get_scheduler_metadata = ops.get_scheduler_metadata


def get_flash_attn_version(requires_alibi: bool = False) -> Optional[int]:
    # import here to avoid circular dependencies
    from aphrodite.platforms import current_platform
    if current_platform.is_xpu():
        return 2
    try:
        from aphrodite.aphrodite_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason, is_fa_version_supported)
        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        fa_version = 3 if (device_capability.major == 9
                           and is_fa_version_supported(3)) else 2

        # 2. override if passed by environment
        if envs.APHRODITE_FLASH_ATTN_VERSION is not None:
            assert envs.APHRODITE_FLASH_ATTN_VERSION in [2, 3]
            fa_version = envs.APHRODITE_FLASH_ATTN_VERSION

        # 3. fallback for unsupported combinations
        if device_capability.major == 10 and fa_version == 3:
            log_once(
                "WARNING",
                "Cannot use FA version 3 on Blackwell platform "
                "defaulting to FA version 2.")
            fa_version = 2

        if requires_alibi and fa_version == 3:
            log_once("WARNING", "Cannot use FA version 3 with ALiBi, "
                                "defaulting to FA version 2.")
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            log_once("ERROR", "Cannot use FA version {} is not supported due to {}",
                         fa_version, fa_version_unsupported_reason(fa_version))

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


def flash_attn_supports_fp8() -> bool:
    return get_flash_attn_version() == 3 and \
        current_platform.get_device_capability().major == 9


def flash_attn_supports_mla():
    from aphrodite.platforms import current_platform
    if current_platform.is_cuda():
        try:
            from aphrodite.aphrodite_flash_attn.flash_attn_interface import (
                is_fa_version_supported)
            return is_fa_version_supported(3) \
                and current_platform.get_device_capability()[0] == 9
        except (ImportError, AssertionError):
            pass
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return current_platform.is_cuda() or current_platform.is_xpu()
