from typing import Optional

from loguru import logger

from aphrodite.common import envs
from aphrodite.common.logger import log_once


def get_flash_attn_version(requires_alibi: bool = False) -> Optional[int]:
    # import here to avoid circular dependencies
    from aphrodite.platforms import current_platform
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
            log_once("WARNING",
                "Cannot use FA version 3 on Blackwell platform "
                "defaulting to FA version 2.")
            fa_version = 2

        if requires_alibi and fa_version == 3:
            log_once("WARNING", "Cannot use FA version 3 with ALiBi, "
                                "defaulting to FA version 2.")
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error("Cannot use FA version {} is not supported due to {}",
                         fa_version, fa_version_unsupported_reason(fa_version))

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


def flash_attn_supports_fp8() -> bool:
    from aphrodite.platforms import current_platform
    return get_flash_attn_version() == 3 and \
        current_platform.get_device_capability().major == 9
