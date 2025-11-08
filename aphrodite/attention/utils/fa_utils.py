from aphrodite import envs
from aphrodite.logger import init_logger
from aphrodite.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_cuda():
    from aphrodite import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    from aphrodite_kernels.aphrodite_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
elif current_platform.is_xpu():
    from aphrodite._ipex_ops import ipex_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    flash_attn_varlen_func = ops.flash_attn_varlen_func
    get_scheduler_metadata = ops.get_scheduler_metadata


def get_flash_attn_version(requires_alibi: bool = False) -> int | None:
    # import here to avoid circular dependencies
    from aphrodite.platforms import current_platform

    if current_platform.is_xpu():
        return 2
    try:
        from aphrodite_kernels.aphrodite_flash_attn import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        fa_version = 3 if (device_capability.major == 9 and is_fa_version_supported(3)) else 2

        # 2. override if passed by environment
        env_version = envs.APHRODITE_FLASH_ATTN_VERSION
        if env_version is not None:
            assert env_version in [2, 3], f"APHRODITE_FLASH_ATTN_VERSION must be 2 or 3, got {env_version}"
            fa_version = env_version

        # 3. fallback for unsupported combinations
        if device_capability.major == 10 and fa_version == 3:
            logger.warning_once("Cannot use FA version 3 on Blackwell platform defaulting to FA version 2.")
            fa_version = 2

        if requires_alibi and fa_version == 3:
            logger.warning_once("Cannot use FA version 3 with ALiBi, defaulting to FA version 2.")
            fa_version = 2

        # Check if version is supported, but respect environment variable override
        if not is_fa_version_supported(fa_version):
            if env_version is not None:
                # User explicitly set version, but it's not supported - log warning but still return it
                logger.warning_once(
                    "FA version %d was explicitly requested via APHRODITE_FLASH_ATTN_VERSION=%d, "
                    "but it is not supported on this device: %s. "
                    "This may cause runtime errors.",
                    fa_version,
                    env_version,
                    fa_version_unsupported_reason(fa_version),
                    scope="global",
                )
                return fa_version
            else:
                # Version not supported and not explicitly requested - return None
                logger.error(
                    "Cannot use FA version %d is not supported due to %s",
                    fa_version,
                    fa_version_unsupported_reason(fa_version),
                )
                return None

        return fa_version
    except ImportError:
        # If import fails, we can't determine version
        return None
    except AssertionError as e:
        # If assertion fails (e.g., device_capability is None), return None
        # But if it's about env_version validation, re-raise
        if "APHRODITE_FLASH_ATTN_VERSION" in str(e):
            raise
        return None


def flash_attn_supports_fp8() -> bool:
    return get_flash_attn_version() == 3 and current_platform.get_device_capability().major == 9


def flash_attn_supports_mla():
    from aphrodite.platforms import current_platform

    if current_platform.is_cuda():
        try:
            from aphrodite_kernels.aphrodite_flash_attn import is_fa_version_supported

            return is_fa_version_supported(3) and current_platform.get_device_capability()[0] == 9
        except (ImportError, AssertionError):
            pass
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return current_platform.is_cuda() or current_platform.is_xpu()
