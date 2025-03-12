from aphrodite.triton_utils.importing import HAS_TRITON

__all__ = ["HAS_TRITON"]

if HAS_TRITON:

    from aphrodite.triton_utils.custom_cache_manager import (
        maybe_set_triton_cache_manager)
    from aphrodite.triton_utils.libentry import libentry

    __all__ += ["maybe_set_triton_cache_manager", "libentry"]
