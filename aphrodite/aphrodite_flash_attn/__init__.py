__version__ = "2.7.2.post1"

# Use relative import to support build-from-source installation in Aphrodite
from .flash_attn_interface import (fa_version_unsupported_reason,
                                   flash_attn_varlen_func,
                                   flash_attn_with_kvcache,
                                   get_scheduler_metadata,
                                   is_fa_version_supported, sparse_attn_func,
                                   sparse_attn_varlen_func)

__all__ = [
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "get_scheduler_metadata",
    "is_fa_version_supported",
    "sparse_attn_func",
    "sparse_attn_varlen_func",
]
