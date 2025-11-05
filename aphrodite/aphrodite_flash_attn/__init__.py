"""
Compatibility shim for aphrodite.aphrodite_flash_attn.
This module re-exports everything from aphrodite_kernels.aphrodite_flash_attn
to maintain backward compatibility with existing imports.
"""

from aphrodite_kernels.aphrodite_flash_attn import *  # noqa: F403, F401
