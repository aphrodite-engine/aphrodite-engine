"""
Compatibility shim for aphrodite._custom_ops.
This module re-exports everything from aphrodite_kernels._custom_ops
to maintain backward compatibility with existing imports.
"""

# Import everything from aphrodite_kernels._custom_ops
# Re-export all public symbols
from aphrodite_kernels._custom_ops import *  # noqa: F403, F401

from aphrodite_kernels import _custom_ops

# Also make the module itself available for `import aphrodite._custom_ops as ops`
__all__ = _custom_ops.__all__ if hasattr(_custom_ops, "__all__") else []
