import torch

from .interface import CpuArchEnum, Platform, PlatformEnum, UnspecifiedPlatform

current_platform: Platform

try:
    import libtpu
except ImportError:
    libtpu = None

is_xpu = False
try:
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        is_xpu = True
except Exception:
    pass

is_cpu = False
try:
    from importlib.metadata import version
    is_cpu = "cpu" in version("aphrodite-engine")
except Exception:
    pass

if libtpu is not None:
    # people might install pytorch built with cuda but run on tpu
    # so we need to check tpu first
    from .tpu import TpuPlatform
    current_platform = TpuPlatform()
elif torch.version.cuda is not None:
    from .cuda import CudaPlatform
    current_platform = CudaPlatform()
elif torch.version.hip is not None:
    from .rocm import RocmPlatform
    current_platform = RocmPlatform()
elif is_cpu:
    from .cpu import CpuPlatform
    current_platform = CpuPlatform()
elif is_xpu:
    from .xpu import XPUPlatform
    current_platform = XPUPlatform()
else:
    current_platform = UnspecifiedPlatform()

__all__ = ['Platform', 'PlatformEnum', 'current_platform', 'CpuArchEnum']
