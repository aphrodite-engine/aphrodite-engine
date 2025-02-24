from aphrodite.common.logger import log_once
from aphrodite.platforms import current_platform

from .punica_base import PunicaWrapperBase


def get_punica_wrapper(*args, **kwargs) -> PunicaWrapperBase:
    if current_platform.is_cuda():
        # Lazy import to avoid ImportError
        from aphrodite.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
        log_once(level="INFO", message="Using PunicaWrapperGPU.")
        return PunicaWrapperGPU(*args, **kwargs)
    else:
        raise NotImplementedError
