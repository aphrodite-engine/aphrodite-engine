from typing import TYPE_CHECKING, Optional

import torch
from loguru import logger

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:  
    from aphrodite.common.config import AphroditeConfig  

  
class MpsPlatform(Platform):  
    _enum = PlatformEnum.MPS
    device_name: str = "mps"  
    device_type: str = "mps"  
    dispatch_key: str = "MPS"  

    @classmethod  
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,  
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],  
                             block_size: int, use_v1: bool,  
                             use_mla: bool) -> str:  
        raise NotImplementedError

    @classmethod  
    def set_device(cls, device: torch.device) -> None:  
        pass  
      
    @classmethod  
    def get_device_name(cls, device_id: int = 0) -> str:  
        return "Apple MPS"  

    @classmethod  
    def get_device_total_memory(cls, device_id: int = 0) -> int:  
        import psutil
        return psutil.virtual_memory().total

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        from aphrodite.common import envs
        if enforce_eager and not envs.APHRODITE_USE_V1:
            logger.warning(
                "To see benefits of async output processing, enable MPS "
                "graph. Since, enforce-eager is enabled, async output "
                "processor cannot be used")
            return False
        return True

    @classmethod
    def inference_mode(cls):
        """A device-specific wrapper of `torch.inference_mode`."""
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, aphrodite_config: "AphroditeConfig") -> None:
        """Check and update the configuration for MPS platform."""
        parallel_config = aphrodite_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "aphrodite.worker.worker.Worker"

        if parallel_config.tensor_parallel_size > 1:
            raise RuntimeError("MPS backend does not support tensor parallelism")
        if parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError("MPS backend does not support pipeline parallelism")

        cache_config = aphrodite_config.cache_config
        if cache_config.block_size is None:
            cache_config.block_size = 16

        compilation_config = aphrodite_config.compilation_config
        compilation_config.use_cudagraph = False

    @classmethod
    def get_current_memory_usage(cls, device: torch.device) -> int:
        """Get current memory usage for MPS device."""
        return torch.mps.current_allocated_memory()
