import os
from typing import TYPE_CHECKING, Optional

import torch
from loguru import logger

from aphrodite.common import envs

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from aphrodite.common.config import AphroditeConfig
else:
    AphroditeConfig = None



class HpuPlatform(Platform):
    _enum = PlatformEnum.HPU
    device_name: str = "hpu"
    device_type: str = "hpu"
    dispatch_key: str = "HPU"
    ray_device_key: str = "HPU"
    device_control_env_var: str = "HABANA_VISIBLE_MODULES"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        logger.info("Using HPUAttention backend.")
        return "aphrodite.attention.backends.hpu_attn.HPUAttentionBackend"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True

    @staticmethod
    def inference_mode():
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, aphrodite_config: AphroditeConfig) -> None:

        scheduler_config = aphrodite_config.scheduler_config
        parallel_config = aphrodite_config.parallel_config
        if scheduler_config.is_multi_step:
            parallel_config.worker_cls = \
                "aphrodite.worker.multi_step_hpu_worker.MultiStepHPUWorker"

        if aphrodite_config.speculative_config is not None:
            raise NotImplementedError(
                "Speculative decoding is not implemented for HPU")

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "aphrodite.worker.hpu_worker.HPUWorker"

        # NOTE(kzawora): default block size for Gaudi should be 128
        # smaller sizes still work, but very inefficiently
        cache_config = aphrodite_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 128
        if (parallel_config.distributed_executor_backend == 'mp'
                and envs.APHRODITE_WORKER_MULTIPROC_METHOD == 'fork'):
            if os.environ.get("APHRODITE_WORKER_MULTIPROC_METHOD",
                              None) is not None:
                logger.warning("On HPU, APHRODITE_WORKER_MULTIPROC_METHOD=fork "
                               "might cause application hangs on exit. Using "
                               "APHRODITE_WORKER_MULTIPROC_METHOD=fork anyway, "
                               "as it was explicitly requested.")
            else:
                logger.warning(
                    "On HPU, APHRODITE_WORKER_MULTIPROC_METHOD=fork "
                    "might cause application hangs on exit. Setting "
                    "APHRODITE_WORKER_MULTIPROC_METHOD to 'spawn'. "
                    "To override that behavior, please set "
                    "APHRODITE_WORKER_MULTIPROC_METHOD=fork explicitly.")
                os.environ["APHRODITE_WORKER_MULTIPROC_METHOD"] = "spawn"

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on HPU.")
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "aphrodite.lora.punica_wrapper.punica_hpu.PunicaWrapperHPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "aphrodite.distributed.device_communicators.hpu_communicator.HpuCommunicator"  # noqa
