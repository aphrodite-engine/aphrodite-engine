from typing import TYPE_CHECKING, Optional, Union

import torch
from loguru import logger

import aphrodite.common.envs as envs
from aphrodite.common.sampling_params import SamplingParams, SamplingType
from aphrodite.inputs import ProcessorInputs, PromptType

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from aphrodite.common.config import AphroditeConfig, ModelConfig
    from aphrodite.common.pooling_params import PoolingParams
else:
    ModelConfig = None
    AphroditeConfig = None
    PoolingParams = None



class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "XLA"
    ray_device_key: str = "TPU"
    device_control_env_var: str = "TPU_VISIBLE_CHIPS"

    supported_quantization: list[str] = ["tpu_int8", "compressed-tensors"]

    additional_env_vars: list[str] = [
        "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS"
    ]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        if (selected_backend != _Backend.PALLAS
                and selected_backend != _Backend.PALLAS_APHRODITE_V1):
            logger.info("Cannot use {} backend on TPU.", selected_backend)

        if use_v1:
            logger.info("Using Pallas V1 backend.")
            return "aphrodite.v1.attention.backends.pallas.PallasAttentionBackend"  # noqa
        else:
            logger.info("Using Pallas backend.")
            return "aphrodite.attention.backends.pallas.PallasAttentionBackend"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "tpu"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return not envs.APHRODITE_USE_V1

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, aphrodite_config: AphroditeConfig) -> None:
        from aphrodite.common.config import CompilationLevel

        cache_config = aphrodite_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        compilation_config = aphrodite_config.compilation_config

        # TPU only supports DYNAMO_ONCE compilation level
        if compilation_config.level != CompilationLevel.DYNAMO_ONCE:
            logger.info("[TPU] Forcing DYNAMO_ONCE compilation level")
            compilation_config.level = CompilationLevel.DYNAMO_ONCE

        if compilation_config.backend == "":
            compilation_config.backend = "openxla"

        assert aphrodite_config.speculative_config is None, \
            "TPU does not support speculative decoding"

        if aphrodite_config.model_config.dtype in (torch.float16,
                                                   torch.float32):
            logger.warning(
                "The TPU backend currently does not support {}. "
                "Using bfloat16 instead.", aphrodite_config.model_config.dtype)
            aphrodite_config.model_config.dtype = torch.bfloat16

        if envs.APHRODITE_USE_V1:
            from aphrodite.v1.attention.backends.pallas import (
                PallasAttentionBackend)
            min_page_size = PallasAttentionBackend.get_min_page_size(
                aphrodite_config)
            if min_page_size > aphrodite_config.cache_config.block_size:
                logger.warning(
                    "Increase the page size from {} to {} to make sure there's"
                    "no SMEM OOM",
                    aphrodite_config.cache_config.block_size,
                    min_page_size,
                )
                aphrodite_config.cache_config.block_size = min_page_size

        parallel_config = aphrodite_config.parallel_config
        scheduler_config = aphrodite_config.scheduler_config
        if parallel_config.worker_cls == "auto":
            if scheduler_config.is_multi_step:
                if envs.APHRODITE_USE_V1:
                    raise NotImplementedError(
                        "Multi-step scheduling is not supported (and not "
                        "needed) on Aphrodite V1. Please launch without "
                        "--num-scheduler-steps.")
                else:
                    parallel_config.worker_cls = \
                        "aphrodite.worker.multi_step_tpu_worker.MultiStepTPUWorker"
            else:
                if envs.APHRODITE_USE_V1:
                    parallel_config.worker_cls = \
                        "aphrodite.v1.worker.tpu_worker.TPUWorker"
                else:
                    parallel_config.worker_cls = \
                        "aphrodite.worker.tpu_worker.TPUWorker"

        assert not aphrodite_config.speculative_config, (
            "Speculative decoding is not yet supported for TPU backend")

        if scheduler_config.is_multimodal_model and not \
            scheduler_config.disable_chunked_mm_input:
            logger.warning("TPU does not support running Multimodal models"\
            " without setting `--disable_chunked_mm_input`. " \
            "Forcing --disable_chunked_mm_input.")
            scheduler_config.disable_chunked_mm_input = True

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on TPU.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "aphrodite.distributed.device_communicators.tpu_communicator.TpuCommunicator"  # noqa

    @classmethod
    def use_all_gather(cls) -> bool:
        return True

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        # V1 support on TPU is experimental
        return True

    @classmethod
    def validate_request(
        cls,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        processed_inputs: ProcessorInputs,
    ) -> None:
        """Raises if this request is unsupported on this platform"""
        if isinstance(params, SamplingParams):
            if params.guided_decoding is not None and not envs.APHRODITE_USE_V1:
                raise ValueError("Structured output is not supported on "
                                 f"{cls.device_name} V0.")
            if params.sampling_type == SamplingType.RANDOM_SEED:
                raise ValueError(
                    "Torch XLA does not support per-request seed.")
