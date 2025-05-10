from typing import TYPE_CHECKING, Optional

from loguru import logger

from aphrodite import envs

from .interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from aphrodite.common.config import AphroditeConfig
else:
    AphroditeConfig = None



class NeuronPlatform(Platform):
    _enum = PlatformEnum.NEURON
    device_name: str = "neuron"
    device_type: str = "neuron"
    ray_device_key: str = "neuron_cores"
    supported_quantization: list[str] = ["neuron_quant"]
    device_control_env_var: str = "NEURON_RT_VISIBLE_CORES"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "neuron"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def check_and_update_config(cls, aphrodite_config: AphroditeConfig) -> None:
        parallel_config = aphrodite_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = \
                "aphrodite.worker.neuron_worker.NeuronWorker"

        if parallel_config.world_size > 1:
            parallel_config.distributed_executor_backend = "uni"

        assert (aphrodite_config.lora_config
                is None), "LoRA is not supported for Neuron backend."
        assert (not aphrodite_config.speculative_config
                ), "Speculative decoding not yet supported for Neuron backend."

        cache_config = aphrodite_config.cache_config
        if cache_config:
            # neuron needs block_size = max_model_len
            aphrodite_config.cache_config.block_size = \
                aphrodite_config.model_config.max_model_len  # type: ignore

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Neuron.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        if envs.APHRODITE_USE_V1:
            return "aphrodite.distributed.device_communicators.neuron_communicator.NeuronCommunicator"  # noqa
        else:
            return Platform.get_device_communicator_cls()

    @classmethod
    def use_all_gather(cls) -> bool:
        return True
