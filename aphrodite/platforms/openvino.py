import torch

import aphrodite.common.envs as envs
from aphrodite.common.utils import print_warning_once

from .interface import Platform, PlatformEnum


class OpenVinoPlatform(Platform):
    _enum = PlatformEnum.OPENVINO

    @classmethod
    def get_device_name(self, device_id: int = 0) -> str:
        return "openvino"

    @classmethod
    def inference_mode(self):
        return torch.inference_mode(mode=True)

    @classmethod
    def is_openvino_cpu(self) -> bool:
        return "CPU" in envs.APHRODITE_OPENVINO_DEVICE

    @classmethod
    def is_openvino_gpu(self) -> bool:
        return "GPU" in envs.APHRODITE_OPENVINO_DEVICE

    @classmethod
    def is_pin_memory_available(self) -> bool:
        print_warning_once("Pin memory is not supported on OpenViNO.")
        return False
