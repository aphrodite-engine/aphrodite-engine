import enum
import platform
from typing import NamedTuple, Optional, Tuple

import torch


class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    XPU = enum.auto()
    CPU = enum.auto()
    UNSPECIFIED = enum.auto()


class CpuArchEnum(enum.Enum):
    X86 = enum.auto()
    ARM = enum.auto()
    POWERPC = enum.auto()
    OTHER = enum.auto()
    UNKNOWN = enum.auto()


class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """
        Express device capability as an integer ``<major><minor>``.
        It is assumed that the minor version is always a single digit.
        """
        assert 0 <= self.minor < 10
        return self.major * 10 + self.minor


class Platform:
    _enum: PlatformEnum

    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    def is_xpu(self) -> bool:
        return self._enum == PlatformEnum.XPU

    @staticmethod
    def get_device_capability(device_id: int = 0) -> Optional[Tuple[int, int]]:
        return None

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        raise NotImplementedError

    @staticmethod
    def inference_mode():
        """A device-specific wrapper of `torch.inference_mode`.
        This wrapper is recommended because some hardware backends such as TPU
        do not support `torch.inference_mode`. In such a case, they will fall
        back to `torch.no_grad` by overriding this method.
        """
        return torch.inference_mode(mode=True)


    @classmethod
    def get_cpu_architecture(cls) -> CpuArchEnum:
        """
        Determine the CPU architecture of the current system.
        Returns CpuArchEnum indicating the architecture type.
        """
        machine = platform.machine().lower()

        if machine in ("x86_64", "amd64", "i386", "i686"):
            return CpuArchEnum.X86
        elif machine.startswith("arm") or machine.startswith("aarch"):
            return CpuArchEnum.ARM
        elif machine.startswith("ppc"):
            return CpuArchEnum.POWERPC

        return CpuArchEnum.OTHER if machine else CpuArchEnum.UNKNOWN


class UnspecifiedPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
