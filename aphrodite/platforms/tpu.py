import os

import torch

import aphrodite.common.envs as envs
from aphrodite.compilation.levels import CompilationLevel
from aphrodite.plugins import set_torch_compile_backend

from .interface import Platform, PlatformEnum

if "APHRODITE_TORCH_COMPILE_LEVEL" not in os.environ:
    os.environ["APHRODITE_TORCH_COMPILE_LEVEL"] = str(
        CompilationLevel.DYNAMO_ONCE)

assert envs.APHRODITE_TORCH_COMPILE_LEVEL < CompilationLevel.INDUCTOR, \
    "TPU does not support Inductor."

set_torch_compile_backend("openxla")


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU

    @staticmethod
    def inference_mode():
        return torch.no_grad()
