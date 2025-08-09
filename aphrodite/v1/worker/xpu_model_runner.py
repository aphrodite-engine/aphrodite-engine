from typing import TYPE_CHECKING

import torch

from aphrodite.common.config import AphroditeConfig
from aphrodite.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    pass


class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
        device: torch.device,
    ):
        super().__init__(aphrodite_config, device)
        # FIXME: To be verified.
        self.cascade_attn_enabled = False

    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.xpu.synchronize()
