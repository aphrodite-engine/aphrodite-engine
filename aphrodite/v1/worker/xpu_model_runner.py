from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from aphrodite.config import AphroditeConfig
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
        with _torch_cuda_wrapper():
            super().__init__(aphrodite_config, device)
        # FIXME: To be verified.
        self.cascade_attn_enabled = False

    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.xpu.synchronize()


@contextmanager
def _torch_cuda_wrapper():

    class _EventPlaceholder:

        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    try:
        # replace cuda Event with xpu Event, this should work by default
        torch.cuda.Event = torch.xpu.Event
        yield
    finally:
        # if anything goes wrong, just patch it with a placeholder
        torch.cuda.Event = _EventPlaceholder
