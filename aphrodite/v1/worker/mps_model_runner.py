from types import SimpleNamespace

import psutil
import torch

from aphrodite.common.config import AphroditeConfig
from aphrodite.v1.worker.gpu_model_runner import GPUModelRunner


class MPSModelRunner(GPUModelRunner):

    def __init__(self, aphrodite_config: AphroditeConfig, device: torch.device):
        super().__init__(aphrodite_config, device)
        # MPS does not support CUDA graphs or pinned memory pipeline
        self.pin_memory = False
        self.use_cuda_graph = False
        self.full_cuda_graph = False
        self.cudagraph_batch_sizes = []

    # --- Device-specific hooks ---
    def _init_device_properties(self) -> None:  # override
        total_mem = psutil.virtual_memory().total
        # Minimal set used downstream; values are placeholders where CUDA-only
        self.device_properties = SimpleNamespace(
            name="Apple MPS",
            total_memory=total_mem,
            multi_processor_count=1,
        )

    def _sync_device(self) -> None:  # override
        # Best-effort sync on MPS
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    # CUDA-graph capture is not supported on MPS
    def capture_model(self) -> None:  # override
        return


