from typing import Optional

import psutil
import torch

from aphrodite.common.config import AphroditeConfig
from aphrodite.modeling import set_random_seed
from aphrodite.platforms import current_platform
from aphrodite.v1.core.sched.output import SchedulerOutput
from aphrodite.v1.outputs import ModelRunnerOutput
from aphrodite.v1.worker.gpu_worker import Worker as _BaseGPUWorker
from aphrodite.v1.worker.gpu_worker import init_worker_distributed_environment
from aphrodite.v1.worker.mps_model_runner import MPSModelRunner


class Worker(_BaseGPUWorker):

    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(aphrodite_config=aphrodite_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

    def init_device(self) -> None:
        # Select MPS device
        self.device = torch.device("mps")
        current_platform.set_device(self.device)

        # Initialize distributed (use gloo backend on MPS)
        init_worker_distributed_environment(self.aphrodite_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank,
                                            backend=current_platform.dist_backend or "gloo")

        # Seed
        set_random_seed(self.model_config.seed)

        # Construct model runner on MPS
        self.model_runner = MPSModelRunner(self.aphrodite_config, self.device)

    def determine_available_memory(self) -> int:
        """Return available memory budget for KV cache on MPS.

        MPS does not expose detailed free memory APIs. Use a conservative
        fraction of total host memory scaled by gpu_memory_utilization.
        """
        total_bytes = psutil.virtual_memory().total
        util = float(self.cache_config.gpu_memory_utilization)
        # Leave ample headroom (use 40% of requested fraction)
        return int(total_bytes * util * 0.4)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        # Defer to base GPU path (no CUDA ops used here)
        return super().execute_model(scheduler_output)


