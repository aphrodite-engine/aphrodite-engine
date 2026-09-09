# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from aphrodite.omni.platforms.npu.worker.base import OmniNPUWorkerBase
from aphrodite.omni.platforms.npu.worker.npu_generation_model_runner import NPUGenerationModelRunner
from aphrodite.omni.worker.mixins import OmniWorkerMixin
from aphrodite.v1.worker.workspace import init_workspace_manager


class NPUGenerationWorker(OmniWorkerMixin, OmniNPUWorkerBase):
    """NPU generation worker for code2wav stage in Omni model."""

    model_runner_cls = NPUGenerationModelRunner

    def init_device(self):
        self.device = self._init_device()
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)

        self.model_runner = self.model_runner_cls(self.aphrodite_config, self.device)
