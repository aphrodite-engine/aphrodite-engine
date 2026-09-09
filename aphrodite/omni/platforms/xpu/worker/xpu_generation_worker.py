# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from aphrodite.logger import init_logger
from aphrodite.omni.platforms.xpu.worker.xpu_generation_model_runner import XPUGenerationModelRunner
from aphrodite.omni.worker.mixins import OmniWorkerMixin
from aphrodite.v1.worker.xpu_worker import XPUWorker

logger = init_logger(__name__)


class XPUGenerationWorker(OmniWorkerMixin, XPUWorker):
    """XPU generation worker for the code2wav (non-AR waveform generation) stage in the Omni model."""

    model_runner_cls = XPUGenerationModelRunner

    def init_device(self):
        super().init_device()
        if self.use_v2_model_runner:
            # OMNI: v2 model runner does not yet include omni hooks.
            logger.warning("OMNI XPUGenerationWorker forces v1 model runner for omni hooks.")
            self.use_v2_model_runner = False
        self.model_runner = self.model_runner_cls(self.aphrodite_config, self.device)
