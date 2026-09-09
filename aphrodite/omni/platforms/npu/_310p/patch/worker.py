# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Omni NPU worker device initialization for 310P.

This patches:
    aphrodite.omni.platforms.npu.worker.base.OmniNPUWorkerBase

Omni AR/generation workers share this base initialization path, so the 310P
compile-mode setup is applied here after device initialization.
"""

from __future__ import annotations

from aphrodite_ascend._310p.sample.sampler import AscendSampler310

from aphrodite.omni.platforms.npu._310p import disable_jit_compile
from aphrodite.omni.platforms.npu.worker import base as worker_base
from aphrodite.v1.sample.sampler import Sampler


class _OmniNPUWorkerBase310P(worker_base.OmniNPUWorkerBase):
    def _init_device(self):
        device = super()._init_device()
        disable_jit_compile()
        return device


def apply_patch() -> None:
    # Triton-Ascend does not target 310P; use Sonar's native penalty path.
    AscendSampler310.apply_penalties = staticmethod(Sampler.apply_penalties)
    setattr(worker_base, "OmniNPUWorkerBase", _OmniNPUWorkerBase310P)
