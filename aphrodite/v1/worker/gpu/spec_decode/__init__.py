# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from aphrodite.config import AphroditeConfig


def init_speculator(aphrodite_config: AphroditeConfig, device: torch.device):
    speculative_config = aphrodite_config.speculative_config
    assert speculative_config is not None
    if speculative_config.use_eagle():
        from aphrodite.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

        return EagleSpeculator(aphrodite_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
