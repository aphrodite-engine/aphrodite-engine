# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
import torch
import torch.nn as nn

from aphrodite.config import AphroditeConfig
from aphrodite.v1.worker.gpu.mm.encoder_cache import EncoderCache


def init_model_state(
    aphrodite_config: AphroditeConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
):
    if "WhisperForConditionalGeneration" in aphrodite_config.model_config.architectures:
        from aphrodite.v1.worker.gpu.model_states.whisper import WhisperModelState

        return WhisperModelState(aphrodite_config, model, encoder_cache, device)

    from aphrodite.v1.worker.gpu.model_states.default import DefaultModelState

    return DefaultModelState(aphrodite_config, model, encoder_cache, device)
