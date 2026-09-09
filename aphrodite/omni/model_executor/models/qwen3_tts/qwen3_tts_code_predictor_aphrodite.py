# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3-TTS Code Predictor -- thin wrapper over CodePredictorWrapper."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from aphrodite.config import AphroditeConfig
from aphrodite.config.aphrodite import set_current_aphrodite_config
from aphrodite.omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorBaseModel,
    CodePredictorWrapper,
    CodePredictorWrapperConfig,
)
from aphrodite.omni.platforms import current_omni_platform

from .configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig

# Backward-compat alias used by tests
Qwen3TTSTalkerCodePredictorModelAPHRODITE = CodePredictorBaseModel


class Qwen3TTSTalkerCodePredictorForConditionalGenerationAPHRODITE(CodePredictorWrapper):
    """Qwen3-TTS code predictor (per-call sampling, projection)."""

    def __init__(
        self,
        *,
        aphrodite_config: AphroditeConfig,
        config: Qwen3TTSTalkerCodePredictorConfig,
        talker_config: Qwen3TTSTalkerConfig,
        prefix: str = "code_predictor",
    ) -> None:
        super().__init__(
            aphrodite_config=aphrodite_config,
            cp_config=config,
            wrapper_config=CodePredictorWrapperConfig(
                use_cuda_graphs=current_omni_platform.is_npu(),
                use_parallel_embedding=False,
                use_projection=(config.hidden_size != talker_config.hidden_size),
                return_proj_buf=False,
                sampling_mode="per_call",
            ),
            talker_hidden_size=int(talker_config.hidden_size),
            prefix=prefix,
        )
        # Store talker_config for backward compat (accessed by some callers)
        self.talker_config = talker_config
        self._aphrodite_config = aphrodite_config

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with aphrodite config context (required for VocabParallelEmbedding)."""
        with set_current_aphrodite_config(self._aphrodite_config):
            return super().load_weights(weights)
