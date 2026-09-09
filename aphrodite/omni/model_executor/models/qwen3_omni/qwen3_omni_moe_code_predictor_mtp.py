# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3-Omni Code Predictor -- thin wrapper over CodePredictorWrapper."""

from aphrodite.config import AphroditeConfig
from aphrodite.omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorWrapper,
    CodePredictorWrapperConfig,
)
from aphrodite.omni.platforms import current_omni_platform


class Qwen3OmniMoeTalkerCodePredictor(CodePredictorWrapper):
    """Qwen3-Omni code predictor (no CUDA graphs, VocabParallelEmbedding)."""

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = "") -> None:
        cp_config = aphrodite_config.model_config.hf_config.code_predictor_config
        super().__init__(
            aphrodite_config=aphrodite_config,
            cp_config=cp_config,
            wrapper_config=CodePredictorWrapperConfig(
                use_cuda_graphs=current_omni_platform.is_npu(),
                use_parallel_embedding=True,
                use_projection=False,
                return_proj_buf=True,
                sampling_mode="stored",
            ),
            talker_hidden_size=cp_config.hidden_size,
            prefix=prefix,
        )
