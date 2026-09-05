# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backbone validation for Jina Embeddings V5.

The V5 family ships two backbones under one `architectures` entry: `-small` is
a Qwen3 decoder, while `-nano` is a bidirectional EuroBERT encoder. Upstream
ships a separate `configuration_*.py` per repository, so the only signal
distinguishing them is `is_decoder`, which the encoder variant sets to False.
`JinaEmbeddingsV5ModelConfig` uses it to enable bidirectional attention for the
encoder variant; `JinaEmbeddingsV5Model` then dispatches to the correct backbone.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from transformers import PretrainedConfig

from aphrodite.config import ModelConfig
from aphrodite.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    JinaEmbeddingsV5ModelConfig,
)


def _model_config(hf_config: PretrainedConfig) -> ModelConfig:
    """Return a minimal stand-in; the validator only reads hf_config."""
    return cast(ModelConfig, SimpleNamespace(hf_config=hf_config))


@pytest.mark.cpu_test
def test_registered_for_the_architecture() -> None:
    assert MODELS_CONFIG_MAP["JinaEmbeddingsV5Model"] is JinaEmbeddingsV5ModelConfig


@pytest.mark.cpu_test
def test_encoder_backbone_enables_bidirectional_attention():
    """An encoder checkpoint (is_decoder=False) is supported.

    The handler sets is_causal=False so the Llama backbone uses
    EncoderOnlyAttention; JinaEmbeddingsV5Model then dispatches to the encoder
    implementation.
    """
    hf_config = PretrainedConfig(is_decoder=False)

    JinaEmbeddingsV5ModelConfig.verify_and_update_model_config(_model_config(hf_config))

    assert hf_config.is_causal is False


@pytest.mark.cpu_test
def test_supported_decoder_backbone_is_accepted() -> None:
    absent = PretrainedConfig()
    assert not hasattr(absent, "is_decoder")

    JinaEmbeddingsV5ModelConfig.verify_and_update_model_config(_model_config(absent))
    JinaEmbeddingsV5ModelConfig.verify_and_update_model_config(_model_config(PretrainedConfig(is_decoder=True)))
