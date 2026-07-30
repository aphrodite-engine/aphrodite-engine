# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Literal

import pytest
from transformers import PretrainedConfig

from aphrodite.config.model import ModelConfig
from aphrodite.config.parallel import ParallelConfig
from aphrodite.config.speculative import SpeculativeConfig

EAGLE3_DRAFT = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
LLAMA3_TARGET = "unsloth/Meta-Llama-3.1-8B-Instruct"
AR_MODEL = "JackFram/llama-68m"

_LOGGER = "aphrodite.config.speculative"
_OVERRIDE_MSG = "Overriding draft model max_position_embeddings"


@pytest.fixture
def aphrodite_caplog(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(logging.getLogger("aphrodite"), "propagate", True)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        yield caplog


def _override_logged(caplog: pytest.LogCaptureFixture) -> bool:
    return any(_OVERRIDE_MSG in record.getMessage() for record in caplog.records)


@pytest.mark.cpu_test
def test_override_raises_smaller_value(
    aphrodite_caplog: pytest.LogCaptureFixture,
):
    hf_config = PretrainedConfig(max_position_embeddings=2048)
    SpeculativeConfig._maybe_override_draft_max_position_embeddings(hf_config, target_max_model_len=8192)
    assert hf_config.max_position_embeddings == 8192
    assert _override_logged(aphrodite_caplog)


@pytest.mark.cpu_test
def test_override_keeps_sufficient_value(
    aphrodite_caplog: pytest.LogCaptureFixture,
):
    hf_config = PretrainedConfig(max_position_embeddings=8192)
    SpeculativeConfig._maybe_override_draft_max_position_embeddings(hf_config, target_max_model_len=8192)
    assert hf_config.max_position_embeddings == 8192
    assert not _override_logged(aphrodite_caplog)


@pytest.mark.cpu_test
def test_override_ignores_missing_attribute(
    aphrodite_caplog: pytest.LogCaptureFixture,
):
    hf_config = PretrainedConfig()
    hf_config.__dict__.pop("max_position_embeddings", None)
    SpeculativeConfig._maybe_override_draft_max_position_embeddings(hf_config, target_max_model_len=8192)
    assert not hasattr(hf_config, "max_position_embeddings")
    assert not _override_logged(aphrodite_caplog)


@pytest.mark.cpu_test
@pytest.mark.parametrize("method", ["eagle", "eagle3"])
def test_eagle_draft_inherits_target_max_model_len(
    method: Literal["eagle", "eagle3"],
    aphrodite_caplog: pytest.LogCaptureFixture,
):
    target_model_config = ModelConfig(LLAMA3_TARGET)
    assert target_model_config.max_model_len > 2048
    speculative_config = SpeculativeConfig(
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
        model=EAGLE3_DRAFT,
        method=method,
        num_speculative_tokens=3,
    )
    draft_hf_config = speculative_config.draft_model_config.hf_config
    assert draft_hf_config.max_position_embeddings == target_model_config.max_model_len
    assert _override_logged(aphrodite_caplog)


@pytest.mark.cpu_test
def test_independent_draft_model_keeps_its_own_limit(
    aphrodite_caplog: pytest.LogCaptureFixture,
):
    target_model_config = ModelConfig(AR_MODEL, hf_overrides={"max_position_embeddings": 8192})
    assert target_model_config.max_model_len == 8192
    speculative_config = SpeculativeConfig(
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
        model=AR_MODEL,
        method="draft_model",
        num_speculative_tokens=3,
    )
    draft_hf_config = speculative_config.draft_model_config.hf_config
    assert draft_hf_config.max_position_embeddings == 2048
    assert not _override_logged(aphrodite_caplog)
