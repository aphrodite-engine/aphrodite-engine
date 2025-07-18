"""Tests whether gptq models with quantized lm_head can be loaded.

Run `pytest tests/quantization/test_quant_lm_head_true.py --forked`.
"""

import pytest
import torch

from aphrodite.quantization.gptq import GPTQLinearMethod
from aphrodite.quantization.gptq_marlin import (
    GPTQMarlinLinearMethod)
from aphrodite.quantization.marlin import MarlinLinearMethod
from aphrodite.modeling.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod)

PROMPT = "On the surface of Mars, we found"

MODELS_QUANT = [
    ("ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head", True),
    ("ModelCloud/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit-10-25-2024", False),
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", False),
    ("neuralmagic/Meta-Llama-3-8B-Instruct-FP8", False)
]


@pytest.mark.parametrize("model_id, lm_head_quantized", MODELS_QUANT)
def test_lm_head(
    aphrodite_runner,
    model_id: str,
    lm_head_quantized: bool,
    monkeypatch,
) -> None:
    # aphrodite_runner.apply_model() relies on V0 internals.
    monkeypatch.setenv("APHRODITE_USE_V1", "0")
    with aphrodite_runner(model_id, dtype=torch.float16,
                     max_model_len=2048) as aphrodite_model:

        def check_model(model):
            lm_head_layer = model.lm_head
            if lm_head_quantized:
                assert isinstance(lm_head_layer.quant_method,
                                  (GPTQLinearMethod, GPTQMarlinLinearMethod,
                                   MarlinLinearMethod))
            else:
                assert isinstance(lm_head_layer.quant_method,
                                  UnquantizedEmbeddingMethod)

        aphrodite_model.apply_model(check_model)

        print(
            aphrodite_model.generate_greedy(prompts=["Hello my name is"],
                                       max_tokens=10)[0][1])
