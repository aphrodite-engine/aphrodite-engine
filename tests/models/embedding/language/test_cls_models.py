"""Compare the outputs of HF and Aphrodite when using classifier models.
This test only tests small models. Big models such as 7B should be tested from
test_big_models.py because it could use a larger instance to run tests.
Run `pytest tests/models/embedding/language/test_cls_models.py`.
"""
import pytest
import torch
from transformers import AutoModelForSequenceClassification

CLASSIFICATION_MODELS = ["jason9693/Qwen2.5-1.5B-apeach"]


@pytest.mark.parametrize("model", CLASSIFICATION_MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_classification_models(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with hf_runner(model,
                   dtype=dtype,
                   auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    with aphrodite_runner(model, dtype=dtype) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.classify(example_prompts)

    print(hf_outputs, aphrodite_outputs)

    # check logits difference
    for hf_output, aphrodite_output in zip(hf_outputs, aphrodite_outputs):
        hf_output = torch.tensor(hf_output)
        aphrodite_output = torch.tensor(aphrodite_output)

        assert torch.allclose(hf_output, aphrodite_output, 1e-3)


@pytest.mark.parametrize("model", CLASSIFICATION_MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_classification_model_print(
    aphrodite_runner,
    model: str,
    dtype: str,
) -> None:
    with aphrodite_runner(model, dtype=dtype) as aphrodite_model:
        # This test is for verifying whether the model's extra_repr
        # can be printed correctly.
        print(aphrodite_model.model.llm_engine.model_executor.driver_worker.
              model_runner.model)
