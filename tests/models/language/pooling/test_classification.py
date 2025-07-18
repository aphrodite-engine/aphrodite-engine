import pytest
import torch
from transformers import AutoModelForSequenceClassification

from aphrodite.platforms import current_platform


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("jason9693/Qwen2.5-1.5B-apeach",
                     marks=[pytest.mark.core_model, pytest.mark.cpu_model]),
    ],
)
@pytest.mark.parametrize("dtype",
                         ["half"] if current_platform.is_rocm() else ["float"])
def test_models(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
    monkeypatch,
) -> None:
    if current_platform.is_rocm():
        # ROCm Triton FA does not currently support sliding window attention
        # switch to use ROCm CK FA backend
        monkeypatch.setenv("APHRODITE_USE_TRITON_FLASH_ATTN", "False")

    with aphrodite_runner(model, dtype=dtype) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.classify(example_prompts)

    with hf_runner(model,
                   dtype=dtype,
                   auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    # check logits difference
    for hf_output, aphrodite_output in zip(hf_outputs, aphrodite_outputs):
        hf_output = torch.tensor(hf_output)
        aphrodite_output = torch.tensor(aphrodite_output)

        # the tolerance value of 1e-2 is selected based on the
        # half datatype tests in
        # tests/models/embedding/language/test_embedding.py
        assert torch.allclose(hf_output, aphrodite_output,
                              1e-3 if dtype == "float" else 1e-2)
