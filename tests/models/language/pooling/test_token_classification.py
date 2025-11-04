import pytest
import torch
from transformers import AutoModelForTokenClassification

from tests.models.utils import softmax


@pytest.mark.parametrize("model", ["boltuix/NeuroBERT-NER"])
# The float32 is required for this tiny model to pass the test.
@pytest.mark.parametrize("dtype", ["float"])
@torch.inference_mode
def test_bert_models(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with aphrodite_runner(model, max_model_len=None, dtype=dtype) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.token_classify(example_prompts)

    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForTokenClassification) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))

    # check logits difference
    for hf_output, aphrodite_output in zip(hf_outputs, aphrodite_outputs):
        hf_output = torch.tensor(hf_output).cpu().float()
        aphrodite_output = torch.tensor(aphrodite_output).cpu().float()
        assert torch.allclose(hf_output, aphrodite_output, 1e-2)


@pytest.mark.parametrize("model", ["disham993/electrical-ner-ModernBERT-base"])
@pytest.mark.parametrize("dtype", ["float"])
@torch.inference_mode
def test_modernbert_models(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with aphrodite_runner(model, max_model_len=None, dtype=dtype) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.token_classify(example_prompts)

    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForTokenClassification) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))

    # check logits difference
    for hf_output, aphrodite_output in zip(hf_outputs, aphrodite_outputs):
        hf_output = torch.tensor(hf_output).cpu().float()
        aphrodite_output = torch.tensor(aphrodite_output).cpu().float()
        assert torch.allclose(hf_output, aphrodite_output, atol=1e-2)
