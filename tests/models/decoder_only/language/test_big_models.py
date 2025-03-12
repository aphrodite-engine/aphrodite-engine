"""Compare the outputs of HF and Aphrodite when using greedy sampling.

This tests bigger models and use half precision.

Run `pytest tests/models/test_big_models.py`.
"""
import pytest

from aphrodite.platforms import current_platform

from ...utils import check_outputs_equal

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    # "mistralai/Mistral-7B-v0.1",  # Tested by test_mistral.py
    # "Deci/DeciLM-7b",  # Broken
    # "tiiuae/falcon-7b",  # Broken
    "EleutherAI/gpt-j-6b",
    # "mosaicml/mpt-7b",  # Broken
    # "Qwen/Qwen1.5-0.5B"  # Broken,
]

if not current_platform.is_cpu():
    # MiniCPM requires fused_moe which is not supported by CPU
    MODELS.append("openbmb/MiniCPM3-4B")

#TODO: remove this after CPU float16 support ready
target_dtype = "float" if current_platform.is_cpu() else "half"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [32])
def test_models(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    with aphrodite_runner(model, dtype=dtype,
                          enforce_eager=True) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.generate_greedy(
            example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=aphrodite_outputs,
        name_0="hf",
        name_1="aphrodite",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [target_dtype])
def test_model_print(
    aphrodite_runner,
    model: str,
    dtype: str,
) -> None:
    with aphrodite_runner(
        model, dtype=dtype, enforce_eager=True) as aphrodite_model:
        # This test is for verifying whether the model's extra_repr
        # can be printed correctly.
        print(aphrodite_model.model.llm_engine.model_executor.driver_worker.
              model_runner.model)
