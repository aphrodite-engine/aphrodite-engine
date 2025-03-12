"""Compare the outputs of HF and Aphrodite when using greedy sampling.

This tests danube3 separately because its head size isn't supported on CPU yet.

Run `pytest tests/models/test_danube3_4b.py`.
"""
import pytest

from ...utils import check_outputs_equal

MODELS = ["h2oai/h2o-danube3-4b-base"]

target_dtype = "half"


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

    with aphrodite_runner(model, dtype=dtype) as aphrodite_model:
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
    with aphrodite_runner(model, dtype=dtype) as aphrodite_model:
        # This test is for verifying whether the model's extra_repr
        # can be printed correctly.
        print(aphrodite_model.model.llm_engine.model_executor.driver_worker.
              model_runner.model)
