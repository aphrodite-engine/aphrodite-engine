import pytest

from ...utils import check_logprobs_close

MODELS = [
    # TODO(sang): Sliding window should be tested separately.
    "ibm/PowerLM-3b",
    "ibm/PowerMoE-3b",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    with aphrodite_runner(model, dtype=dtype) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=aphrodite_outputs,
        name_0="hf",
        name_1="aphrodite",
    )
