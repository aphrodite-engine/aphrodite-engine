import pytest
import torch

from aphrodite import SamplingParams

from ..conftest import AphroditeRunner

MODELS = ["distilbert/distilgpt2"]


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    This module is V0 only since it uses dtype=float, so
    set APHRODITE_USE_V1=0 for all tests in the module.
    """
    monkeypatch.setenv('APHRODITE_USE_V1', '0')


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype",
                         ["float"])  # needed for comparing logprobs with HF
@pytest.mark.parametrize("chunked_prefill_token_size", [1, 4, 16, -1])
@pytest.mark.parametrize("num_top_logprobs", [0, 6])  # 32000 == vocab_size
@pytest.mark.parametrize("detokenize", [True, False])
def test_get_prompt_logprobs(
    hf_runner,
    aphrodite_runner,
    model,
    dtype,
    chunked_prefill_token_size: int,
    num_top_logprobs: int,
    detokenize: bool,
    example_prompts,
):
    max_num_seqs = 256
    enable_chunked_prefill = False
    max_num_batched_tokens = None
    if chunked_prefill_token_size != -1:
        enable_chunked_prefill = True
        max_num_seqs = min(chunked_prefill_token_size, max_num_seqs)
        max_num_batched_tokens = chunked_prefill_token_size

    max_tokens = 5
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_logprobs = hf_model.generate_greedy_logprobs(
            example_prompts,
            max_tokens=max_tokens,
        )

    with aphrodite_runner(
            model,
            dtype=dtype,
            max_logprobs=num_top_logprobs,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
    ) as aphrodite_model:
        aphrodite_sampling_params = SamplingParams(max_tokens=max_tokens,
                                              logprobs=num_top_logprobs,
                                              prompt_logprobs=num_top_logprobs,
                                              temperature=0.0,
                                              detokenize=detokenize)
        aphrodite_results = aphrodite_model.model.generate(
            example_prompts, sampling_params=aphrodite_sampling_params)

    # Test whether logprobs are included in the results.
    for result in aphrodite_results:
        assert result.prompt_logprobs is not None
        assert result.outputs[0].logprobs is not None
        assert len(result.outputs[0].logprobs) == max_tokens
        for logprobs in result.outputs[0].logprobs:
            # If the output token is not included in the top X
            # logprob, it can return 1 more data
            assert (len(logprobs) == num_top_logprobs
                    or len(logprobs) == num_top_logprobs + 1)
        output_text = result.outputs[0].text
        output_string_from_most_likely_tokens_lst: list[str] = []
        for top_logprobs in result.outputs[0].logprobs:
            top_logprob = next(iter(top_logprobs.values()))
            output_string_from_most_likely_tokens_lst.append(
                top_logprob.decoded_token)

        if detokenize:
            output_string_from_most_likely_tokens = "".join(
                output_string_from_most_likely_tokens_lst)
            assert output_text == output_string_from_most_likely_tokens, (
                "The output text from the top logprob for each token position "
                "should be the same as the output text in the result.")
        else:
            assert output_text == ''
            assert output_string_from_most_likely_tokens_lst == ([None] *
                                                                 max_tokens)

        # The first prompt logprob is always None
        assert result.prompt_logprobs[0] is None
        for prompt_logprobs in result.prompt_logprobs[1:]:
            # If the prompt token is not included in the top X
            # logprob, it can return 1 more data
            assert (len(prompt_logprobs) == num_top_logprobs
                    or len(prompt_logprobs) == num_top_logprobs + 1)

    # Test whether prompt logprobs are consistent with HF
    for aphrodite_result, hf_logprob in zip(aphrodite_results, hf_logprobs):
        # Check prompt logprobs
        # The first prompt logprob is always None, so we compare it from 1:.
        aphrodite_prompt_logprobs = aphrodite_result.prompt_logprobs[1:]
        for i, aphrodite_prompt_logprob_dict in enumerate(aphrodite_prompt_logprobs):
            for token_id, logprob in aphrodite_prompt_logprob_dict.items():
                torch.testing.assert_close(logprob.logprob,
                                           hf_logprob[0][i][token_id].item(),
                                           atol=1e-2,
                                           rtol=1e-2)
        aphrodite_sample_logprobs = aphrodite_result.outputs[0].logprobs
        for i, top_logprobs in enumerate(aphrodite_sample_logprobs):
            for token_id, sample_logprob in top_logprobs.items():
                logprob = sample_logprob.logprob
                torch.testing.assert_close(logprob,
                                           hf_logprob[i][-1][token_id].item(),
                                           atol=1e-2,
                                           rtol=1e-2)
                if detokenize:
                    assert isinstance(sample_logprob.decoded_token, str), (
                        "The token should be decoded by the time it is returned"
                        " to the user.")

    # Test if prompt logprobs are correctly set.
    for aphrodite_result in aphrodite_results:
        token_ids = aphrodite_result.prompt_token_ids
        prompt_logprobs = aphrodite_result.prompt_logprobs

        # The first token doesn't have logprob.
        assert prompt_logprobs[0] is None

        for token_id, logprob_dict in zip(token_ids[1:], prompt_logprobs[1:]):
            assert token_id in logprob_dict


def test_max_logprobs():
    runner = AphroditeRunner("facebook/opt-125m", max_logprobs=1)
    aphrodite_sampling_params = SamplingParams(logprobs=1)
    # should pass
    runner.generate(["Hello world"], sampling_params=aphrodite_sampling_params)

    bad_sampling_params = SamplingParams(logprobs=2)
    with pytest.raises(ValueError):
        runner.generate(["Hello world"], sampling_params=bad_sampling_params)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("chunked_prefill_token_size", [1, 4, 16, -1])
@pytest.mark.parametrize("detokenize", [True, False])
def test_none_logprobs(aphrodite_runner, model, chunked_prefill_token_size: int,
                       detokenize: bool, example_prompts):
    max_num_seqs = 256
    enable_chunked_prefill = False
    max_num_batched_tokens = None
    if chunked_prefill_token_size != -1:
        enable_chunked_prefill = True
        max_num_seqs = min(chunked_prefill_token_size, max_num_seqs)
        max_num_batched_tokens = chunked_prefill_token_size
    max_tokens = 5

    with aphrodite_runner(
            model,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
    ) as aphrodite_model:
        sampling_params_logprobs_none = SamplingParams(max_tokens=max_tokens,
                                                       logprobs=None,
                                                       temperature=0.0,
                                                       detokenize=detokenize)
        results_logprobs_none = aphrodite_model.model.generate(
            example_prompts, sampling_params=sampling_params_logprobs_none)

    for i in range(len(results_logprobs_none)):
        assert results_logprobs_none[i].outputs[0].logprobs is None
        assert results_logprobs_none[i].outputs[0].cumulative_logprob is None
