import pytest

from tests.utils import multi_gpu_test
from aphrodite.engine.args_tools import EngineArgs
from aphrodite.common.sampling_params import SamplingParams

from ...utils import check_logprobs_close, check_outputs_equal

# NOTE: The first model in each list is taken as the primary model,
# meaning that it will be used in all tests in this file
# The rest of the models will only be tested by test_models

SSM_MODELS = [
    "state-spaces/mamba-130m-hf",
    "tiiuae/falcon-mamba-tiny-dev",
    # TODO: Compare to a Mamba2 model. The HF transformers implementation of
    # Mamba2 is buggy for Codestral as it doesn't handle n_groups.
    # See https://github.com/huggingface/transformers/pull/35943
    # "mistralai/Mamba-Codestral-7B-v0.1",
]

HYBRID_MODELS = [
    "ai21labs/Jamba-tiny-dev",
    # NOTE: Running Plamo2 in transformers implementation requires to install
    # causal-conv1d package, which is not listed as a test dependency as it's
    # not compatible with pip-compile.
    "pfnet/plamo-2-1b",
    "Zyphra/Zamba2-1.2B-instruct",
    "hmellor/bamba-tiny-random",
]

# Avoid OOM
MAX_NUM_SEQS = 4


@pytest.mark.parametrize("model", SSM_MODELS + HYBRID_MODELS)
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    with aphrodite_runner(model, max_num_seqs=MAX_NUM_SEQS) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=aphrodite_outputs,
        name_0="hf",
        name_1="aphrodite",
    )


@pytest.mark.parametrize("model", SSM_MODELS + HYBRID_MODELS)
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_batching(
    aphrodite_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    for_loop_outputs = []
    with aphrodite_runner(model, max_num_seqs=MAX_NUM_SEQS) as aphrodite_model:
        for prompt in example_prompts:
            single_output, = aphrodite_model.generate_greedy_logprobs([prompt],
                                                                 max_tokens,
                                                                 num_logprobs)
            for_loop_outputs.append(single_output)

        batched_outputs = aphrodite_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=for_loop_outputs,
        outputs_1_lst=batched_outputs,
        name_0="for_loop_aphrodite",
        name_1="batched_aphrodite",
    )


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("chunked_prefill_token_size", [1, 4, 16])
def test_chunked_prefill(
    aphrodite_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
    chunked_prefill_token_size: int,
) -> None:
    max_num_seqs = chunked_prefill_token_size
    max_num_batched_tokens = chunked_prefill_token_size

    with aphrodite_runner(model,
                     enable_chunked_prefill=True,
                     max_num_batched_tokens=max_num_batched_tokens,
                     max_num_seqs=max_num_seqs) as aphrodite_model:
        chunked = aphrodite_model.generate_greedy_logprobs(example_prompts,
                                                      max_tokens, num_logprobs)

    with aphrodite_runner(model,
                     enable_chunked_prefill=False,
                     max_num_seqs=max_num_seqs) as aphrodite_model:
        non_chunked = aphrodite_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=chunked,
        outputs_1_lst=non_chunked,
        name_0="chunked",
        name_1="non_chunked",
    )


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [10])
def test_chunked_prefill_with_parallel_sampling(
    aphrodite_runner,
    example_prompts,
    model: str,
    max_tokens: int,
) -> None:
    """
    Tests chunked prefill in conjunction with n > 1. 
    
    In this case, prefill is populated with decoding tokens and
    we test that it doesn't fail.

    This test might fail if cache is not allocated correctly for n > 1
    decoding steps inside a chunked prefill forward pass
    (where we have both prefill and decode together)
    """
    sampling_params = SamplingParams(n=3,
                                     temperature=1,
                                     seed=0,
                                     max_tokens=max_tokens)
    with aphrodite_runner(
            model,
            enable_chunked_prefill=True,
            # forces prefill chunks with decoding
            max_num_batched_tokens=MAX_NUM_SEQS * 3,
            max_num_seqs=MAX_NUM_SEQS,
    ) as aphrodite_model:
        aphrodite_model.generate(example_prompts, sampling_params)


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [20])
def test_mamba_cache_cg_padding(
    aphrodite_runner,
    example_prompts,
    model: str,
    max_tokens: int,
) -> None:
    """
    This test is for verifying that mamba cache is padded to CG captured
    batch size. If it's not, a torch RuntimeError will be raised because
    tensor dimensions aren't compatible.
    """
    aphrodite_config = EngineArgs(model=model,
                             trust_remote_code=True).create_engine_config()
    while len(example_prompts) == aphrodite_config.pad_for_cudagraph(
            len(example_prompts)):
        example_prompts.append(example_prompts[0])

    try:
        with aphrodite_runner(model) as aphrodite_model:
            aphrodite_model.generate_greedy(example_prompts, max_tokens)
    except RuntimeError:
        pytest.fail(
            "Couldn't run batch size which is not equal to a Cuda Graph "
            "captured batch size. "
            "Could be related to mamba cache not padded correctly")


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [20])
def test_models_preemption_recompute(
    aphrodite_runner,
    example_prompts,
    model: str,
    max_tokens: int,
) -> None:
    """
    Tests that outputs are identical with and w/o preemptions (recompute).
    """
    with aphrodite_runner(model, max_num_seqs=MAX_NUM_SEQS) as aphrodite_model:
        scheduler = aphrodite_model.model.llm_engine.scheduler[0]
        scheduler.ENABLE_ARTIFICIAL_PREEMPT = True
        preempt_aphrodite_outputs = aphrodite_model.generate_greedy(
            example_prompts, max_tokens)

        scheduler.ENABLE_ARTIFICIAL_PREEMPT = False
        aphrodite_outputs = aphrodite_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=preempt_aphrodite_outputs,
        outputs_1_lst=aphrodite_outputs,
        name_0="aphrodite_preepmtions",
        name_1="aphrodite",
    )


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
def test_fail_upon_inc_requests_and_finished_requests_lt_available_blocks(
    aphrodite_runner,
    example_prompts,
    model: str,
) -> None:
    """
    This test is for verifying that the hybrid inner state management doesn't
    collapse in case where the number of incoming requests and
    finished_requests_ids is larger than the maximum mamba block capacity.

    This could generally happen due to the fact that hybrid does support
    statelessness mechanism where it can cleanup new incoming requests in
    a single step.
    """
    try:
        with aphrodite_runner(model, max_num_seqs=MAX_NUM_SEQS) as aphrodite_model:
            aphrodite_model.generate_greedy([example_prompts[0]] * 100, 10)
    except ValueError:
        pytest.fail("Hybrid inner state wasn't cleaned up properly between"
                    "steps finished requests registered unnecessarily ")


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
def test_state_cleanup(
    aphrodite_runner,
    example_prompts,
    model: str,
) -> None:
    """ 
    This test is for verifying that the Hybrid state is cleaned up between
    steps.
    
    If its not cleaned, an error would be expected.
    """
    try:
        with aphrodite_runner(model, max_num_seqs=MAX_NUM_SEQS) as aphrodite_model:
            for _ in range(10):
                aphrodite_model.generate_greedy([example_prompts[0]] * 100, 1)
    except ValueError:
        pytest.fail("Hybrid inner state wasn't cleaned up between states, "
                    "could be related to finished_requests_ids")


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [64])
def test_multistep_correctness(
    aphrodite_runner,
    example_prompts,
    model: str,
    max_tokens: int,
) -> None:
    with aphrodite_runner(model, num_scheduler_steps=8,
                     max_num_seqs=2) as aphrodite_model:
        aphrodite_outputs_multistep = aphrodite_model.generate_greedy(
            example_prompts, max_tokens)

    with aphrodite_runner(model, num_scheduler_steps=1,
                     max_num_seqs=2) as aphrodite_model:
        aphrodite_outputs_single_step = aphrodite_model.generate_greedy(
            example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=aphrodite_outputs_multistep,
        outputs_1_lst=aphrodite_outputs_single_step,
        name_0="aphrodite_outputs_multistep",
        name_1="aphrodite_outputs_single_step",
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_distributed_correctness(
    aphrodite_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with aphrodite_runner(model, tensor_parallel_size=1,
                     max_num_seqs=2) as aphrodite_model:
        aphrodite_outputs_tp_1 = aphrodite_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    with aphrodite_runner(model, tensor_parallel_size=2,
                     max_num_seqs=2) as aphrodite_model:
        aphrodite_outputs_tp_2 = aphrodite_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=aphrodite_outputs_tp_1,
        outputs_1_lst=aphrodite_outputs_tp_2,
        name_0="aphrodite_tp_1",
        name_1="aphrodite_tp_2",
    )
