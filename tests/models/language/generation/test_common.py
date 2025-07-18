import os
from typing import Optional

import pytest
import torch

from aphrodite.platforms import current_platform

from ....utils import large_gpu_mark
from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_logprobs_close

# These have unsupported head_dim for FA. We do not
# not have a clean way to fall back, so we fail with
# a clear msg when it happens.
REQUIRES_V0 = ["microsoft/phi-2", "stabilityai/stablelm-3b-4e1t"]

# This list contains the model that are using AITER kernel.
# Skip model that are not using AITER tests.
# When more AITER kernels are added, this list will not be
# needed as all the models will be calling AITER kernels
# in parts of the operators
AITER_MODEL_LIST = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "openbmb/MiniCPM3-4B",
    "Qwen/Qwen-7B-Chat",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TitanML/tiny-mixtral",
]


# @maybe_test_rocm_aiter
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            "bigscience/bloom-560m",  # bloom - testing alibi slopes
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "openai-community/gpt2",  # gpt2
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param("Milos/slovak-gpt-j-405M"),  # gptj
        pytest.param("bigcode/tiny_starcoder_py"),  # gpt_bigcode
        pytest.param("EleutherAI/pythia-70m"),  # gpt_neox
        pytest.param(
            "google/gemma-1.1-2b-it",  # gemma
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "THUDM/chatglm3-6b",  # chatglm (text-only)
        ),
        pytest.param(
            "meta-llama/Llama-3.2-1B-Instruct",  # llama
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "openbmb/MiniCPM3-4B",
            # fused_moe not supported on CPU
            marks=[pytest.mark.core_model,
                   large_gpu_mark(min_gb=32)],
        ),
        pytest.param(
            "facebook/opt-125m",  # opt
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "microsoft/phi-2",  # phi
            marks=[pytest.mark.core_model],
        ),
        pytest.param(
            "Qwen/Qwen-7B-Chat",  # qwen (text-only)
        ),
        pytest.param(
            "Qwen/Qwen2.5-0.5B-Instruct",  # qwen2
            marks=[pytest.mark.core_model],
        ),
        pytest.param("stabilityai/stablelm-3b-4e1t"),  # stablelm
        pytest.param("bigcode/starcoder2-3b"),  # starcoder2
        pytest.param(
            "TitanML/tiny-mixtral",  # mixtral
            marks=[pytest.mark.cpu_model],
        )
    ])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False])
def test_models(hf_runner, aphrodite_runner, example_prompts, model: str,
                max_tokens: int, num_logprobs: int, use_rocm_aiter: bool,
                monkeypatch) -> None:

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    if model in REQUIRES_V0:
        monkeypatch.setenv("APHRODITE_USE_V1", "0")

    if use_rocm_aiter and (model in AITER_MODEL_LIST):
        monkeypatch.setenv("APHRODITE_ROCM_USE_AITER", "1")
    elif use_rocm_aiter and model not in AITER_MODEL_LIST:
        # Skip model that are not using AITER tests.
        # When more AITER kernels are added, this list will not be
        # needed as all the models will be calling AITER kernels
        # in parts of the operators
        pytest.skip(f"Skipping '{model}' model test with AITER kernel.")

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

        prompt_embeds: Optional[list[torch.Tensor]] = [] if os.getenv(
            "APHRODITE_USE_V1") == "0" else None
        prompt_token_ids = []
        for prompt in example_prompts:
            token_ids = hf_model.tokenizer(prompt,
                                           return_tensors="pt").input_ids.to(
                                               hf_model.model.device)
            prompt_token_ids.append(token_ids)
            if prompt_embeds is not None:
                prompt_embeds.append(hf_model.model.get_input_embeddings()(
                    token_ids).squeeze(0))

    with aphrodite_runner(
            model,
            tokenizer_name=model_info.tokenizer or model,
            tokenizer_mode=model_info.tokenizer_mode,
            trust_remote_code=model_info.trust_remote_code,
            max_num_seqs=2,
    ) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)
        if prompt_embeds is not None:
            aphrodite_outputs_from_embeds = aphrodite_model.generate_greedy_logprobs(
                prompt_embeds, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=aphrodite_outputs,
        name_0="hf",
        name_1="aphrodite",
    )
    if prompt_embeds is not None:
        check_logprobs_close(
            outputs_0_lst=aphrodite_outputs,
            outputs_1_lst=aphrodite_outputs_from_embeds,
            name_0="aphrodite",
            name_1="aphrodite_from_embeds",
        )

    if use_rocm_aiter:
        # this is to ensure that aphrodite engine
        # has deallocated the memory before running the next
        # unit tests. On ROCm, when using AITER
        # the memory might not be deallocated completely
        # before running the next test case
        torch.cuda.synchronize()
