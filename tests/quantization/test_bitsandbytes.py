'''Tests whether bitsandbytes computation is enabled correctly.

Run `pytest tests/quantization/test_bitsandbytes.py`.
'''

import gc

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported

models_4bit_to_test = [
    ('huggyllama/llama-7b', 'quantize model inflight'),
]

models_pre_qaunt_4bit_to_test = [
    ('lllyasviel/omost-llama-3-8b-4bits',
     'read pre-quantized 4-bit NF4 model'),
    ('PrunaAI/Einstein-v6.1-Llama3-8B-bnb-4bit-smashed',
     'read pre-quantized 4-bit FP4 model'),
]

models_pre_quant_8bit_to_test = [
    ('meta-llama/Llama-Guard-3-8B-INT8', 'read pre-quantized 8-bit model'),
]


@pytest.mark.skipif(not is_quant_method_supported("bitsandbytes"),
                    reason='bitsandbytes is not supported on this GPU type.')
@pytest.mark.parametrize("model_name, description", models_4bit_to_test)
def test_load_4bit_bnb_model(hf_runner, aphrodite_runner, example_prompts,
                             model_name, description) -> None:

    hf_model_kwargs = {"load_in_4bit": True}
    validate_generated_texts(hf_runner, aphrodite_runner, example_prompts[:1],
                             model_name, hf_model_kwargs)


@pytest.mark.skipif(not is_quant_method_supported("bitsandbytes"),
                    reason='bitsandbytes is not supported on this GPU type.')
@pytest.mark.parametrize("model_name, description",
                         models_pre_qaunt_4bit_to_test)
def test_load_pre_quant_4bit_bnb_model(hf_runner, aphrodite_runner,
                                       example_prompts,
                                       model_name, description) -> None:

    validate_generated_texts(hf_runner, aphrodite_runner, example_prompts[:1],
                             model_name)


@pytest.mark.skipif(not is_quant_method_supported("bitsandbytes"),
                    reason='bitsandbytes is not supported on this GPU type.')
@pytest.mark.parametrize("model_name, description",
                         models_pre_quant_8bit_to_test)
def test_load_8bit_bnb_model(hf_runner, aphrodite_runner, example_prompts,
                             model_name, description) -> None:

    validate_generated_texts(hf_runner, aphrodite_runner, example_prompts[:1],
                             model_name)


def log_generated_texts(prompts, outputs, runner_name):
    logged_texts = []
    for i, (_, generated_text) in enumerate(outputs):
        log_entry = {
            "prompt": prompts[i],
            "runner_name": runner_name,
            "generated_text": generated_text,
        }
        logged_texts.append(log_entry)
    return logged_texts


def validate_generated_texts(hf_runner,
                             aphrodite_runner,
                             prompts,
                             model_name,
                             hf_model_kwargs=None):

    if hf_model_kwargs is None:
        hf_model_kwargs = {}

    # Run with HF runner
    with hf_runner(model_name, model_kwargs=hf_model_kwargs) as llm:
        hf_outputs = llm.generate_greedy(prompts, 8)
        hf_logs = log_generated_texts(prompts, hf_outputs, "HfRunner")

    # Clean up the GPU memory for the next test
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    #Run with Aphrodite runner
    with aphrodite_runner(model_name,
                     quantization='bitsandbytes',
                     load_format='bitsandbytes',
                     enforce_eager=False,
                     gpu_memory_utilization=0.8) as llm:
        aphrodite_outputs = llm.generate_greedy(prompts, 8)
        aphrodite_logs = log_generated_texts(prompts, aphrodite_outputs,
                                             "AphroditeRunner")

    # Clean up the GPU memory for the next test
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Compare the generated strings
    for hf_log, aphrodite_log in zip(hf_logs, aphrodite_logs):
        hf_str = hf_log["generated_text"]
        aphrodite_str = aphrodite_log["generated_text"]
        prompt = hf_log["prompt"]
        assert hf_str == aphrodite_str, (f"Model: {model_name}"
                                    f"Mismatch between HF and Aphrodite "
                                    "outputs:\n"
                                    f"Prompt: {prompt}\n"
                                    f"HF Output: '{hf_str}'\n"
                                    f"Aphrodite Output: '{aphrodite_str}'")
