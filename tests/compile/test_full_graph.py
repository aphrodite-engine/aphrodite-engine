import os

import pytest

from aphrodite.common.utils import cuda_device_count_stateless

from ..utils import fork_new_process_for_each_test


@pytest.mark.parametrize("model", ["meta-llama/Meta-Llama-3-8B"])
@pytest.mark.parametrize("tp_size", [1, 2])
@fork_new_process_for_each_test
def test_full_graph(model, tp_size):
    # Skip the test if there are not enough CUDA devices.
    if cuda_device_count_stateless() < tp_size:
        pytest.skip("Not enough CUDA devices for the test.")
    # make sure these models can be captured in full graph mode
    os.environ["APHRODITE_TEST_DYNAMO_GRAPH_CAPTURE"] = "1"

    from aphrodite import LLM, SamplingParams
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)
    llm = LLM(model=model, enforce_eager=True, tensor_parallel_size=tp_size)
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
