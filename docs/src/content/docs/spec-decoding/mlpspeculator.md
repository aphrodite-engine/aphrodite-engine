---
title: MLPSpeculator decoding
---

Reference: [Accelerating Production LLMs with Combined Token/Embedding Speculators](https://arxiv.org/abs/2404.19124)

This method was proposed by IBM recently, to accelerate LLM decoding.

Speculative decoding is based on the premise that the model is powerful enough to predict multiple tokens in a single forward pass. However, the current inference servers are optimized to predict only a single token at a time. In this approach, we attach multiple speculative heads (in addition to the usual one) to the LLM to predict $N+1$-th, $N+2$-th, $N+3$-th … token. For example, 3 heads will predict 3 additional tokens. Details of the speculator architecture are explained in a later part of this blog. There are two challenges to achieve efficiency and correctness during inference - one is to predict without replicating KV-cache and the other is to verify that the predictions match the original model’s outcomes.

In a typical generation loop, after the prompt is processed in a single forward step, a sequence length of 1 (next token predicted) is fed into the forward pass of the model along with the kv-cache. In a naive speculative decoding implementation, each speculative head would have its own kv-cache, but instead we use paged attention kernels developed in the [vLLM project](https://vllm.ai) to enable efficient kv-cache maintenance. This ensures that throughput does not reduce at larger batch sizes. Further, we modify the attention masks to enable verification of the N+1’th token and thus enable speculative decoding without deviating from the original model’s output.

You can find the code to train your own MLP speculators [here](https://github.com/foundation-model-stack/fms-fsdp/pull/35).

There a variety of MLPSpeculator models trained for different models, you can find them here:

- [Llama-3 and Llama-3.1](https://huggingface.co/collections/ibm-fms/speculators-66a1b6838f0d2327e0a3a8c3)
- [Granite](https://huggingface.co/collections/ibm-granite/granite-speculators-664b97a44ddc5640e8cd73ac)

## Usage

Python example:

```py
from aphrodite import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_model="ibm-fms/llama3-70b-accelerator",  # [!code highlight]
    speculative_draft_tensor_parallel_size=1,  # [!code highlight]
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

CLI example:

```sh
aphrodite run meta-llama/Meta-Llama-3.1-70B-Instruct \
    --speculative-model ibm-fms/llama3-70b-accelerator \  # [!code highlight]
    --speculative-draft-tensor-parallel-size 1 \  # [!code highlight]
    --use-v2-block-manager  # [!code highlight]
```

