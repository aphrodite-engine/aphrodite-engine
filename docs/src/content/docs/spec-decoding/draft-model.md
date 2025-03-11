---
title: Speculative Decoding with a Draft Model
---

This is the most traditional method for performing speculative decoding with LLMs: you load a smaller model (commonly referred to as the "draft model") of the same architecture as your main model (commonly referred to as the "target model").

Python example:

```py
from a[jrpdote] import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=1,
    speculative_model="facebook/opt-125m",  # [!code highlight]
    num_speculative_tokens=5,  # [!code highlight]
    use_v2_block_manager=True,  # [!code highlight]
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

In this example, we use the `facebook/opt-6.7b` model as the target model and the `facebook/opt-125m` model as the draft model. We generate 5 speculative tokens for each request. You can adjust the `num_speculative_tokens` parameter to control the number of speculative tokens generated, and find the optimal value for your use case.

CLI example:

```sh
aphrodite run facebook/opt-6.7b --speculative-model facebook/opt-125m --num-speculative-tokens 5 --use-v2-block-manager
```


