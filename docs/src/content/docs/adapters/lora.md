---
title: Low-rank Adaptation (LoRA)
---

Aphrodite allows loading [LoRA](https://arxiv.org/abs/2106.09685) adapters on top of supported LLMs.

We use the method proposed by [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285) to efficiently serve LoRA adapters to thousands of users.

## Python Usage

We can load LoRA adapters directly from Hugging Face or from disk. They're served on a per-request basis.

```py
from aphrodite import LLM, SamplingParams
from aphrodite.lora.request import LoRARequest  # [!code highlight]

llm = LLM(model="meta-llama/Llama-2-7b-hf",
          enable_lora=True) # [!code highlight]
```

With this, we've launched the model with LoRA support enabled. We can now submit our prompts and call `llm.generate()` with the `lora_request` parameter. The first parameter of `LoRARequest` is a human identifiable name, the second is a globally unique ID for the adapter, and the third is the path to the LoRA adapter.

```py
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    stop=["[/assistant]"]
)

prompts = [
     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
]


outputs = llm.generate(  # [!code highlight]
    prompts,  # [!code highlight]
    sampling_params, # [!code highlight]
    lora_request=LoRARequest("alpindale/l2-lora-test", 1, sql_lora_path)   # [!code highlight]
)
```

## Serving with OpenAI API server

LoRA adapters can be served with our OpenAI-compatible API server. To do so, specify `--lora-modules {name}={path} {name}={path} ...` in the CLI. 

```sh
aphrodite run meta-llama/Llama-2-7b-hf \
    --enable-lora \
    --lora-modules l2-lora-test=alpindale/l2-lora-test
```

The YAML config:

```yaml
model: meta-llama/Llama-2-7b-hf
lora_modules:
    - l2-lora-test=alpindale/l2-lora-test
```

The server endpoint accepts all LoRA configuration parameters (`max_loras`, `max_lora_rank`, `max_cpu_loras`), which will apply to all forthcoming requests. Upon querying the `/v1/models` endpoint, you should see the LoRA along with its base model:

```json
{
    "object": "list",
    "data": [
        {
            "id": "meta-llama/Llama-2-7b-hf",
            "object": "model",
            ...
        },
        {
            "id": "l2-lora-test",
            "object": "model",
            ...
        }
    ]
}
```

Requests can specify the LoRA adapter as if it were any other `model` via the model request parameter. The requests will be processed according to the server-wide LoRA configuration (i.e. in parallel with base model requests, and potentially other LoRA adapter requests if they were provided and `max_loras` is set high enough).

Example curl request:

```sh
curl http://localhost:2242/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "l2-lora-test",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }' | jq
```