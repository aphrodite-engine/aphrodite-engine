---
title: Soft prompts
---

Similar to [LoRA](/adapters/lora), soft prompts are another way to adapt the behaviour of an LLM without fully training all its parameters. Soft prompts are learnable tensors concatenated with the input embeddings that can be optimized to a dataset; the downside is that they aren't human readable because you aren't matching these "virtual tokens" to the embeddings of a eal word.

Please refer to this [PEFT documentation](https://huggingface.co/docs/peft/main/en/conceptual_guides/prompting) for more info.

## Python Usage
Unlike LoRA, loading directly from Hugging Face has not been tested yet, so it's recommended to download your soft prompts to disk beforehand. You can use them like this:

```py
from aphrodite import LLM, SamplingParams
from aphrodite.prompt_adapter.request import PromptAdapterRequest

llm = LLM(model="meta-llama/Llama-2-7b-hf",
          enable_prompt_adapter=True,  # [!code highlight]
          max_prompt_adapter_token=512)  # [!code highlight]
```

Then you can submit your prompts and call `llm.generate()` with the `prompt_adapter_request` parameter. The first parameter of `PromptAdapterRequest` is a human identifiable name, the second is a globally unique ID for the adapter, and the third is the path to the soft prompt.

```py
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    stop=["[/assistant]"]
)

prompt = "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]"

outputs = llm.generate(
    prompts,
    sampling_params,
    prompt_adapter_request=PromptAdapterRequest(  # [!code highlight]
    "tweet_adapter", 1, "swapnilbp/llama_tweet_ptune")  # [!code highlight]
)
```

## Serving with OpenAI API server
Soft prompts can be served with our OpenAI-compatible API server. To do so, specify `--prompt-adapters {name}={path} {name}={path} ...` in the CLI.

```sh
aphrodite run meta-llama/Llama-2-7b-hf \
    --enable-prompt-adapter \
    --prompt-adapters tweet_adapter=swapnilbp/llama_tweet_ptune \
    --max-prompt-adapter-token 512
```

YAML config:
```yaml
model: meta-llama/Llama-2-7b-hf
enable_prompt_adapter: true
prompt_adapters:
  tweet_adapter: swapnilbp/llama_tweet_ptune
max_prompt_adapter_token: 512
```

The server endpoint accepts all Prompt adapter configuration parameters (`max_prompt_adapter_token`, `max_prompt_adapters`), which will apply to all forthcoming requests. Upon querying the `/v1/models` endpoint, you should see the Prompt adapter along with its base model:

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
            "id": "tweet_adapter",
            "object": "model",
            ...
        }
    ]
}
```

Requests can specify the LoRA adapter as if it were any other `model` via the model request parameter. The requests will be processed according to the server-wide LoRA configuration (i.e. in parallel with base model requests, and potentially other LoRA adapter requests if they were provided and `max_prompt_adapters` is set high enough).

Example curl request:

```sh
curl http://localhost:2242/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "tweet_adapter",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }' | jq
```