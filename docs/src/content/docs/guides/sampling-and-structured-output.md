---
title: Sampling and structured output
description: Control generation and constrain Sonar output.
---

Sampling parameters change token selection for each request. Server arguments
set model and scheduler behavior for the complete process.

## Deterministic generation

Use temperature zero for greedy token selection:

```python
params = SamplingParams(temperature=0.0, max_tokens=256)
```

A fixed seed helps reproduce stochastic sampling. GPU kernels and batch shapes
can still cause small numerical differences.

## Common sampling controls

`temperature` changes distribution sharpness. `top_p` keeps the smallest token
set that reaches the selected probability mass. `top_k` limits selection to a
fixed number of tokens. `min_p` removes tokens below a fraction of the most
likely token probability.

Start with one truncation method. Combining several methods makes behavior
harder to interpret.

Use `presence_penalty` and `frequency_penalty` for OpenAI-style penalties.
`repetition_penalty` applies a multiplicative penalty. DRY targets repeated
sequences and has separate controls in `SamplingParams`.

## Stop generation

Set `max_tokens` for every request. Add text stops or token ID stops when the
application has a protocol delimiter.

```python
params = SamplingParams(
    temperature=0.2,
    max_tokens=256,
    stop=["</answer>"],
)
```

The model end-of-sequence token also stops generation unless `ignore_eos` is
enabled.

## Generate JSON from Python

```python
from aphrodite import LLM, SamplingParams
from aphrodite.sampling_params import StructuredOutputsParams

params = SamplingParams(
    max_tokens=128,
    structured_outputs=StructuredOutputsParams(
        json={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
        }
    ),
)
```

The schema constrains valid output tokens. Validate the returned JSON in the
application because a schema does not apply business rules.

Structured output also supports a regular expression, a choice list, a grammar,
and JSON-object mode. Set one primary constraint per request.

## Use the OpenAI response format

OpenAI-compatible clients can pass a JSON schema through `response_format`.
Use the current server schema at `/openapi.json` to confirm the accepted request
shape.

## Performance

Complex schemas increase preprocessing and token-mask work. Reuse schemas when
possible. Benchmark large choice sets and deeply nested schemas before
production use.
