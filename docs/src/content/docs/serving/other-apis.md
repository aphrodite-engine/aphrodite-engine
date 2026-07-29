---
title: Other APIs
description: Use Sonar endpoints outside the OpenAI-compatible API.
---

Sonar provides endpoint families for tasks that do not fit the OpenAI API.
Available routes depend on the model runner and server configuration.

## Pooling and scoring

Pooling models can provide embeddings, classifications, rewards, and token-level
representations. Use the task endpoint that matches the model runner.

The principal routes are:

| Route | Typical use |
| --- | --- |
| `/v1/embeddings` | OpenAI-compatible embeddings |
| `/v2/embed` | Native embedding requests |
| `/pooling` | Raw pooled model output |
| `/score` | Score one or more text pairs |
| `/rerank` and `/v2/rerank` | Rank documents against a query |
| `/generative_scoring` | Score candidates with a generation model |

Start a model with the runner and task required by that model. Check `/docs` on
a running server for the installed version's exact request schema.

## Tokenization

Use `/tokenize` to convert text to token IDs. Use `/detokenize` for the reverse
operation. These endpoints use the tokenizer loaded by the server.

## Render and derender

The render API performs request preprocessing without model execution. The
derender API converts generated token IDs into a response. These endpoints help
separate CPU preprocessing from GPU inference.

OpenAI-compatible render routes include `/v1/chat/completions/render` and
`/v1/completions/render`. Sonar also provides
`/inference/v1/generate` for token-input and token-output workflows. These APIs
are useful when a gateway performs tokenization or when CPU preprocessing runs
separately from GPU workers.

Do not assume token IDs are portable between models. The tokenizer, added
tokens, chat template, and model revision must match.

## Anthropic Messages

Sonar provides `/v1/messages` and `/v1/messages/count_tokens` for
Anthropic-compatible clients. Confirm that the selected model has a suitable
chat template. Parser support for reasoning and tool calls remains
model-specific.

## Transcription

Audio transcription models expose `/v1/audio/transcriptions`. Use multipart
form data as required by the OpenAI-compatible transcription schema. Check the
[model matrix](/reference/models/) before you select a model.

## Kobold API

Use `--launch-kobold-api` to enable the Kobold-compatible routes. Check the
generated [server arguments](/reference/server-arguments/) for related options.

## Health and metrics

- `/health` reports server health.
- `/metrics` exports Prometheus metrics.
- `/docs` shows the interactive OpenAPI documentation.
- `/openapi.json` returns the machine-readable schema.

Do not expose administrative and diagnostic endpoints to an untrusted network.

## Discover routes on your server

The available routes depend on the selected model and server mode. Inspect the
schema instead of relying on a route from another deployment:

```bash
curl -fsS http://127.0.0.1:2242/openapi.json > openapi.json
```

Use the schema from the same Sonar commit that runs in production. This avoids
client drift when an endpoint gains a field or changes validation.
