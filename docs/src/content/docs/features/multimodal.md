---
title: Multimodal inputs
description: Send images, audio, and video to supported models.
---

Select an architecture from the
[multimodal model registry](/reference/models/#multimodal). Each model defines
its accepted modalities, chat template, and preprocessing limits.

## Send an image

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://assets.example.com/image.jpg"},
                },
            ],
        }
    ],
)
```

Use a data URL only for small inputs. Base64 increases request size and proxy
memory use.

## Restrict remote media

```bash
aphrodite serve MODEL \
  --allowed-media-domains assets.example.com images.example.net
```

Allow only hosts that you control. Remote fetching increases request latency
and can access unexpectedly large files.

Use `--allowed-local-media-path` only for a trusted application. Point it at a
dedicated read-only directory.

## Limit items per prompt

```bash
aphrodite serve MODEL \
  --limit-mm-per-prompt '{"image": 4, "video": 1, "audio": 1}'
```

Limits protect encoder memory and preprocessing capacity. The model processor
can impose a lower limit.

## Configure preprocessing

`--mm-processor-kwargs` passes model-specific settings to the processor. Use
settings documented by the model implementation. Image resolution, video frame
count, and audio duration affect both latency and memory.

## Benchmark media

Text-only tests do not measure encoder or download cost. Use
`aphrodite bench mm-processor` for preprocessing and a multimodal serving
dataset for end-to-end latency.

Warm media caches before a warm-cache test. Keep a separate cold-media result.

## Troubleshoot inputs

Check the model chat template when text works and media fails. Confirm MIME
type, URL access from the server, item count, and processor limits.

Set a request body limit at the proxy. Return a client error before the server
downloads media that violates policy.
