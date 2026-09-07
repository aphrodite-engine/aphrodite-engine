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

## Configure video decoding

Aphrodite decodes video bytes into frames using a selectable decoding backend.

| Backend | Device | Description |
| --- | --- | --- |
| `opencv` (default) | CPU | OpenCV-based decoder |
| `pyav` | CPU | PyAV decoder |
| `torchcodec` | CPU | TorchCodec (PyTorch-native) decoder |
| `pynvvideocodec` | GPU | NVIDIA PyNvVideoCodec decoder |
| `deepstream` | GPU | NVIDIA DeepStream decoder |

The three CPU backends are ultimately backed by FFmpeg. `torchcodec` lets you
choose which FFmpeg version is used, while `opencv` and `pyav` rely on the
FFmpeg build they were linked against.

Select the decoder with `--media-io-kwargs`:

```bash
aphrodite serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --media-io-kwargs '{"video": {"backend": "torchcodec"}}'
```

### GPU video decoding with PyNvVideoCodec

The `pynvvideocodec` backend uses NVIDIA NVDEC to decode sampled video frames
on the GPU before copying them into host memory for multimodal preprocessing.
For workloads with large videos and relatively light inference, such as video
tagging, this can alleviate bottlenecks in CPU-based video decoders.

> **Warning:** [CUDA Multi-Process Service
> (MPS)](https://docs.nvidia.com/deploy/mps/quick-start.html) is required when
> using this backend. Video decoding runs in the API server process while model
> serving runs in the engine process, so multiple CUDA processes share the same
> GPU. Configure and start MPS before starting Aphrodite.

You must also set a positive `--mm-ipc-gpu-memory-gb` value to reserve VRAM for
video decoding. Aphrodite carves this budget out of the memory available to the
KV cache and uses it to bound concurrent frontend decode allocations. If the
budget is exhausted, decode work waits instead of consuming the engine's VRAM
headroom and potentially causing an out-of-memory error while serving requests.

Select the backend with an environment variable and specify a
workload-appropriate VRAM budget. For example, to reserve 1 GiB:

```bash
export APHRODITE_VIDEO_LOADER_BACKEND=pynvvideocodec
aphrodite serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --mm-ipc-gpu-memory-gb 1
```

Alternatively, select it with `--media-io-kwargs`:

```bash
aphrodite serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --media-io-kwargs '{"video": {"backend": "pynvvideocodec"}}' \
  --mm-ipc-gpu-memory-gb 1
```

Choose a budget large enough for the largest sampled video that a single API
server process must decode. When using multiple API server processes,
Aphrodite divides the configured budget evenly among them.

For streaming video sources, use the `deepstream` backend instead.

## Configure audio decoding

Aphrodite decodes audio bytes into waveforms using a selectable decoding backend.

| Backend | Description |
| --- | --- |
| `auto` (default) | soundfile, falling back to torchcodec, then PyAV |
| `soundfile` | libsndfile only, no fallback |
| `pyav` | PyAV (FFmpeg) only, no fallback |
| `torchcodec` | TorchCodec (PyTorch-native) only, no fallback |

Select the decoder with `--media-io-kwargs`:

```bash
aphrodite serve mistralai/Voxtral-Mini-3B-2507 \
  --media-io-kwargs '{"audio": {"audio_backend": "torchcodec"}}'
```

PyAV drives FFmpeg through a per-frame Python generator, so concurrent decoding
can contend on the GIL. TorchCodec decodes each stream in a single call that
releases the GIL for its duration. Select it explicitly for concurrent decoding
workloads.

`auto` prefers soundfile for supported formats to preserve their existing
decoding behavior, including encoder padding. Audio extracted from formats
that soundfile cannot read, such as video containers, can use torchcodec
through the fallback chain.

TorchCodec requires both the package and a system FFmpeg installation. Install
it manually if it is not included on your platform. If the package or FFmpeg
is unavailable, `auto` uses the soundfile → PyAV chain.

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
