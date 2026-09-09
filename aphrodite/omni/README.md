# Sonar Omni

Experimental in-tree port of [vLLM-Omni](https://github.com/vllm-project/vllm-omni).
The runtime supports staged generation, including autoregressive models followed
by audio or diffusion stages.

## Entry points

Install the `omni` extra in a Sonar development environment:

```sh
uv pip install -e '.[omni]'
aphrodite serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni
```

Review the stage memory limits in `deploy/qwen3_tts.yaml` before launching.
Each stage has its own worker and memory allocation.

The Python entry points are:

```python
from aphrodite.omni import AsyncOmni, Omni
```

Importing this package activates Omni's process-wide patches. Use a separate
process for ordinary Sonar serving. Without `--omni` or an explicit Omni import,
the general plugin leaves the core runtime unchanged.

Omni shares Sonar's package version. Its additional Python dependencies are in
`requirements/omni.txt`. Some models require further platform-specific packages.

## Serving limits

Set these environment variables before starting the server:

```sh
export APHRODITE_OMNI_SERVER_MAX_VIDEO_JOBS=32
export APHRODITE_OMNI_SERVER_MAX_REFERENCE_UPLOAD_BYTES=67108864
```

`MAX_VIDEO_JOBS` limits queued and running jobs submitted through `POST /v1/videos`
in each API process. A full queue returns HTTP 503 with `Retry-After: 5`.
This limit does not apply to the synchronous video or image endpoints.

`MAX_REFERENCE_UPLOAD_BYTES` limits each video reference upload read by the
standard upload helpers to 64 MiB by default. Model-specific limits can be lower.
Control uploads have a separate limit. These checks do not limit the entire HTTP
request body, decoded media size, or media downloaded from URLs. Configure request
body limits and outbound network restrictions at the deployment boundary.

Video job metadata is held in memory and is lost on restart. Use one API process
per replica until shared job storage is available. To expire generated files:

```sh
export APHRODITE_OMNI_SERVER_STORAGE__FILE_TTL=86400
export APHRODITE_OMNI_SERVER_STORAGE__TTL_SWEEP_INTERVAL=300
```

File expiry does not remove job metadata. Production validation is still pending
for long-running workloads, distributed execution, and clean wheel installs.
