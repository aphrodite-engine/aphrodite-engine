---
title: Upgrade Sonar
description: Upgrade Sonar while preserving a working rollback.
---

Test an upgrade with the production model and configuration before you replace
a running replica.

## Record the current deployment

Save the Sonar version, commit, model revision, tokenizer revision, PyTorch
version, CUDA or ROCm version, and complete server arguments.

Export a small set of representative requests with expected output properties.
Exact generated text can vary because of floating-point and batching changes.

## Upgrade a wheel installation

```bash
uv pip install --upgrade aphrodite-engine --torch-backend=cu130
```

For a nightly build:

```bash
uv pip install --upgrade aphrodite-engine \
  --extra-index-url https://sonar.dphn.ai/nightly
```

Use a fresh virtual environment when core dependencies change. Keep the old
environment until the replacement passes validation.

## Upgrade a source checkout

```bash
git fetch origin
git switch main
git pull --ff-only
uv pip install --editable . --torch-backend=cu130
```

Regenerate the CMake preset when Python, CUDA, the compiler, or the virtual
environment changes. Rebuild native extensions after CMake or `csrc/` changes.

## Validate compatibility

Check startup warnings for renamed flags and deprecated values. Compare the
generated [server argument reference](/reference/server-arguments/) with the
saved command.

Run chat, streaming, structured output, tool calling, LoRA, and multimodal
requests that the service supports. Compare latency and memory with the saved
baseline.

## Roll back

Keep the old image or virtual environment and its model revision. Remove the
failed replacement from traffic, then restore the old replica.

Do not reuse a compilation cache across incompatible PyTorch or CUDA builds
when a rollback produces loader or symbol errors. Use separate cache keys for
each build stack.
