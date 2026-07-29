---
title: Troubleshooting
description: Diagnose common Sonar startup, memory, model, and runtime failures.
---

Capture the complete command, Sonar commit, model revision, platform, Python
version, PyTorch version, and accelerator runtime before you change the system.
Keep the first error in the log. Later errors often describe cleanup failures.

## CUDA out of memory during startup

Reduce `--max-num-seqs`. Then set a smaller context or use
`--max-model-len auto`. Lower `--gpu-memory-utilization` when another process
uses the GPU.

Remember that the utilization fraction applies to total GPU capacity. Sonar
fails when that requested amount is not free.

## CUDA out of memory during requests

Check KV cache preemptions and request length. Reduce concurrency or the maximum
context. Test FP8 KV cache when the platform supports it.

An unusually large multimodal input can allocate encoder memory outside the
normal text-only workload. Set per-prompt modality limits.

## Unsupported model architecture

Read `architectures` in the model `config.json`. Compare it with the
[supported-model table](/reference/models/). Try the Transformers backend only
when the model has a compatible Transformers implementation.

Pin the model revision. A repository owner can change its architecture without
changing your command.

## Missing native extension or undefined symbol

Confirm that the wheel matches the Python and PyTorch environment. Rebuild
after a source checkout changes native code:

```bash
cmake --build --preset release --target install
```

Remove stale build output only after you record the failing command. A clean
build hides whether the incremental dependency graph is wrong.

## Server starts but requests hang

Check `/health` locally. Inspect waiting requests and GPU activity. For a
distributed job, verify that every rank joined and that firewalls permit the
selected ports.

Set NCCL debug logging for a suspected collective hang:

```bash
export NCCL_DEBUG=INFO
```

Do not leave verbose collective logs enabled during normal production traffic.

## Tool calls or reasoning text parse incorrectly

Confirm that the selected parser matches the model family. Test one known
request without sampling. Disable automatic tool choice to separate parser
behavior from model output.

## Output is slow

Separate time to first token from inter-token latency. Long time to first token
usually indicates prefill, queueing, or media preprocessing. Slow inter-token
latency points to decode kernels, communication, or low speculative acceptance.

Use the [optimization guide](/deployment/optimization/) with a fixed workload.

## Report a reproducible issue

Include a minimal command and a public model when possible. Remove secrets,
prompts, and private storage paths. Include the complete first traceback and the
result of the same request with optional features disabled.
