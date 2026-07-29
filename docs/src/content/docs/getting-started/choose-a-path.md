---
title: Choose a path
description: Select the Sonar workflow that matches your task and hardware.
---

Start with the smallest deployment that meets the requirement. Add distributed
execution after one GPU reaches a measured memory or performance limit.

## Run inference from Python

Use the [`LLM` API](/guides/offline-inference/) for batch jobs, evaluations, and
applications that run in one Python process. It loads the model when the object
is created and releases resources when the process exits.

## Provide an HTTP API

Use the [OpenAI-compatible server](/serving/openai/) for applications that
already use an OpenAI client. The same server provides health checks, metrics,
tokenization, pooling, and optional compatibility routes.

Read [Production deployment](/deployment/production/) before you expose the
server outside a trusted network.

## Minimize latency

Use the [optimization procedure](/deployment/optimization/) to establish a
baseline. Then test [speculative decoding](/features/speculative-decoding/) at
the expected concurrency.

## Maximize throughput

Tune the scheduler before you add replicas. Use data parallelism when one model
replica fits and the service needs more request capacity.

## Serve a model that does not fit

Read [Choose parallelism](/deployment/parallelism/). Tensor parallelism is the
usual first choice inside one server. Pipeline parallelism can span layer
stages when tensor parallel communication is unsuitable.

## Use constrained hardware

Select a quantized checkpoint from the
[quantization matrix](/reference/quantization/). Set the required context
length and reduce concurrent sequences. Use
`--max-model-len auto` when the usable context is unknown.

## Find a complete example

The [deployment recipes](/deployment/recipes/) provide starting commands for a
consumer GPU, a datacenter GPU, an NVLink server, a multi-node deployment, AMD
ROCm, Jetson Thor, and CPU development.
