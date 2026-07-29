---
title: Security
description: Reduce risk when Sonar serves untrusted clients or models.
---

Treat model files, remote code, media URLs, prompt content, and adapters as
untrusted inputs unless you control their source.

## Limit network exposure

Use `--api-key` for bearer authentication. Place public deployments behind a
proxy that provides TLS, rate limits, and request-size limits.

Do not expose diagnostic or administrative routes to the public network.
Restrict `/metrics`, profiling routes, adapter management, cache reset, sleep,
weight update, and collective RPC routes.

## Pin model revisions

Use `--revision` and `--tokenizer-revision` with immutable commit identifiers.
Review a model repository before you set `--trust-remote-code`.

Remote model code executes in the server process. It receives the same file,
network, and device access as Sonar.

## Restrict multimodal media

Allow only required remote hosts:

```bash
aphrodite serve MODEL \
  --allowed-media-domains images.example.com assets.example.net
```

Leave `--allowed-local-media-path` empty for a public service. If local files
are required, point it at a dedicated read-only directory. Never allow `/`.

The proxy should limit media size and request duration. Remote media fetching
can otherwise access slow or unexpectedly large resources.

## Protect adapter management

Dynamic LoRA loading changes server state and reads a model path. Put the load
and unload routes on an administrative network. Use an allowlist for adapter
locations.

## Isolate tenants

Use separate replicas when tenants require strong cache or resource isolation.
Shared prefix caches, queues, logs, and metrics can reveal workload
characteristics even when response data remains separate.

## Container controls

Run without privileged mode. Mount model storage read-only when possible. Give
the process a writable cache directory and no access to unrelated host paths.

The container still needs GPU device access and shared memory. These
requirements do not require broad host filesystem access.

## Log handling

Prompts and generated text can contain secrets. Check log settings before
production use. Restrict access to traces and request logs, and apply a
retention period.
