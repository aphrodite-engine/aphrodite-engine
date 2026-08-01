# EXL3 upstream source

The inference kernels in `exllamav3_ext` are vendored from
[`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3).

- Branch: `dev`
- Commit: `cf055324f0725c30a94a9e96927f4597ec56fbc2`
- Upstream version: `1.3.0`

Sonar supplies its own stable Torch ABI registrations and integrates the
kernels with its own model loader, tensor-parallel implementation, and engine.
Conversion kernels and the standalone ExLlama runtime are not vendored.
