---
title: Supported Hardware
---

The table below shows the hardware support matrix for Aphrodite. Google TPU doesn't support most.

| Quantization Method   | Volta | Turing | Ampere | Ada | Hopper | AMD GPU | Intel GPU | x86 CPU | AWS Inferentia | Google TPU |
| --------------------- | ----- | ------ | ------ | --- | ------ | ------- | --------- | ------- | -------------- | ---------- |
| AQLM                  | ✅    | ✅     | ✅     | ✅  | ✅     | ❌      | ❌        | ❌      | ❌             | ❌         |
| AWQ                   | ✅    | ✅     | ✅     | ✅  | ✅     | ❌      | ❌        | ✅      | ❌             | ❌         |
| GPTQ                  | ✅    | ✅     | ✅     | ✅  | ✅     | ✅      | ❌        | ❌      | ❌             | ❌         |
| Marlin (GPTQ/AWQ/FP8) | ❌    | ❌     | ✅     | ✅  | ✅     | ❌      | ❌        | ❌      | ❌             | ❌         |
| INT8 (W8A8)           | ❌    | ✅     | ✅     | ✅  | ✅     | ✅      | ❌        | ❌      | ❌             | ❌         |
| FP8 (W8A8)            | ❌    | ❌     | ❌     | ✅  | ✅     | ✅      | ❌        | ❌      | ❌             | ❌         |
| BitsAndBytes          | ✅    | ✅     | ✅     | ✅  | ✅     | ✅      | ❌        | ❌      | ❌             | ❌         |
| DeepspeedFP           | ✅    | ✅     | ✅     | ✅  | ✅     | ❌      | ❌        | ❌      | ❌             | ❌         |
| QuantLLM (FP2-FP7)    | ❌    | ❌     | ✅     | ✅  | ✅     | ❌      | ❌        | ❌      | ❌             | ❌         |
| GGUF                  | ✅    | ✅     | ✅     | ✅  | ✅     | ❌      | ❌        | ❌      | ❌             | ❌         |
| SqueezeLLM            | ✅    | ✅     | ✅     | ✅  | ✅     | ❌      | ❌        | ❌      | ❌             | ❌         |
| QuIP#                 | ❌    | ❌     | ✅     | ✅  | ✅     | ❌      | ❌        | ❌      | ❌             | ❌         |
| EETQ                  | ❌    | ✅     | ✅     | ✅  | ✅     | ✅      | ❌        | ❌      | ❌             | ❌         |
| QQQ                   | ❌    | ❌     | ✅     | ✅  | ✅     | ❌      | ❌        | ❌      | ❌             | ❌         |
| VPTQ                  | ✅    | ✅     | ✅     | ✅  | ✅     | ✅      | ❌        | ❌      | ❌             | ❌         |
| TPU-INT8              | ❌    | ❌     | ❌     | ❌  | ❌     | ❌      | ❌        | ❌      | ❌             | ✅         |
| Neuron-Quant          | ❌    | ❌     | ❌     | ❌  | ❌     | ❌      | ❌        | ❌      | ✅             | ❌         |
| ModelOpt FP8          | ❌    | ❌     | ❌     | ✅  | ✅     | ❌      | ❌        | ❌      | ❌             | ❌         |
| FPGEMM-FP8            | ❌    | ❌     | ❌     | ✅  | ✅     | ❌      | ❌        | ❌      | ❌             | ❌         |
| Experts-INT8          | ❌    | ❌     | ✅     | ✅  | ✅     | ✅      | ❌        | ❌      | ❌             | ❌         |

## Notes

- Volta refers to SM 7.0, Turing to SM 7.5, Ampere to SM 8.0/8.6, Ada to SM 8.9, and Hopper to SM 9.0. Blackwell is not supported as of v0.6.7.
- ✅ indicates that the hardware supports the quantization method.
- ❌ indicates that the hardware does not support the quantization method.

Please note that this compatibility chart may be subject to change as Aphrodite continues to evolve and expand its support for different hardware platforms and quantization methods.

For the most up-to-date information on hardware support and quantization methods, please check the [quantization directory](https://github.com/PygmalionAI/aphrodite-engine/tree/main/aphrodite/quantization/) in the source code.
