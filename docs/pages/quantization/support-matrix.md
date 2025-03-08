---
outline: deep
---

# Supported Hardware

The table below shows the hardware support matrix for Aphrodite. Google TPU doesn't support most.

| Quantization Method   | Volta              | Turing             | Ampere             | Ada                | Hopper             | AMD GPU            | Intel GPU | x86 CPU            | AWS Inferentia     | Google TPU         |
| --------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | --------- | ------------------ | ------------------ | ------------------ |
| AQLM                  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                | :x:       | :x:                | :x:                | :x:                |
| AWQ                   | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                | :x:       | :white_check_mark: | :x:                | :x:                |
| GPTQ                  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:       | :x:                | :x:                | :x:                |
| Marlin (GPTQ/AWQ/FP8) | :x:                | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                | :x:       | :x:                | :x:                | :x:                |
| INT8 (W8A8)           | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:       | :x:                | :x:                | :x:                |
| FP8 (W8A8)            | :x:                | :x:                | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:       | :x:                | :x:                | :x:                |
| BitsAndBytes          | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:       | :x:                | :x:                | :x:                |
| DeepspeedFP           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                | :x:       | :x:                | :x:                | :x:                |
| QuantLLM (FP2-FP7)    | :x:                | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                | :x:       | :x:                | :x:                | :x:                |
| GGUF                  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                | :x:       | :x:                | :x:                | :x:                |
| SqueezeLLM            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                | :x:       | :x:                | :x:                | :x:                |
| QuIP#                 | :x:                | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                | :x:       | :x:                | :x:                | :x:                |
| EETQ                  | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:       | :x:                | :x:                | :x:                |
| QQQ                   | :x:                | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                | :x:       | :x:                | :x:                | :x:                |
| VPTQ                  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:       | :x:                | :x:                | :x:                |
| TPU-INT8              | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:       | :x:                | :x:                | :white_check_mark: |
| Neuron-Quant          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:       | :x:                | :white_check_mark: | :x:                |
| ModelOpt FP8          | :x:                | :x:                | :x:                | :white_check_mark: | :white_check_mark: | :x:                | :x:       | :x:                | :x:                | :x:                |
| FPGEMM-FP8            | :x:                | :x:                | :x:                | :white_check_mark: | :white_check_mark: | :x:                | :x:       | :x:                | :x:                | :x:                |
| Experts-INT8          | :x:                | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:       | :x:                | :x:                | :x:                |

## Notes
- Volta refers to SM 7.0, Turing to SM 7.5, Ampere to SM 8.0/8.6, Ada to SM 8.9, and Hopper to SM 9.0. Blackwell is not supported as of v0.6.7.
- :white_check_mark: indicates that the hardware supports the quantization method.
- :x: indicates that the hardware does not support the quantization method.

Please note that this compatibility chart may be subject to change as Aphrodite continues to evolve and expand its support for different hardware platforms and quantization methods.

For the most up-to-date information on hardware support and quantization methods, please check the [quantization directory](https://github.com/PygmalionAI/aphrodite-engine/tree/main/aphrodite/quantization/) in the source code.
