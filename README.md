<h1 align="center">
Breathing Life into Language
</h1>


![aphrodite](https://raw.githubusercontent.com/PygmalionAI/aphrodite-engine/main/assets/aphrodite.png)

Aphrodite is an inference engine that optimizes the serving of HuggingFace-compatible models at scale. Built on vLLM's Paged Attention technology, it delivers high-performance model inference for multiple concurrent users. Developed through a collaboration between [PygmalionAI](https://pygmalion.chat) and [Ruliad](https://ruliad.co), Aphrodite serves as the backend engine powering both organizations' chat platforms and API infrastructure.

Aphrodite builds upon and integrates the exceptional work from [various projects](#acknowledgements), primarily [vLLM](https://vllm.ai).

## Features

- Continuous Batching
- Efficient K/V management with [PagedAttention](https://vllm.ai) from vLLM
- Optimized CUDA kernels for improved inference
- Quantization support via [AQLM](https://arxiv.org/abs/2401.06118), [AutoRound](https://arxiv.org/abs/2309.05516), [AWQ](https://arxiv.org/abs/2306.00978), [BitNet](https://arxiv.org/abs/2310.11453), [Bitsandbytes](https://arxiv.org/abs/2208.07339), [EETQ](https://github.com/NetEase-FuXi/EETQ), [GGUF](https://github.com/ggml-org/llama.cpp), [GPTQ](https://arxiv.org/abs/2210.17323), [QuIP#](https://arxiv.org/abs/2402.04396), [SqueezeLLM](https://arxiv.org/abs/2306.07629), [Marlin](https://arxiv.org/abs/2408.11743), FP2-FP12 [[1]](https://arxiv.org/abs/2401.14112) [[2]](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) [[3]](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer), [TorchAO](https://github.com/pytorch/ao), [VPTQ](https://arxiv.org/abs/2409.17066), [compressed_tensors](https://github.com/vllm-project/llm-compressor), [MXFP4](https://huggingface.co/blog/RakshitAralimatti/learn-ai-with-me), and more.
- Distributed inference
- 8-bit KV Cache for higher context lengths and throughput, at both FP8 E5M3 and E4M3 formats
- Support for modern samplers such as DRY, XTC, Mirostat, and more
- Disaggregated inference
- Speculative decoding
- Multimodal support
- Multi-LoRA support


## Quickstart

Install the engine:
```sh
pip install -U aphrodite-engine --extra-index-url https://downloads.pygmalion.chat/whl
```

Then launch a model:

```sh
aphrodite run Qwen/Qwen3-0.6B
```

If you're not serving at scale, you can append the `--single-user-mode` flag to limit memory usage.

This will create a [OpenAI](https://platform.openai.com/docs/api-reference/)-compatible API server that can be accessed at port 2242 of the localhost. You can plug in the API into a UI that supports OpenAI, such as [SillyTavern](https://github.com/SillyTavern/SillyTavern).

Please refer to the [documentation](https://aphrodite.pygmalion.chat) for the full list of arguments and flags you can pass to the engine, or simply run `aphrodite run -h` to see the full list of arguments.

You can play around with the engine in the demo here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlpinDale/misc-scripts/blob/main/Aphrodite.ipynb)

### Docker

Additionally, we provide a Docker image for easy deployment. Here's a basic command to get you started:

```sh
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    #--env "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" \
    -p 2242:2242 \
    --ipc=host \
    alpindale/aphrodite-openai:latest \
    --model NousResearch/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 8 \
    --api-key "sk-empty"
```

This will pull the Aphrodite Engine image, and launch the engine with the Llama-3.1-8B-Instruct model at port 2242.

## Requirements

- Operating System: Linux, Windows (WSL2)
- Python: 3.9 to 3.12

#### Build Requirements:
- CUDA >= 12

For supported devices, see [here](https://aphrodite.pygmalion.chat/pages/quantization/support-matrix.html). Generally speaking, all semi-modern GPUs are supported - down to Pascal (GTX 10xx, P40, etc.) We also support AMD GPUs, Intel CPUs and GPUs, Google TPU, and AWS Inferentia.




### Notes

1. By design, Aphrodite takes up 90% of your GPU's VRAM. If you're not serving an LLM at scale, you may want to limit the amount of memory it takes up. You can do this in the API example by launching the server with the `--gpu-memory-utilization 0.6` (0.6 means 60%), or `--single-user-mode` to only allocate as much memory as needed for a single sequence.

2. You can view the full list of commands by running `aphrodite run --help`.

## Acknowledgements
Aphrodite Engine would have not been possible without the phenomenal work of other open-source projects. A (non-exhaustive) list:
- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [xFormers](https://github.com/facebookresearch/xformers)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [SqueezeLLM](https://github.com/SqueezeAILab/SqueezeLLM/)
- [Exllamav2](https://github.com/turboderp/exllamav2)
- [TabbyAPI](https://github.com/theroyallab/tabbyAPI)
- [AQLM](https://github.com/Vahe1994/AQLM)
- [KoboldAI](https://github.com/henk717/KoboldAI)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Ray](https://github.com/ray-project/ray)

### Sponsors
Past and present, in alphabetical order:

- [Arc Compute](https://www.arccompute.io/)
- [Prime Intellect](https://www.primeintellect.ai/)
- [PygmalionAI](https://pygmalion.chat)
- [Ruliad AI](https://ruliad.ai)


## Contributing
Everyone is welcome to contribute. You can support the project by opening Pull Requests for new features, fixes, or general UX improvements.
