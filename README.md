<h1 align="center">
Breathing Life into Language
</h1>


![aphrodite](/assets/art/aphrodite_light.png)

Aphrodite is an inference engine that optimizes the serving of HuggingFace-compatible models at scale. Built on vLLM's Paged Attention technology, it delivers high-performance model inference for multiple concurrent users. Developed through a collaboration between [PygmalionAI](https://pygmalion.chat) and [Ruliad](https://ruliad.co), Aphrodite serves as the backend engine powering both organizations' chat platforms and API infrastructure.

Aphrodite builds upon and integrates the exceptional work from [various projects](#acknowledgements), primarily [vLLM](https://vllm.ai).

> [!CAUTION]
> This project is not, and will never be, associated with any cryptocurrencies. Please do not fall for scams.


## 🔥 News
(09/2024) v0.6.1 is here. You can now load FP16 models in FP2 to FP7 quant formats, to achieve extremely high throughput and save on memory.

(09/2024) v0.6.0 is released, with huge throughput improvements, many new quant formats (including fp8 and llm-compressor), asymmetric tensor parallel, pipeline parallel and more! Please check out the [exhaustive documentation](https://aphrodite.pygmalion.chat) for the User and Developer guides.

## Features

- Continuous Batching
- Efficient K/V management with [PagedAttention](https://vllm.ai) from vLLM
- Optimized CUDA kernels for improved inference
- Quantization support via AQLM, AWQ, Bitsandbytes, GGUF, GPTQ, QuIP#, Smoothquant+, SqueezeLLM, Marlin, FP2-FP12, and more
- Distributed inference
- 8-bit KV Cache for higher context lengths and throughput, at both FP8 E5M3 and E4M3 formats
- Support for modern samplers such as DRY, XTC, and more


## Quickstart

Install the engine:
```sh
pip install -U aphrodite-engine
```

Then launch a model:

```sh
aphrodite run meta-llama/Meta-Llama-3.1-8B-Instruct
```

This will create a [OpenAI](https://platform.openai.com/docs/api-reference/)-compatible API server that can be accessed at port 2242 of the localhost. You can plug in the API into a UI that supports OpenAI, such as [SillyTavern](https://github.com/SillyTavern/SillyTavern).

Please refer to the [documentation](https://aphrodite.pygmalion.chat) for the full list of arguments and flags you can pass to the engine.

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
    --api-keys "sk-empty"
```

This will pull the Aphrodite Engine image (~8GiB download), and launch the engine with the Llama-3.1-8B-Instruct model at port 2242.

## Requirements

- Operating System: Linux, Windows (Needs building from source)
- Python: 3.8 to 3.12

#### Build Requirements:
- CUDA >= 11

For supported devices, see [here](https://aphrodite.pygmalion.chat/pages/quantization/support-matrix.html). Generally speaking, all semi-modern GPUs are supported - down to Pascal (GTX 10xx, P40, etc.) We also support AMD GPUs, Intel CPUs and GPUs, Google TPU, and AWS Inferentia.




### Notes

1. By design, Aphrodite takes up 90% of your GPU's VRAM. If you're not serving an LLM at scale, you may want to limit the amount of memory it takes up. You can do this in the API example by launching the server with the `--gpu-memory-utilization 0.6` (0.6 means 60%), or `--single-user-mode` to only allocate as much memory as needed for a single sequence.

2. You can view the full list of commands by running `aphrodite run --help`.

## Acknowledgements
Aphrodite Engine would have not been possible without the phenomenal work of other open-source projects. Credits go to:
- [vLLM](https://github.com/vllm-project/vllm) (CacheFlow)
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

## Contributing
Everyone is welcome to contribute. You can support the project by opening Pull Requests for new features, fixes, or general UX improvements.
