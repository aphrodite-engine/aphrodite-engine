
# Installation with NVIDIA

The installation process for Aphrodite is different depending on your hardware. The primary hardware for Aphrodite is NVIDIA GPUs, with x86_64 CPU architecture; we provide pre-built binaries for that combination. If you're using other types of hardware, you may need to build from source. We have provided Docker files to ease the process. You can also find detailed instructions for building manually if you cannot use Docker.


## Requirements

- Linux (or WSL on Windows)
- Python 3.8 - 3.11
- NVIDIA GPU (compute capability 6.0 or higher)

To find out what compute capability your GPU has, you can run this in a python interpreter:

```py
>>> import torch
>>> torch.cuda.get_device_capability()
(8, 6)
```

The (8, 6) indicates the GPU has a compute capability of 8.6, for example.


## Installation with pip

You can install Aphrodite using pip:

```sh
# Create a new environment (optional)
python -m venv ./aphrodite-venv --prompt "aphrodite"
# You will need to run this every time you close the terminal
source ./aphrodite-venv/bin/activate
# Install Aphrodite with CUDA 12.4
pip install -U aphrodite-engine --extra-index-url https://downloads.pygmalion.chat/whl
```

:::warning
Since our binaries are compiled with CUDA 12.4, you may need to build from source for different (major) CUDA versions. Please see below for instructions.
:::

## Building from source

You can build Aphrodite from source if needed.

```sh
git clone https://github.com/PygmalionAI/aphrodite-engine.git
cd aphrodite-engine

python -m venv ./venv --prompt "aphrodite"
source ./venv/bin/activate  # do this every time

pip install -e .  # this may take a while
```

If you don't have enough RAM, you may need to run `export MAX_JOBS=n`, where `n` is the number of jobs.

You can also use the embedded micromamba runtime for one-command installation of Aphrodite:

```sh
./update-runtime.sh
```

Afterwards, prefix every Aphrodite-related command with `./runtime.sh`. e.g.:
```sh
./runtime.sh aphrodite run -h
```

## Linux arm64/aarch64/GH200 tips

The NVIDIA GH200 comes with an ARM CPU, so you might have to look around for the binaries you need.
As of November 2024, this produced a working aphrodite build:

```sh
conda create -y -n 311 python=3.11; conda activate 311
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
conda install -y -c conda-forge cuda-nvcc_linux-aarch64=12.4 libstdcxx-ng=12
conda install -y cmake sccache

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
python -c 'import torch; print(torch.tensor(5).cuda() + 1, "torch cuda ok")'

cd aphrodite-engine

pip install nvidia-ml-py==12.555.43 protobuf==3.20.2 ninja msgspec coloredlogs portalocker pytimeparse -r requirements-common.txt
pip install --no-clean --no-deps --no-build-isolation -v .

# if you want flash attention:
cd ..
git clone https://github.com/AlpinDale/flash-attention
cd flash-attention
pip install --no-clean --no-deps --no-build-isolation -v .
```

A few places to look for aarch64 binaries if you're having trouble:

- [conda aarch64 defaults channel](https://repo.anaconda.com/pkgs/main/linux-aarch64/)
- pytorch.org hosts wheels at https://download.pytorch.org/whl and https://download.pytorch.org/whl/cuXXX (eg https://download.pytorch.org/whl/cu124). Note that `/whl/cu124` is a separate index, not a folder in `/whl`. There is also https://download.pytorch.org/whl/nightly/.
- [nvidia's NGC docker containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) come with many tools and python packages bundled
- Sometimes a project will have ARM binaries in their github build artifacts before the official releases. [example](https://github.com/pytorch/pytorch/actions/workflows/generated-linux-aarch64-binary-manywheel-nightly.yml)
- The spack package manager may be helpful for building especially tricky sources, like pytorch.

## Installation with Docker
We provide both a pre-built docker image, and a Dockerfile.

### Using the pre-built Docker image

```sh
# Run this with sudo if you run into permission issues.
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    #--env "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" \
    -p 2242:2242 \
    --ipc=host \
    alpindale/aphrodite-openai:latest \
    --model NousResearch/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --api-keys "sk-empty"
```

### Building via Docker image

```sh
DOCKER_BUILDKIT=1 docker build . \
    --target aphrodite-openai \
    --tag alpindale/aphrodite-openai
    # optionally: \
    --build-arg max_jobs=8 \
    --build-arg nvcc_threads 2
```

This Dockerfile will build for all CUDA arches, which may take hours. You can limit the arch to your GPU by adding this flag:

```sh
--build-arg torch_cuda_arch_list='8.9'  # Ada lovelace, e.g. RTX 4090
```

You can run your built image using the command in the previous section.
