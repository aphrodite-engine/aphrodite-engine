---
title: Installation with ROCm
---

Aphrodite supports AMD GPUs using ROCm 6.4.

## Requirements

- Linux (or WSL on Windows)
- Python 3.9 - 3.11
- GPU: MI200 (gfx90a), MI300 (gfx942), RX 7900 Series (gfx1100)
- ROCm 6.4


## Installation with Docker

You can build Aphrodite Engine from source. First, build a docker image from the provided `Dockerfile.rocm`, then launch a container from the image.


To build Aphrodite on high-end datacenter GPUs (e.g. MI300X), run this:

```sh
DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm -t aphrodite-rocm .
```

To build Aphrodite on NAVI GPUs (e.g. RTX 7900 XTX), run this:

```sh
DOCKER_BUILDKIT=1 docker build --build-arg BUILD_FA="0" -f Dockerfile.rocm -t aphrodite-rocm .
```

Then run your image:

```sh
docker run -it \
  --network=host \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE  \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v ~/.cache/huggingface/root/.cache/huggingface \
  aphrodite-rocm \
  bash
```


## Installation from source

You can also build Aphrodite from source, but it's more complicated, so we recommend Docker.

You will need the following installed beforehand:

- [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
- [PyTorch](https://pytorch.org/get-started/locally/) with `pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.3`


Then install Triton
```sh
git clone https://github.com/triton-lang/triton
cd triton
git checkout e5be006a4
cd python
pip install ninja cmake wheel pybind11
pip install .
cd ../..
```

You may also Install [CK Flash Attention](https://github.com/ROCm/flash-attention) if needed.
This only works on gfx90a gfx942
```sh
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
git checkout b7d29fb
git submodule update --init
GPU_ARCHS="gfx90a;gfx942" python3 setup.py install
cd ..
```

:::warning
You may need to downgrade `ninja` version to 1.10.
:::

Finally, build Aphrodite:

```sh
git clone https://github.com/PygmalionAI/aphrodite-engine.git
cd aphrodite-engine
pip install -U -r requirements-rocm.txt
python setup.py develop  #  pip install -e . won't work for now
```
