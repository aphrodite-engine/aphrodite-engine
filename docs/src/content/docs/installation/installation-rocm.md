---
title: Installation with ROCm
---

Aphrodite supports AMD GPUs using ROCm 6.1.

## Requirements

- Linux (or WSL on Windows)
- Python 3.9 - 3.11
- GPU: MI200 (gfx90a), MI300 (gfx942), RX 7900 Series (gfx1100)
- ROCm 6.1


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
- [PyTorch](https://pytorch.org/get-started/locally/)
- [hipBLAS](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/install.html#install)


Then install [Triton for ROCm](http://github.com/ROCm/triton). You may also Install [CK Flash Attention](https://github.com/ROCm/flash-attention) if needed.

:::warning
You may need to downgrade `ninja` version to 1.10.
:::

Finally, build Aphrodite:

```sh
git clone https://github.com/PygmalionAI/aphrodite-engine.git
cd aphrodite-engine

chmod +x ./amdpatch.sh
./amdpatch.sh 
pip install -U -r requirements/rocm.txt
python setup.py develop  #  pip install -e . won't work for now
```
