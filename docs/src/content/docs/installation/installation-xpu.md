---
title: Installation with Intel XPU
---

Aphrodite supports basic model inference on Intel Datacenter GPUs.

## Requirements
- Linux
- Intel Data Center GPU
- OneAPI 2024.1

## Dockerfile
```sh
docker build -f Dockerfile.xpu -t aphrodite-xpu --shm-size=4g .
docker run -it \
  --rm \
  --network=host \
  --ipc=host \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  aphrodite-xpu
```

## Building from Source

First, install the required driver and intel OneAPI 2024.1 or later.
Second, install Python packages for Aphrodite XPU backend:

```sh
source /opt/intel/oneapi/setvars.sh
pip install -U pip
pip install -v -r requirements-xpu.txt
```

Finally, build:

```sh
APHRODITE_TARGET_DEVICE=xpu python setup.py develop
```

Currently, only FP16 data type is supported.