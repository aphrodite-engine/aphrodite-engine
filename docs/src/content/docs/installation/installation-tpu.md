---
title: Installation with TPU
---

Aphrodite supports Google Cloud TPUs using PyTorch XLA.

## Requirements
- Google Cloud TPU VM (single and multi-host)
- TPU versions: v5e, v5p, v4
- Python: 3.10

## Dockerfile
```sh
docker build -f Dockerfile.tpu -t aphrodite-tpu .
```

Then run Aphrodite:

```sh
docker run --privileged --net host --ipc=host --shm-size=16G -it aphrodite-tpu
```

## Building from Source
First, install the dependencies:

```sh
# (Recommended) Create a new conda environment.

conda create -n myenv python=3.10 -y

conda activate myenv

# Clean up the existing torch and torch-xla packages.

pip uninstall torch torch-xla -y

# Install PyTorch and PyTorch XLA.
export DATE="20240713"
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly${DATE}-cp310-cp310-linux_x86_64.whl

pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly${DATE}-cp310-cp310-linux_x86_64.whl

# Install JAX and Pallas.

pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

# Install other build dependencies.

pip install -r requirements-tpu.txt

```

Finally, build Aphrodite:

```sh
APHRODITE_TARGET_DEVICE="tpu" python setup.py develop
```

### Tips and More Information

Since TPU relies on XLA which requires static shapes, Aphrodite bucketizes the possible input shapes and compiles an XLA graph for each different shape. The compilation time may take 20~30 minutes in the first run. However, the time reduces to ~5 minutes afterwards because the XLA graphs are cached in the disk (by default, `~/.cache/aphrodite/xla_cache`). You can set where the cache is located by passing a path to the `APHRODITE_XLA_CACHE_PATH` env variable.

If you encounter this error:

```console
ImportError: libopenblas.so.0: cannot open shared object file: No such file or directory
```

Then run this:

```sh
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev
```