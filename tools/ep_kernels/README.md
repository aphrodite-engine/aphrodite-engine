# Expert parallel kernels

Large-scale cluster-level expert parallel, as described in the [DeepSeek-V3 Technical Report](http://arxiv.org/abs/2412.19437), is an efficient way to deploy sparse MoE models with many experts. However, such deployment requires many components beyond a normal Python package, including system package support and system driver support. It is impossible to bundle all these components into a Python package.

Here we break down the requirements in 2 steps:

1. Build and install the Python libraries (both [pplx-kernels](https://github.com/ppl-ai/pplx-kernels) and [DeepEP](https://github.com/deepseek-ai/DeepEP)), including necessary dependencies like NVSHMEM. This step does not require any privileged access. Any user can do this.
2. Configure NVIDIA driver to enable IBGDA. This step requires root access, and must be done on the host machine.

2 is necessary for multi-node deployment.

All scripts accept a positional argument as workspace path for staging the build, defaulting to `$(pwd)/ep_kernels_workspace`.

## NCCL version requirement (CUDA 13+)

DeepEPv2 uses the NCCL GIN backend, which requires NCCL >= 2.30.4 at both
compile time and runtime. PyTorch 2.11 pins `nvidia-nccl-cu13==2.28.9`, so
you need to override it.

**With uv** (recommended):

```bash
echo "nvidia-nccl-cu13>=2.30.4" > /tmp/nccl-override.txt
export UV_OVERRIDE=/tmp/nccl-override.txt
uv pip install aphrodite-engine
```

**With pip**:

```bash
pip install aphrodite-engine
pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
```

The override must happen before building DeepEP and remain in place at runtime.

## Usage

```bash
# for hopper
TORCH_CUDA_ARCH_LIST="9.0" bash install_python_libraries.sh
# for blackwell
TORCH_CUDA_ARCH_LIST="10.0" bash install_python_libraries.sh
```

Additional step for multi-node deployment:

```bash
sudo bash configure_system_drivers.sh
sudo reboot # Reboot is required to load the new driver
```
