---
title: Debugging Tips
---

This page will walk you through debugging your issues with Aphrodite. It's recommended you do these steps before submitting an issue. It may solve your problem, or help the maintainers have a clearer idea of what's wrong.


## Debugging hang/crash issues

When an Aphrodite instance hangs or crashes unexpectedly, debugging might be quite difficult. It's certainly possible that Aphrodite is doing something that takes a long time:

- **Downloading a model**: If you don't have the model already downloaded on disk, Aphrodite will download it from Hugging Face or Modelscope first. Depending on your network connection, or whether or not you're using `hf-transfer`, this may take a while. It's recommended to download the model beforehand using [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli), then use the local path to the model. This way, we can further isolate the issue.

- **Loading the model from disk**: If the model is very large and your disk I/O is slow, it may take a long time to load. Make sure you're storing the model in fast storage; some clusters have shared filesystems across nodes, which can be slow. It's better to store the weights in local disk (unless you need to do multi-node inference, but that's a separate issue). Additionally, make sure to watch out for CPU memory usage! If the model is too large, it'll take a lot of CPU memory, and that may slow down the operating system because it'll need to swap to disk.

- **Tensor Parallel inference**: If your model is too large, you may be loading it on multiple GPUs using tensor parallelism. In this case, every process will read the whole model and split it into chunks, which makes the disk reading time even slower (proportional to the TP size). You can convert the model checkpoint to a sharded checkpoint using the [provided script](https://github.com/PygmalionAI/aphrodite-engine/tree/main/examples/save_sharded_state.py). The conversion process might takea bit, but you can load the checkpoint much faster later on, and the time will remain constant regardless of the TP size.


Now if you've taken care of the above points but the issue still persists, with CPU and GPU utilization at near zero, it's very likely that Aphrodite is stuck somewhere. Here are some general tips:

Set these environment variables:
- `export APHRODITE_LOG_LEVEL=debug` to enable more logging.
- `export CUDA_LAUNCH_BLOCKING=1` to know exactly which CUDA kernel is causing the issue.
- `export NCCL_DEBUG=TRACE` to enable more logging for NCCL.
- `export APHRODITE_TRACE_FUNCTION=1` so that all function calls in Aphrodite is recorded. Inspect these log files or attach them in your issue.

***

If your instance crashes and the error trace shows somewhere around `self.graph.replay()` in `aphrodite/worker/model_runner.py`, then it's very likely a CUDA error inside the CUDAGraph. To know the particular operation that causes the error, you can add `--enforce-eager` in the CLI, `- enforce_eager: true` in the YAML config, or `enforce_eager=True` in the `LLM` class. This will disable CUDAGraph optimization, which might make it easier to find the root cause.

Here's some other common issues that might cause freezes:

- **Incorrect hardware/driver**: GPU/CPU communication can't be established. You can run the following sanity check script to see if the GPU/CPU comms is working correctly:

```py
# Test PyTorch NCCL
import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank() % torch.cuda.device_count()
torch.cuda.set_device(local_rank)
data = torch.FloatTensor([1,] * 128).to("cuda")
dist.all_reduce(data, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
value = data.mean().item()
world_size = dist.get_world_size()
assert value == world_size, f"Expected {world_size}, got {value}"

print("PyTorch NCCL is successful!")

# Test PyTorch GLOO
gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
cpu_data = torch.FloatTensor([1,] * 128)
dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
value = cpu_data.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"

print("PyTorch GLOO is successful!")

# Test Aphrodite NCCL, with cuda graph
from aphrodite.distributed.device_communicators.pynccl import PyNcclCommunicator

pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
pynccl.disabled = False

s = torch.cuda.Stream()
with torch.cuda.stream(s):
    data.fill_(1)
    pynccl.all_reduce(data, stream=s)
    value = data.mean().item()
    assert value == world_size, f"Expected {world_size}, got {value}"

print("Aphrodite NCCL is successful!")

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(cuda_graph=g, stream=s):
    pynccl.all_reduce(data, stream=torch.cuda.current_stream())

data.fill_(1)
g.replay()
torch.cuda.current_stream().synchronize()
value = data.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"

print("Aphrodite NCCL with cuda graph is successful!")

dist.destroy_process_group(gloo_group)
dist.destroy_process_group()
```

Save this script to `test_nccl.py`.

If you're on a single node, run it like this:

```sh
NCCL_DEBUG=TRACE torchrun --nproc-per-node=8 test_nccl.py
```

Adjust `--nproc-per-node` to the number of GPUs.

On multiple nodes:

```sh
NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=2 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR test_nccl.py
```

Adjust the values as needed. Make sure `MASTER_ADDR`:

- is the correct IP address of the master node,
- is reachable from all nodes,
- is set before running the script.


If despite all this, the problem persists, then please [open a GitHub issue](https://github.com/PygmalionAI/aphrodite-engine/issues/new/choose), with a detailed description of the issue, your environment, and the logs.


:::warning
After finding the root cause of your issue, make sure to turn off all debugging env variables defined above, or start a new shell toa void being affected by the debugging settings. If you don't do this, the system might become very slow due to the debugging functionalities.
:::