---
title: Distributed Inference
---

Aphrodite supports serving LLMs on multiple GPUs (and CPUs).

## What distributed inference strategy should I use?

Before going into the details, let's make it clear when to use distributed inference and what strategies we offer. The common practice is:

- **Single GPU**: If your model fits on a single GPU, you probably don't need to use distributed inference. However, if you have two GPUs connected via P2P link, it might be faster to run it distributed.
- **Single-Node Multi-GPU**: If your model is too large to fit in a single GPU, but it can fit on a single node with multiple GPUs, you can use tensor parallelism. The tensor parallel size is the number of GPUs you want to use. For example, if you have 3 GPUs in a single node, you can set the tensor parallel size to 3. **However**, keep in mind that for quantized models, you're limited to Tensor Parallel sizes of 2, 4, and 8. If you need to run a quantized model on 3 GPUs, you may need to use Pipeline Parallelism, which is significantly slower than Tensor Parallel.
- **Multi-Node Multi-GPU**: If your model is too large to fit in a single node, you can use Tensor Parallel + Pipeline Parallel. The Tensor Parallel size is the number of GPUs within each node, and the Pipeline Parallel size is the number of nodes to use. For example, if you have 16 GPUs across 2 nodes (8 each), you can set `--tensor-parallel-size 8 --pipeline-parallel-size 2`.

In short, you should increase the number of GPUs and the number of nodes until you have enough GPU memory to hold the model. You can also opt to use quantization if needed.

After adding enough GPUs and nodes to hold the model, you can run Aphrodite. It will print logs such as ` # GPU blocks: 2048`. Multiply the number by `16` (the default block size) and you can get a rough estimate of the maximum number of tokens that can be served concurrently on the current configuration. If the number is not satisfying and you need higher throughput, you can further increase the number of GPUs or nodes.

## Details for Distributed Inference

Aphrodite supports distributed tensor parallel and pipeline parallel strategies. We implemented [Megatron-LM's tensor parallel algorithm](https://arxiv.org/pdf/1909.08053.pdf), which is also implemented by vLLM. We add extra features such as support for serving models on asymmetric number of GPUs. We manage the distributed runtime with either [Ray](https://github.com/ray-project/ray) or python-native multiprocessing. Multiprocessing is the default for single nodes, and Ray for multi-node configurations.

Multiprocessing will be used by default when not running in a Ray placement group and if there are sufficient GPUs available on the same node for the configured `tensor_parallel_size`, otherwise Ray will be used. The default can be overridden via the `LLM` class argument `distributed_executor_backend` or the `--distributed-executor-backend` CLI arg in the API server. Set it to `mp` for multiprocessing and `ray` for Ray. It's not required to install the `ray` python package if using multiprocessing.

To run multi-GPU inference with the LLM class, set the `tensor_parallel_size` to the number of GPUs you want to use. For example, on 4 GPUs you'd run:

```py
from aphrodite import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Franciso is a")
```

CLI:

```sh
aphrodite run facebook/opt-13b -tp 4
```

To use pipeline parallelism, you can run:

```sh
aphrodite run facebook/opt-13b -tp 4 -pp 2
```
This will run the model on a single node, but across 8 GPUs. A useful heuristic for the number of GPUs used is `tensor_parallel_size * pipeline_parallel_size`.

:::tip
Pipeline Parallelism is currently in beta, and only supports Llama, Mixtral, Qwen, Qwen2, and Nemotron model architectures.
:::


## Multi-Node inference

If a single node isn't enough to hold the model, you can try running it on multiple nodes. You'll have to make sure the execution environment is the exact same across all nodes, including the model path and the Python environment. The recommended way is to use docker images to ensure this, and hide the heterogeneity of the host machines via mapping them into the same docker configuration.

The first step is to start containers and organize them into a cluster. We've provided a [helper script](https://github.com/PygmalionAI/aphrodite-engine/tree/main/examples/run_cluster.sh) to get you started.

Pick a node as the head node and run this command:

```sh
bash run_cluster.sh \
  alpindale/aphrodite-openai \
  ip_of_head_node \
  --head \
  /path/to/the/huggingface/home/in/this/node
```

On the rest of the worker nodes, run this:

```sh
bash run_cluster.sh \
  alpindale/aphrodite-openai \
  ip_of_head_node \
  --worker \
  /path/to/the/huggingface/home/in/this/node
```

Then you get a ray cluster of containers. Note that you need to keep the shells running these commands alive to hold the cluster. Any shell disconnect will terminate the cluster. You can use `tmux` to help with this. In addition, please note that the argument `ip_of_head_node` should be the IP address of the head node, which is accessible by all the worker nodes. A common misunderstanding is to use the IP address of the worker, which is not correct.

Then, on any node, use `docker exec -it node /bin/bash` to enter the container, execute `ray status` to check the status of the Ray cluster. You should see the right number of nodes and GPUs.

After that, on any node, you can use Aphrodite as normal, just as you would if all the GPUs were on one node. The common practice is to set the Tensor Parallel size to the number of GPUs in each node, and the pipeline parallel size to the number of nodes. For example, if you have 16 GPUs across 2 nodes, you can set it up like this:

```sh
aphrodite run /path/to/the/model/in/the/container -tp 8 -pp 2
```

You can also use tensor parallel without pipeline parallel; just set the tp size to the total number of GPUs:

```sh
aphrodite run /path/to/the/model/in/the/container -tp 16
```

To make tensor parallel performant, you should make sure the communication between nodes is efficient, e.g. using high-speed network cards like Infiniband. To correctly set up the cluster to use Infiniband, append additional arguments like `--privileged -e NCCL_IB_HCA=mlx5` to the `run_cluster.s`h script. Please contact your system administrator for more information on how to set up the flags. One way to confirm if the Infiniband is working is to run Aphrodite with `NCCL_DEBUG=TRACE` environment variable set, e.g. `NCCL_DEBUG=TRACE aphrodite run ...` and check the logs for the NCCL version and the network used. If you find `[send] via NET/Socket` in the logs, it means NCCL uses raw TCP Socket, which is not efficient for cross-node tensor parallel. If you find `[send] via NET/IB/GDRDMA` in the logs, it means NCCL uses Infiniband with GPU-Direct RDMA, which is efficient.

:::warning
After starting the Ray cluster, you should also check the GPU<->GPU communication between the nodes. It may be non-trivial to set up. Please refer to the sanity check script above for more information. If you need to set some env variables for the communication config, you can append them to the `run_cluster.sh` script, e.g. `-e NCCL_SOCKET_IFNAME=eth0`. Note that setting env variables in the shell (e.g. `NCCL_SOCKET_IFNAME=eth0 aphrodite run ...`) only works for the processes in the same node, and not for the other nodes. Setting env variables when you create the cluster is the recommended way.
:::

:::warning
Please make sure to download the model to all nodes (with the same path), or the model is downloaded to some distributed file system that is accessible by all nodes.

When you use Hugging Face repo ID to refer to the model, you should append your Hugging Face token to the `run_cluster.sh` script, e.g. `-e HF_TOKEN=`. The recommended way is to download the model first, and then use the path to refer to the model.
:::