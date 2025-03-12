---
title: Adding a new model
---

This document provides a high-level guide on integrating a Hugging Face transformers model into Aphrodite Engine.

The complexity of adding a new model depends heavily on the model's architecture. The process is straightforward if the model shares a similar architecture with an existing model in Aphrodite. However, for models that include new operators (e.g. a new attention mechanism), the process can be a bit more complex.

By default, Aphrodite models do not support multi-modal inputs. We have separate guide for enabling that after implementing the model here.

:::tip
If you're having problems implementing the model, feel free to open an issue on the GitHub repo. We'll be happy to help if we can!
:::

## Step 0: Fork the Aphrodite Repository
Start by forking our [GitHub repository](https://github.com/PygmalionAI/aphrodite-engine) and the build it from source. This gives you the ability to modify the source code and test your model.

## Step 1: Bring your model code
Clone the PyTorch model code from the Hugging Face Transformers repository and put it into the `aphrodite/modeling/models` directory. For instance, Aphrodite's `OPT` model was adapted from Hugging Face's [modeling_opt.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py) file.

## Step 2: Rewrite the `forward` methods
Next, you need to rewrite the [`forward()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward) method of your model by following these steps:

1. Remove any unnecessary code, such as the code used for training.
2. Change the input parameters:

```diff
  def forward(
      self,
      input_ids: torch.Tensor,
-     attention_mask: Optional[torch.Tensor] = None,
-     position_ids: Optional[torch.LongTensor] = None,
-     past_key_values: Optional[List[torch.FloatTensor]] = None,
-     inputs_embeds: Optional[torch.FloatTensor] = None,
-     labels: Optional[torch.LongTensor] = None,
-     use_cache: Optional[bool] = None,
-     output_attentions: Optional[bool] = None,
-     output_hidden_states: Optional[bool] = None,
-     return_dict: Optional[bool] = None,
- ) -> Union[Tuple, CausalLMOutputWithPast]:
+     positions: torch.Tensor,
+     kv_caches: List[torch.Tensor],
+     attn_metadata: AttentionMetadata,
+ ) -> Optional[SamplerOutput]:
```
3. Update the code by considering that `input_ids` and `positions` are now flattened tensors.
4. Replace the attention operating with `Attention` imported from `aphrodite.attention`.

### Step 3: Implement Tensor Parallelism and Quantization support
To implement TP, substitute your model's linear and embedding layers with their tensor-parallel versions. For the embedding layer, you can simply replace [`torch.nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding) with `VocabParallelEmbedding`. For the output LM head, you can use `ParallelLMHead`. When it comes to the linear layers, we provide the following options to parallelize them:



- `ReplicatedLinear`: Replicates the inputs and weights across multiple GPUs. No memory saving.

- `RowParallelLinear`: The input tensor is partitioned along the hidden dimension. The weight matrix is partitioned along the rows (input dimension). An all-reduce operation is performed after the matrix multiplication to reduce the results. Typically used for the second FFN layer and the output linear transformation of the attention layer.

- `ColumnParallelLinear`: The input tensor is replicated. The weight matrix is partitioned along the columns (output dimension). The result is partitioned along the column dimension. Typically used for the first FFN layer and the separated QKV transformation of the attention layer in the original Transformer.

- `MergedColumnParallelLinear`: Column-parallel linear that merges multiple ColumnParallelLinear operators. Typically used for the first FFN layer with weighted activation functions (e.g., SiLU). This class handles the sharded weight loading logic of multiple weight matrices.

- `QKVParallelLinear`: Parallel linear layer for the query, key, and value projections of the multi-head and grouped-query attention mechanisms. When number of key/value heads are less than the world size, this class replicates the key/value heads properly. This class handles the weight loading and replication of the weight matrices.

Note that all the linear layers above take `linear_method` as an input. Aphrodite will set this parameter according to different quantization schemes to support weight quantization.

## Step 4: Implement the weight loading logic
You  now need to implement the `load_weights` method in your `*ForCausalLM` class. This method should load the weights from the Hugging Face's checkpoint file and assign them to the correspondingg layers in your model. Specifically, for `MergedColumnParallelLinear` and `QKVParallelLinear` layers, if the original model has separated weight matrices, you need to load the different parts separately.

## Step 5: Register your model
Finally, register your `*ForCausalLM` class to the `*_MODELS` in [aphrodite/modeling/models/__init__.py](https://github.com/PygmalionAI/aphrodite-engine/tree/main/aphrodite/modeling/models/__init__.py).


## Extra: Out-of-Tree Model Integration
We also provide a way to integrate a model without modifying the Aphrodite codebase. Step 2, 3, 4 are still required but you can skip 1 and 5.

Just add the following lines in your code:

```py
from aphrodite import ModelRegistry
from your_code import YourModelForCausalLM
ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)
```

If you're running an API server with `aphrodite run <args>`, you can wrap the endpoint with the following code:

```py
from aphrodite import ModelRegistry
from your_code import YourModelForCausalLM
ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)
import runpy
runpy.run_module("aphrodite.endpoints.openai.api_server", run_name="__main__")
```

Save the code above in a file and run it with `python your_file.py <args>`