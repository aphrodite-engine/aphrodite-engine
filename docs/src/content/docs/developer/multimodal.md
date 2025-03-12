---
title: Adding Multimodal Capabilities
---

This guide will walk you through the steps needed to extend the capabilities of an Aphrodite model so that it accepts multimodal inputs.

:::tip
See also: [Adding a New Model](/developer/adding-model).
:::

## Step 1: Update the base Aphrodite model
We assume that you have already created a new model by following the steps in the [Adding a New Model](/developer/adding-model) guide. If not, please do so before proceeding.

1. Implement the `aphrodite.modeling.models.interfaces.SupportsMultiModal` interface:

```py
from aphrodite.modeling.models.interfaces import SupportsMultiModal  # [!code ++]

class YourModelForImage2Seq(nn.Module):  # [!code --]
class YourModelForImage2Seq(nn.Module, SupportsMultiModal):  # [!code ++]
```

:::tip
The model class does not have to be named `*ForCausalLM`. Check out the[ HuggingFace Transformers documentation](https://huggingface.co/docs/transformers/model_doc/auto#multimodal) for some examples.
:::

2. If you haven't done so already, reserve a keyword parameter in [`forward()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward) for each input tensor that corresponds to a multi-modal input, as shown in the following example:

```py
  def forward(
      self,
      input_ids: torch.Tensor,
      positions: torch.Tensor,
      kv_caches: List[torch.Tensor],
      attn_metadata: AttentionMetadata,
      pixel_values: torch.Tensor, # [!code ++]
  ) -> SamplerOutput:
```

## Step 2: Register input mappers
For each modality type that the model accepts as input, decorate the model class with `aphrodite.multimodal.MultiModalRegistry.register_input_mapper`. This decorator accepts a function that maps multi-modal inputs to the keyword arguments you have previously defined in `forward()`.

```py
from aphrodite.modeling.models.interfaces import SupportsMultiModal
from aphrodite.multimodal import MULTIMODAL_REGISTRY  # [!code ++]

@MULTIMODAL_REGISTRY.register_image_input_mapper()  # [!code ++]
class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```

A default mapper is available for each modality in the core Aphrodite library. This input mapper will be used if you do not provide your own function.

:::tip
See also: [Input Processing Pipeline](/developer/input-processing).
:::

## Step 3: Register maximum number of multi-modal tokens

For each modality type that the model accepts as input, calculate the maximum possible number of tokens and register it via `aphrodite.inputs.InputRegistry.register_max_multimodal_tokens`.

```py
from aphrodite.inputs import INPUT_REGISTRY  # [!code ++]
from aphrodite.modeling.models.interfaces import SupportsMultiModal
from aphrodite.multimodal import MULTIMODAL_REGISTRY

@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(<your_calculation>)
@INPUT_REGISTRY.register_dummy_data(<your_dummy_data_factory>)  # [!code ++]
class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```

Here are some examples:

- Static feature size: [LLaVA-1.5 Model](https://github.com/PygmalionAI/aphrodite-engine/tree/main/aphrodite/modeling/models/llava.py)
- Dynamic feature size: [LLaVA-NeXT Model](https://github.com/PygmalionAI/aphrodite-engine/tree/main/aphrodite/modeling/models/llava_next.py)

:::tip
See also: [Input Processing Pipeline](/developer/input-processing).
:::

## Step 4: (Optional) Register dummy data
During startup, dummy data is passed to the Aphrodite model to allocate memory. This only consists of text input by default, which may not be applicable to multi-modal models. In such cases, you can define your own dummy data by registering a factory method via `aphrodite.inputs.InputRegistry.register_dummy_data`.

```py
from aphrodite.inputs import INPUT_REGISTRY
from aphrodite.modeling.models.interfaces import SupportsMultiModal
from aphrodite.multimodal import MULTIMODAL_REGISTRY

@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(<your_calculation>)
@INPUT_REGISTRY.register_dummy_data(<your_dummy_data_factory>)  # [!code ++]
class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```

:::tip
The dummy data should have the maximum possible number of multi-modal tokens, as described in the previous step.

Here are some examples:
- Static feature size: [LLaVA-1.5 Model](https://github.com/PygmalionAI/aphrodite-engine/tree/main/aphrodite/modeling/models/llava.py)
- Dynamic feature size: [LLaVA-NeXT Model](https://github.com/PygmalionAI/aphrodite-engine/tree/main/aphrodite/modeling/models/llava_next.py)
:::

:::tip
See also: [Input Processing Pipeline](/developer/input-processing).
:::

## Step 5: (Optional) Register input processor
Sometimes, there's a need to process inputs at the `aphrodite.AphroditeEngine` level before they are passed to the model executor. This is often due to the fact that unlike implementations in HuggingFace Transformers, the reshaping and/or expansion of multi-modal embeddings needs to take place outside model's `forward()` method. You can register input processors via  `aphrodite.inputs.InputRegister.register_input_processor`.

```py
from aphrodite.inputs import INPUT_REGISTRY
from aphrodite.modeling.models.interfaces import SupportsMultiModal
from aphrodite.multimodal import MULTIMODAL_REGISTRY

@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(<your_calculation>)
@INPUT_REGISTRY.register_dummy_data(<your_dummy_data_factory>)
@INPUT_REGISTRY.register_input_processor(<your_input_processor>)  # [!code ++]
class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```

A common use case of input processors is inserting placeholder tokens to leverage the Aphrodite framework for attention mask generation. Here are some examples:

- Insert static number of image tokens: [LLaVA-1.5 Model](https://github.com/PygmalionAI/aphrodite-engine/tree/main/aphrodite/modeling/models/llava.py)
- Insert dynamic number of image tokens: [LLaVA-NeXT Model](https://github.com/PygmalionAI/aphrodite-engine/tree/main/aphrodite/modeling/models/llava_next.py)

:::tip
See also: [Input Processing Pipeline](/developer/input-processing).
:::