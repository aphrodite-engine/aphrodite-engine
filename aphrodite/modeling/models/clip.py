"""Minimal implementation of CLIP models.

Includes:
- CLIPVisionModel for use inside VLMs
- CLIPTextModel for standalone text embeddings
"""
from collections.abc import Iterable
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import CLIPTextConfig, CLIPVisionConfig

from aphrodite.attention.layer import MultiHeadAttention
from aphrodite.distributed import divide, get_tensor_model_parallel_world_size
from aphrodite.modeling.layers.activation import get_act_fn
from aphrodite.modeling.layers.linear import (ColumnParallelLinear,
                                              QKVParallelLinear,
                                              RowParallelLinear)
from aphrodite.modeling.model_loader.weight_utils import default_weight_loader
from aphrodite.modeling.models.interfaces import SupportsQuant, default_pooling_type
from aphrodite.quantization import QuantizationConfig

from .vision import VisionEncoderInfo, resolve_visual_encoder_outputs
from aphrodite.modeling.layers.pooler import (DispatchPooler, Pooler,
                                              EmbeddingPoolerHead,
                                              PoolingParamsUpdate,
                                              get_prompt_lens,
                                              get_prompt_token_ids,
                                              build_output)
from aphrodite.modeling.models.utils import AutoWeightsLoader, WeightsMapper
from aphrodite.config import AphroditeConfig
from aphrodite.common.sequence import IntermediateTensors


class CLIPEncoderInfo(VisionEncoderInfo[CLIPVisionConfig]):

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        return self.get_patch_grid_length()**2 + 1

    def get_image_size(self) -> int:
        return self.vision_config.image_size

    def get_patch_size(self) -> int:
        return self.vision_config.patch_size

    def get_patch_grid_length(self) -> int:
        image_size, patch_size = self.get_image_size(), self.get_patch_size()
        assert image_size % patch_size == 0
        return image_size // patch_size


# Adapted from https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/clip/modeling_clip.py#L164 # noqa
class CLIPVisionEmbeddings(nn.Module):

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        assert self.image_size % self.patch_size == 0

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions,
                                               self.embed_dim)
        self.register_buffer("position_ids",
                             torch.arange(self.num_positions).expand((1, -1)),
                             persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

        self.attn = MultiHeadAttention(self.num_heads_per_partition,
                                       self.head_dim, self.scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """Input shape: Batch x Time x Channel"""

        qkv_states, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)
        out = self.attn(query_states, key_states, value_states)
        attn_output, _ = self.out_proj(out)

        return attn_output, None


class CLIPMLP(nn.Module):

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.fc1")
        self.fc2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.fc2")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class CLIPEncoderLayer(nn.Module):

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = CLIPAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config,
                           quant_config=quant_config,
                           prefix=f"{prefix}.mlp")
        self.layer_norm2 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self
    attention layers. Each layer is a [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(config=config,
                             quant_config=quant_config,
                             prefix=f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(num_hidden_layers)
        ])

    def forward(
        self, inputs_embeds: torch.Tensor, return_all_hidden_states: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)
        # If we have multiple feature sample layers, we return all hidden
        # states in order and grab the ones we need by index.
        if return_all_hidden_states:
            return hidden_states_pool
        return hidden_states


class CLIPVisionTransformer(nn.Module):

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)

        # NOTE: This typo of "layrnorm" is not fixed on purpose to match
        # the original transformers code and name of the model weights.
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.encoder = CLIPEncoder(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        # If possible, skip post_layernorm to conserve memory
        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers

        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(embed_dim,
                                               eps=config.layer_norm_eps)
        else:
            self.post_layernorm = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        feature_sample_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        return_all_hidden_states = feature_sample_layers is not None

        # Produces either the last layer output or all of the hidden states,
        # depending on if we have feature_sample_layers or not
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=return_all_hidden_states)

        # Handle post-norm (if applicable) and stacks feature layers if needed
        encoder_outputs = resolve_visual_encoder_outputs(
            encoder_outputs, feature_sample_layers, self.post_layernorm,
            self.config.num_hidden_layers)

        return encoder_outputs


class CLIPVisionModel(nn.Module, SupportsQuant):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vision_model = CLIPVisionTransformer(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            require_post_norm=require_post_norm,
            prefix=f"{prefix}.vision_model")

    def forward(
        self,
        pixel_values: torch.Tensor,
        feature_sample_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:
        return self.vision_model(pixel_values, feature_sample_layers)

    @property
    def device(self):
        return next(self.parameters()).device

    # (TODO) Add prefix argument for filtering out weights to be loaded
    #        ref: https://github.com/aphrodite-project/aphrodite/pull/7186#discussion_r1734163986
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is not needed in CLIPVisionModel
            if (name.startswith("vision_model.post_layernorm")
                    and self.vision_model.post_layernorm is None):
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("vision_model.encoder.layers"):
                layer_idx = int(name.split(".")[3])
                if layer_idx >= layer_count:
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ===============================
# CLIP Text (embedding) support
# ===============================

class CLIPTextEmbeddings(nn.Module):

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings,
                                               embed_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return embeddings


class CLIPTextEncoderLayer(nn.Module):

    def __init__(
        self,
        config: CLIPTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size

        # Reuse vision attention/MLP shapes; CLIP text uses same hidden sizes
        vision_like = CLIPVisionConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            hidden_act=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            attention_dropout=config.attention_dropout,
            initializer_range=config.initializer_range,
            initializer_factor=getattr(config, "initializer_factor", 1.0),
            projection_dim=getattr(config, "projection_dim", config.hidden_size),
            image_size=224,
            patch_size=32,
        )

        self.self_attn = CLIPAttention(
            vision_like,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(
            vision_like,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CLIPTextEncoder(nn.Module):

    def __init__(
        self,
        config: CLIPTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            CLIPTextEncoderLayer(config=config,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size,
                                             eps=config.layer_norm_eps)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


 

"""Minimal implementation of CLIP models.

Includes:
- CLIPVisionModel for use inside VLMs
- CLIPTextModel for standalone text embeddings
"""
from collections.abc import Iterable
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import CLIPTextConfig, CLIPVisionConfig

from aphrodite.attention.layer import MultiHeadAttention
from aphrodite.distributed import divide, get_tensor_model_parallel_world_size
from aphrodite.modeling.layers.activation import get_act_fn
from aphrodite.modeling.layers.linear import (ColumnParallelLinear,
                                              QKVParallelLinear,
                                              RowParallelLinear)
from aphrodite.modeling.model_loader.weight_utils import default_weight_loader
from aphrodite.modeling.models.interfaces import SupportsQuant, default_pooling_type
from aphrodite.quantization import QuantizationConfig

from .vision import VisionEncoderInfo, resolve_visual_encoder_outputs
from aphrodite.modeling.layers.pooler import DispatchPooler, Pooler
from aphrodite.modeling.models.utils import AutoWeightsLoader, WeightsMapper
from aphrodite.common.sequence import IntermediateTensors


class CLIPEncoderInfo(VisionEncoderInfo[CLIPVisionConfig]):

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        return self.get_patch_grid_length()**2 + 1

    def get_image_size(self) -> int:
        return self.vision_config.image_size

    def get_patch_size(self) -> int:
        return self.vision_config.patch_size

    def get_patch_grid_length(self) -> int:
        image_size, patch_size = self.get_image_size(), self.get_patch_size()
        assert image_size % patch_size == 0
        return image_size // patch_size


# Adapted from https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/clip/modeling_clip.py#L164 # noqa
class CLIPVisionEmbeddings(nn.Module):

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        assert self.image_size % self.patch_size == 0

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions,
                                               self.embed_dim)
        self.register_buffer("position_ids",
                             torch.arange(self.num_positions).expand((1, -1)),
                             persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

        self.attn = MultiHeadAttention(self.num_heads_per_partition,
                                       self.head_dim, self.scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """Input shape: Batch x Time x Channel"""

        qkv_states, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)
        out = self.attn(query_states, key_states, value_states)
        attn_output, _ = self.out_proj(out)

        return attn_output, None


class CLIPMLP(nn.Module):

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.fc1")
        self.fc2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.fc2")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class CLIPEncoderLayer(nn.Module):

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = CLIPAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config,
                           quant_config=quant_config,
                           prefix=f"{prefix}.mlp")
        self.layer_norm2 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self
    attention layers. Each layer is a [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(config=config,
                             quant_config=quant_config,
                             prefix=f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(num_hidden_layers)
        ])

    def forward(
        self, inputs_embeds: torch.Tensor, return_all_hidden_states: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)
        # If we have multiple feature sample layers, we return all hidden
        # states in order and grab the ones we need by index.
        if return_all_hidden_states:
            return hidden_states_pool
        return hidden_states


class CLIPVisionTransformer(nn.Module):

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)

        # NOTE: This typo of "layrnorm" is not fixed on purpose to match
        # the original transformers code and name of the model weights.
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.encoder = CLIPEncoder(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        # If possible, skip post_layernorm to conserve memory
        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers

        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(embed_dim,
                                               eps=config.layer_norm_eps)
        else:
            self.post_layernorm = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        feature_sample_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        return_all_hidden_states = feature_sample_layers is not None

        # Produces either the last layer output or all of the hidden states,
        # depending on if we have feature_sample_layers or not
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=return_all_hidden_states)

        # Handle post-norm (if applicable) and stacks feature layers if needed
        encoder_outputs = resolve_visual_encoder_outputs(
            encoder_outputs, feature_sample_layers, self.post_layernorm,
            self.config.num_hidden_layers)

        return encoder_outputs


class CLIPVisionModel(nn.Module, SupportsQuant):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vision_model = CLIPVisionTransformer(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            require_post_norm=require_post_norm,
            prefix=f"{prefix}.vision_model")

    def forward(
        self,
        pixel_values: torch.Tensor,
        feature_sample_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:
        return self.vision_model(pixel_values, feature_sample_layers)

    @property
    def device(self):
        return next(self.parameters()).device

    # (TODO) Add prefix argument for filtering out weights to be loaded
    #        ref: https://github.com/aphrodite-project/aphrodite/pull/7186#discussion_r1734163986
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is not needed in CLIPVisionModel
            if (name.startswith("vision_model.post_layernorm")
                    and self.vision_model.post_layernorm is None):
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("vision_model.encoder.layers"):
                layer_idx = int(name.split(".")[3])
                if layer_idx >= layer_count:
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ===============================
# CLIP Text (embedding) support
# ===============================

class CLIPTextEmbeddings(nn.Module):

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings,
                                               embed_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return embeddings


class CLIPTextEncoderLayer(nn.Module):

    def __init__(
        self,
        config: CLIPTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size

        # Reuse vision attention/MLP shapes; CLIP text uses same hidden sizes
        vision_like = CLIPVisionConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            hidden_act=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            attention_dropout=config.attention_dropout,
            initializer_range=config.initializer_range,
            initializer_factor=getattr(config, "initializer_factor", 1.0),
            projection_dim=getattr(config, "projection_dim", config.hidden_size),
            image_size=224,
            patch_size=32,
        )

        self.self_attn = CLIPAttention(
            vision_like,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(
            vision_like,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CLIPTextEncoder(nn.Module):

    def __init__(
        self,
        config: CLIPTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            CLIPTextEncoderLayer(config=config,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size,
                                             eps=config.layer_norm_eps)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


@default_pooling_type("LAST")
class CLIPTextModel(nn.Module, SupportsQuant):
    config_class = CLIPTextConfig
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    is_pooling_model = True

    def __init__(
        self,
        *,
        aphrodite_config: AphroditeConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hf_config = aphrodite_config.model_config.hf_config
        if hasattr(hf_config, "text_config"):
            text_config: CLIPTextConfig = hf_config.text_config
        else:
            text_config = hf_config  # type: ignore[assignment]

        # Treat as encoder-only
        setattr(text_config, "is_causal", False)

        self.embeddings = CLIPTextEmbeddings(text_config)
        self.encoder = CLIPTextEncoder(config=text_config,
                                       quant_config=aphrodite_config.quant_config,
                                       prefix=f"{prefix}.encoder")
        
        pooler_config = aphrodite_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = DispatchPooler({
            "encode": Pooler.for_encode(pooler_config),
            "embed": CLIPTextEosPooler(eos_token_id=text_config.eos_token_id),
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids=input_ids,
                                            position_ids=positions)
        return self.encoder(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Accept both CLIPModel and CLIPTextModel weight prefixes
        for name, loaded_weight in weights:
            # Keep only text branch
            if ".vision_model" in name or name.startswith("visual."):
                continue

            if name.startswith("model."):
                name = name[len("model."):]
            if name.startswith("text_model."):
                name = name[len("text_model."):]

            # Map final_layer_norm
            if name.startswith("final_layer_norm."):
                name = name.replace("final_layer_norm.", "encoder.final_layer_norm.")

            # Self-attention q/k/v -> qkv stacking
            if ".self_attn." in name:
                if any(x in name for x in ("q_proj", "k_proj", "v_proj")):
                    for param_name, weight_name, shard_id in (
                        ("qkv_proj", "q_proj", "q"),
                        ("qkv_proj", "k_proj", "k"),
                        ("qkv_proj", "v_proj", "v"),
                    ):
                        if weight_name not in name:
                            continue
                        mapped = name.replace(weight_name, param_name)
                        if not mapped.startswith("encoder."):
                            mapped = mapped.replace("encoder.layers", "encoder.layers")
                            mapped = "encoder." + mapped if not mapped.startswith("encoder.") else mapped
                        if mapped in params_dict:
                            param = params_dict[mapped]
                            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                            weight_loader(param, loaded_weight, shard_id)
                            loaded_params.add(mapped)
                        break
                    continue

            # Prefix encoder.* for other encoder weights if missing
            if name.startswith("encoder.") or name.startswith("embeddings."):
                mapped = name
            elif name.startswith("layers."):
                mapped = f"encoder.{name}"
            else:
                mapped = name

            if mapped in params_dict:
                param = params_dict[mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped)
            else:
                # Ignore unrelated weights (e.g., projections, logit_scale)
                continue

        # Fallback to auto loader for any remaining straightforward params
        remaining = [(n, w) for n, w in weights if n not in loaded_params]
        if remaining:
            try:
                loader = AutoWeightsLoader(self)
                loaded_params.update(loader.load_weights(remaining))
            except Exception:
                pass

        return loaded_params


class CLIPTextEosPooler(Pooler):

    def __init__(self, eos_token_id: int) -> None:
        super().__init__()
        self.eos_token_id = eos_token_id
        self.head = EmbeddingPoolerHead()

    def get_supported_tasks(self):
        return {"embed"}

    def get_pooling_updates(self, task):
        return PoolingParamsUpdate(requires_token_ids=True)

    def _pool_one(self, hidden_states: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        # Find first position of eos_token_id (handles pad after EOS)
        pos = (token_ids == self.eos_token_id).int().argmax(dim=-1).item()
        return hidden_states[pos]

    def forward(self, hidden_states, pooling_metadata):
        token_ids_list = get_prompt_token_ids(pooling_metadata)
        if isinstance(hidden_states, list):
            pooled = [self._pool_one(h, t) for h, t in zip(hidden_states, token_ids_list)]
        else:
            # hidden_states: (sum_len, hidden)
            prompt_lens = get_prompt_lens(hidden_states, pooling_metadata)
            offset = 0
            pooled = []
            for length, token_ids in zip(prompt_lens, token_ids_list):
                segment = hidden_states[offset:offset+length]
                pooled.append(self._pool_one(segment, token_ids))
                offset += int(length)
        pooled = self.head(pooled, pooling_metadata)
        return build_output(pooled)

