# coding=utf-8
# Copyright 2024 The vLLM team.
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
from loguru import logger
from torch import nn
from transformers import Gemma2Config

from aphrodite.attention import Attention, AttentionMetadata
from aphrodite.common.config import CacheConfig, LoRAConfig
from aphrodite.common.sequence import IntermediateTensors, PoolerOutput
from aphrodite.compilation.decorators import support_torch_compile
from aphrodite.distributed import (get_pp_group,
                                   get_tensor_model_parallel_world_size)
from aphrodite.modeling.layers.activation import GeluAndMul
from aphrodite.modeling.layers.layernorm import GemmaRMSNorm
from aphrodite.modeling.layers.linear import (MergedColumnParallelLinear,
                                              QKVParallelLinear,
                                              RowParallelLinear)
from aphrodite.modeling.layers.logits_processor import LogitsProcessor
from aphrodite.modeling.layers.pooler import Pooler, PoolingType
from aphrodite.modeling.layers.rotary_embedding import get_rope
from aphrodite.modeling.layers.sampler import Sampler, SamplerOutput
from aphrodite.modeling.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from aphrodite.modeling.model_loader.weight_utils import default_weight_loader
from aphrodite.modeling.pooling_metadata import PoolingMetadata
from aphrodite.modeling.sampling_metadata import SamplingMetadata
from aphrodite.quantization import QuantizationConfig

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)


class Gemma2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        hidden_activation: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        if not (hidden_act == hidden_activation == "gelu_pytorch_tanh"):
            raise ValueError(
                "Gemma2 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`gelu_pytorch_tanh`.")
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma2Attention(nn.Module):

    def __init__(self,
                 layer_idx: int,
                 config: Gemma2Config,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 max_position_embeddings: int,
                 rope_theta: float,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 attn_logits_soft_cap: Optional[float] = None) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=True,
        )

        # FIXME(woosuk): While Gemma 2 uses sliding window attention for every
        # odd layer, vLLM currently ignores it and uses global attention for
        # all layers.
        use_sliding_window = (layer_idx % 2 == 1
                              and config.sliding_window is not None)
        del use_sliding_window  # Unused.
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              logits_soft_cap=attn_logits_soft_cap)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma2DecoderLayer(nn.Module):

    def __init__(
        self,
        layer_idx: int,
        config: Gemma2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma2Attention(
            layer_idx=layer_idx,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=config.attn_logit_softcapping,
        )
        self.hidden_size = config.hidden_size
        self.mlp = Gemma2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size,
                                                      eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size,
                                                       eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Gemma2Model(nn.Module):

    def __init__(
        self,
        config: Gemma2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma2DecoderLayer(int(prefix.split(".")[
                -1]), config, cache_config, quant_config),
            prefix=f"{prefix}.layers")
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Normalize the embedding by sqrt(hidden_size)
        # The normalizer's data type should be downcasted to the model's
        # data type such as bfloat16, not float32.
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = self.config.hidden_size**0.5
        self.register_buffer("normalizer", torch.tensor(normalizer))
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            hidden_states *= self.normalizer

            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            for (param_name, shard_name, shard_id) in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            logger.warning(
                "Some weights are not initialized from checkpoints: "
                f"{unloaded_params}")


class Gemma2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    # Gemma does not apply LoRA to the embedding layer.
    embedding_modules = {}
    embedding_padding_modules = []

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]

    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Gemma2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        del lora_config  # Unused.
        super().__init__()
        self.config = config
        # currently all existing Gemma models have `tie_word_embeddings` enabled
        assert config.tie_word_embeddings
        self.quant_config = quant_config
        self.model = Gemma2Model(config, cache_config, quant_config)
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.final_logit_softcapping)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.model.embed_tokens, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        loader.load_weights(weights)


class Gemma2EmbeddingModel(nn.Module, SupportsPP):
    """
    A model that uses Gemma2 with additional embedding functionalities.
    This class encapsulates the Gemma2Model and provides an interface for
    embedding operations and customized pooling functions.
    Attributes:
        model: An instance of Gemma2Model used for forward operations.
        _pooler: An instance of Pooler used for pooling operations.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model = Gemma2Model(**kwargs)
        self._pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(input_ids, positions, kv_caches, attn_metadata,
                          intermediate_tensors, inputs_embeds)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self.model.load_weights(weights)
