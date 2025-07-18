# adapted from https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/persimmon/modeling_persimmon.py
# Copyright 2023 The vLLM team.
# Copyright 2023 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only persimmon model compatible with HuggingFace weights."""
from typing import Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import PersimmonConfig

from aphrodite.attention import Attention
from aphrodite.common.config import AphroditeConfig, CacheConfig
from aphrodite.common.sequence import IntermediateTensors
from aphrodite.compilation.decorators import support_torch_compile
from aphrodite.distributed import (get_pp_group,
                                   get_tensor_model_parallel_world_size)
from aphrodite.modeling.layers.activation import get_act_fn
from aphrodite.modeling.layers.linear import (ColumnParallelLinear,
                                              QKVParallelLinear,
                                              RowParallelLinear)
from aphrodite.modeling.layers.logits_processor import LogitsProcessor
from aphrodite.modeling.layers.rotary_embedding import get_rope
from aphrodite.modeling.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from aphrodite.modeling.model_loader.weight_utils import default_weight_loader
from aphrodite.modeling.sampling_metadata import SamplingMetadata
from aphrodite.quantization import QuantizationConfig

from .interfaces import SupportsPP
from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


class PersimmonMLP(nn.Module):

    def __init__(self,
                 config: PersimmonConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(config.hidden_size,
                                                  config.intermediate_size,
                                                  quant_config=quant_config)
        self.dense_4h_to_h = RowParallelLinear(config.intermediate_size,
                                               config.hidden_size,
                                               quant_config=quant_config)
        self.act = get_act_fn(config.hidden_act)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states, _ = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.dense_4h_to_h(hidden_states)
        return hidden_states


class PersimmonAttention(nn.Module):

    def __init__(self,
                 config: PersimmonConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.config = config
        tensor_parallel_world_size = get_tensor_model_parallel_world_size()

        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tensor_parallel_world_size
        self.head_dim = self.hidden_size // self.total_num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.is_causal = True

        assert (self.head_dim * self.total_num_heads) == self.hidden_size
        assert self.total_num_heads % tensor_parallel_world_size == 0

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            bias=True,
            quant_config=quant_config,
        )
        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
        )
        self.is_qk_layernorm = config.qk_layernorm

        if self.is_qk_layernorm:
            self.q_layernorm = nn.LayerNorm(self.head_dim)
            self.k_layernorm = nn.LayerNorm(self.head_dim)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
        )
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              scale=self.scaling,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [seq_length, hidden_size] -> [seq_length, num_heads, head_dim]
        seq_length = x.shape[0]
        return x.view(seq_length, self.num_heads, self.head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [seq_length, num_heads, head_dim] -> [seq_length, hidden_size]
        seq_length = x.shape[0]
        return x.view(seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # [seq_length, 3 x hidden_size]
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)

        if self.is_qk_layernorm:
            # [seq_length, num_heads, head_dim]
            q = self._split_heads(q)
            k = self._split_heads(k)

            q = self.q_layernorm(q)
            k = self.k_layernorm(k)

            q = self._merge_heads(q)
            k = self._merge_heads(k)

        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.dense(attn_output)
        return output


class PersimmonDecoderLayer(nn.Module):

    def __init__(self,
                 config: PersimmonConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PersimmonAttention(config=config,
                                            cache_config=cache_config,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.self_attn")
        self.mlp = PersimmonMLP(config, quant_config=quant_config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                     eps=config.layer_norm_eps)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = hidden_states
        return outputs


@support_torch_compile
class PersimmonModel(nn.Module):

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__()

        config = aphrodite_config.model_config.hf_config
        cache_config = aphrodite_config.cache_config
        quant_config = aphrodite_config.quant_config

        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: PersimmonDecoderLayer(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.layers")
        self.final_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states = layer(positions, hidden_states)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]

            if "query_key_value" in name:
                # copy from aphrodite/model_executor/models/bloom.py
                # NOTE: Persimmon's fused QKV's output_dim has the shape of
                # (num_heads * 3 * head_size), while the
                # required shape is (3 * num_heads * head_size).
                # Thus, we need weight conversion.
                output_dim = getattr(param, "output_dim", None)
                num_heads = self.config.num_attention_heads
                if output_dim is not None:
                    loaded_weight_shape = loaded_weight.shape
                    loaded_weight = loaded_weight.view(
                        loaded_weight_shape[:output_dim] + (num_heads, 3, -1) +
                        loaded_weight_shape[output_dim + 1:])
                    loaded_weight = loaded_weight.transpose(
                        output_dim, output_dim + 1)
                    loaded_weight = loaded_weight.reshape(loaded_weight_shape)

            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class PersimmonForCausalLM(nn.Module, SupportsPP):

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__()
        config = aphrodite_config.model_config.hf_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.model = PersimmonModel(aphrodite_config=aphrodite_config,
                                    prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      bias=False)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
