"""Inference-only Snowflake Arctic model."""
from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
from loguru import logger
from torch import nn

from aphrodite.attention import Attention
from aphrodite.common.config import AphroditeConfig, CacheConfig
from aphrodite.common.sequence import IntermediateTensors
from aphrodite.compilation.decorators import support_torch_compile
from aphrodite.distributed import (get_pp_group,
                                   get_tensor_model_parallel_rank,
                                   get_tensor_model_parallel_world_size,
                                   tensor_model_parallel_all_reduce)
from aphrodite.modeling.layers.activation import SiluAndMul
from aphrodite.modeling.layers.fused_moe import fused_experts, fused_topk
from aphrodite.modeling.layers.layernorm import RMSNorm
from aphrodite.modeling.layers.linear import (MergedColumnParallelLinear,
                                              QKVParallelLinear,
                                              ReplicatedLinear,
                                              RowParallelLinear)
from aphrodite.modeling.layers.logits_processor import LogitsProcessor
from aphrodite.modeling.layers.rotary_embedding import get_rope
from aphrodite.modeling.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from aphrodite.modeling.model_loader.weight_utils import default_weight_loader
from aphrodite.modeling.sampling_metadata import SamplingMetadata
from aphrodite.modeling.utils import set_weight_attrs
from aphrodite.platforms import current_platform
from aphrodite.quantization import QuantizationConfig
from aphrodite.quantization.deepspeedfp import (DeepSpeedFPConfig,
                                                DeepSpeedFPParameter)
from aphrodite.transformers_utils.configs.arctic import ArcticConfig

from .interfaces import SupportsPP, SupportsQuant
from .utils import (extract_layer_index, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


class ArcticMLP(nn.Module):

    def __init__(self,
                 config: ArcticConfig,
                 expert_id: int = -1,
                 is_residual_mlp: bool = False,
                 quant_config: Optional[QuantizationConfig] = None,
                 reduce_results: bool = True,
                 prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.expert_id = expert_id

        self.ffn_dim = config.intermediate_size if not is_residual_mlp \
            else self.hidden_size

        self.w13 = MergedColumnParallelLinear(self.hidden_size,
                                              [self.ffn_dim] * 2,
                                              bias=False,
                                              quant_config=quant_config)
        self.w2 = RowParallelLinear(self.ffn_dim,
                                    self.hidden_size,
                                    bias=False,
                                    reduce_results=reduce_results,
                                    quant_config=quant_config)
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states):
        gate_up, _ = self.w13(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.w2(hidden_states)
        return hidden_states


class ArcticMoE(nn.Module):
    """
    Model-parallel implementation of Arctic MoE Layer.
    """

    def __init__(self,
                 config: ArcticConfig,
                 tp_size: Optional[int] = None,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 reduce_results: bool = True,
                 prefix: str = ""):
        super().__init__()

        layer_id = extract_layer_index(prefix)
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.layer_id = layer_id
        self.top_k = config.num_experts_per_tok
        self.intermediate_size = config.intermediate_size // self.tp_size

        self.is_moe_layer = (layer_id + 1) % config.moe_layer_frequency == 0
        self.is_quant = isinstance(quant_config, DeepSpeedFPConfig)
        self.reduce_results = reduce_results
        # Some other parameters
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        if not self.is_moe_layer:
            self.mlp = ArcticMLP(config,
                                 quant_config=quant_config,
                                 reduce_results=reduce_results,
                                 prefix=f"{prefix}.mlp")
        else:
            self.gate = ReplicatedLinear(self.hidden_size,
                                         self.num_experts,
                                         bias=False,
                                         params_dtype=self.params_dtype,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.gate")
            if self.is_quant:
                self.ws = DeepSpeedFPParameter(
                    torch.Size((self.num_experts, 2 * self.intermediate_size,
                                self.hidden_size)),
                    params_dtype=params_dtype,
                    quant_config=quant_config,
                )
                self.w2s = DeepSpeedFPParameter(
                    torch.Size((self.num_experts, self.hidden_size,
                                self.intermediate_size)),
                    params_dtype=params_dtype,
                    quant_config=quant_config,
                )
            else:
                self.ws = nn.Parameter(
                    torch.empty(self.num_experts,
                                2 * self.intermediate_size,
                                self.hidden_size,
                                device=current_platform.device_type,
                                dtype=self.params_dtype))
                self.w2s = nn.Parameter(
                    torch.empty(self.num_experts,
                                self.hidden_size,
                                self.intermediate_size,
                                device=current_platform.device_type,
                                dtype=self.params_dtype))
            set_weight_attrs(self.ws, {
                "weight_loader": self.weight_loader,
            })
            set_weight_attrs(self.w2s, {
                "weight_loader": self.weight_loader,
            })

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.ds_dequantize() if self.is_quant else param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("w1.weight"):
            param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("w3.weight"):
            param_data[expert_id,
                       shard_size:2 * shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("w2.weight"):
            param_data[expert_id, :, :] = loaded_weight[:, shard]
        if self.is_quant:
            param.ds_quantize_(param_data)

    def local_moe_fused(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        do_normalize = self.top_k > 1
        topk_weights, topk_ids = fused_topk(hidden_states,
                                            router_logits,
                                            self.top_k,
                                            renormalize=do_normalize)
        # topk_ids: (num_tokens, k)
        if self.is_quant:
            if 2 * num_tokens <= self.num_experts:
                # If much fewer tokens than experts, use selective dequantize.
                ws_dequantized = self.ws.ds_selective_dequantize(
                    topk_ids.flatten())
                w2s_dequantized = self.w2s.ds_selective_dequantize(
                    topk_ids.flatten())
                # We gathered the experts to the tokens so update the mapping.
                topk_ids = torch.arange(
                    0,
                    topk_ids.numel(),
                    device=topk_ids.device,
                ).reshape(topk_ids.shape)
            else:
                ws_dequantized = self.ws.ds_dequantize()
                w2s_dequantized = self.w2s.ds_dequantize()

        final_hidden_states = fused_experts(
            hidden_states,
            ws_dequantized if self.is_quant else self.ws,
            w2s_dequantized if self.is_quant else self.w2s,
            topk_weights,
            topk_ids,
            inplace=True)
        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        if self.is_moe_layer:
            final_hidden_states = self.local_moe_fused(hidden_states)
        else:
            final_hidden_states = self.mlp(hidden_states)
        return final_hidden_states


class ArcticAttention(nn.Module):

    def __init__(
        self,
        config: ArcticConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(self.hidden_size,
                                          self.head_dim,
                                          self.total_num_heads,
                                          self.total_num_kv_heads,
                                          bias=False,
                                          quant_config=quant_config)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=int(self.rope_theta),
            is_neox_style=True,
        )

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class ArcticDecoderLayer(nn.Module):

    def __init__(
        self,
        config: ArcticConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        layer_idx = extract_layer_index(prefix)
        is_moe_layer = (layer_idx + 1) % config.moe_layer_frequency == 0
        self.use_residual = config.use_residual and is_moe_layer
        self.self_attn = ArcticAttention(config,
                                         cache_config,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.self_attn")
        self.block_sparse_moe = ArcticMoE(
            config,
            quant_config=quant_config,
            reduce_results=(not self.use_residual),
            prefix=f"{prefix}.block_sparse_moe",
        )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        if self.use_residual:
            self.residual_layernorm = RMSNorm(config.hidden_size,
                                              eps=config.rms_norm_eps)
            self.residual_mlp = ArcticMLP(config,
                                          is_residual_mlp=True,
                                          reduce_results=False,
                                          prefix=f"{prefix}.residual_mlp")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual_input = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = residual_input + hidden_states

        residual_attn = hidden_states
        if self.use_residual:
            hidden_states = self.residual_layernorm(hidden_states)
            hidden_states = self.residual_mlp(hidden_states)
            residual_mlp = hidden_states
            hidden_states = self.post_attention_layernorm(residual_input)
            hidden_states = self.block_sparse_moe(hidden_states)
            hidden_states = residual_mlp + hidden_states
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
            hidden_states = residual_attn + hidden_states
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.block_sparse_moe(hidden_states)
            hidden_states = residual_attn + hidden_states
        return hidden_states


@support_torch_compile
class ArcticModel(nn.Module):

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__()

        config = aphrodite_config.model_config.hf_config
        cache_config = aphrodite_config.cache_config
        quant_config = aphrodite_config.quant_config

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=self.vocab_size)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: ArcticDecoderLayer(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.layers")
        self._attn_implementation = config._attn_implementation
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        hidden_states = self.norm(hidden_states)
        return hidden_states


class ArcticForCausalLM(nn.Module, SupportsPP, SupportsQuant):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__()
        config = aphrodite_config.model_config.hf_config
        quant_config = aphrodite_config.quant_config
        self.config = config
        self.model = ArcticModel(aphrodite_config=aphrodite_config,
                                 prefix=maybe_prefix(prefix, "model"))
        self.vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(
            self.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.unpadded_vocab_size = config.vocab_size
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
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
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
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
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        mlp_params_mapping: List[Tuple[str, str, int]] = []
        expert_params_mapping: List[Tuple[str, str, int]] = []
        num_layers = self.config.num_hidden_layers

        for layer in range(num_layers):
            mlp_params_mapping.append(
                (f"layers.{layer}.residual_mlp.w13.weight",
                 f"layers.{layer}.residual_mlp.w1.weight", 0))
            mlp_params_mapping.append(
                (f"layers.{layer}.residual_mlp.w13.weight",
                 f"layers.{layer}.residual_mlp.w3.weight", 1))
            if layer % 2 == 0:
                # MLP layers
                mlp_params_mapping.append(
                    (f"layers.{layer}.block_sparse_moe.mlp.w13.weight",
                     f"layers.{layer}.block_sparse_moe.mlp.w1.weight", 0))
                mlp_params_mapping.append(
                    (f"layers.{layer}.block_sparse_moe.mlp.w13.weight",
                     f"layers.{layer}.block_sparse_moe.mlp.w3.weight", 1))
            else:
                # MoE layers
                for expert_id in range(self.config.num_local_experts):
                    expert_params_mapping.append(
                        ("ws", f"experts.{expert_id}.w1.weight", expert_id))
                    expert_params_mapping.append(
                        ("w2s", f"experts.{expert_id}.w2.weight", expert_id))
                    expert_params_mapping.append(
                        ("ws", f"experts.{expert_id}.w3.weight", expert_id))

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        logger.info(
            "It will take ~10 minutes loading from the 16-bit weights. "
            "Alternatively, use the prequantized 8-bit weights of arctic "
            "and set load-format to `sharded_state` will accelerate loading.")
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
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
                for param_name, weight_name, shard_id in mlp_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    for param_name, weight_name, shard_id \
                            in expert_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param,
                                      loaded_weight,
                                      weight_name,
                                      expert_id=shard_id)
                        break
                    else:
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]

                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
