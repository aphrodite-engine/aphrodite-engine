"""Inference-only Qwen3.5 model."""

import typing
from collections.abc import Callable, Iterable

import torch
from einops import rearrange
from torch import nn

from aphrodite.common.sequence import IntermediateTensors
from aphrodite.compilation.decorators import support_torch_compile
from aphrodite.config import AphroditeConfig
from aphrodite.distributed import get_ep_group, get_pp_group
from aphrodite.logger import init_logger
from aphrodite.modeling.layers.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from aphrodite.modeling.layers.linear import MergedColumnParallelLinear
from aphrodite.modeling.layers.logits_processor import LogitsProcessor
from aphrodite.modeling.layers.mamba.mamba_utils import MambaStateDtypeCalculator, MambaStateShapeCalculator
from aphrodite.modeling.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from aphrodite.modeling.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from aphrodite.multimodal import MULTIMODAL_REGISTRY
from aphrodite.quantization import QuantizationConfig
from aphrodite.transformers_utils.configs.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from aphrodite.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

from .interfaces import HasInnerState, IsHybrid, SupportsEagle3, SupportsLoRA, SupportsPP
from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_next import Qwen3NextAttention, Qwen3NextDecoderLayer, Qwen3NextGatedDeltaNet, Qwen3NextModel
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import AutoWeightsLoader, PPMissingLayer, extract_layer_index, is_pp_missing_parameter, maybe_prefix

logger = init_logger(__name__)


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[key_dim, key_dim, value_dim, value_dim],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads, num_v_heads],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        q_size = self.key_dim // self.tp_size
        k_size = self.key_dim // self.tp_size
        v_size = self.value_dim // self.tp_size
        z_size = self.value_dim // self.tp_size

        query, key, value, z = mixed_qkvz.split([q_size, k_size, v_size, z_size], dim=-1)
        b, a = mixed_ba.chunk(2, dim=-1)

        query = rearrange(query, "l (h d) -> l h d", d=self.head_k_dim)
        key = rearrange(key, "l (h d) -> l h d", d=self.head_k_dim)
        value = rearrange(value, "l (h d) -> l h d", d=self.head_v_dim)
        z = rearrange(z, "l (h d) -> l h d", d=self.head_v_dim)

        return query.contiguous(), key.contiguous(), value.contiguous(), z, b.contiguous(), a.contiguous()


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = aphrodite_config.model_config.hf_text_config
        model_config = aphrodite_config.model_config
        cache_config = aphrodite_config.cache_config
        quant_config = aphrodite_config.quant_config
        speculative_config = aphrodite_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        if config.model_type == "qwen3_5_moe_text":
            from .qwen3_next import Qwen3NextSparseMoeBlock

            self.mlp = Qwen3NextSparseMoeBlock(
                aphrodite_config=aphrodite_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.ffn_layer_scale = torch.nn.Parameter(torch.zeros(1, 1, config.hidden_size))


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "deepstack_input_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        nn.Module.__init__(self)

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = aphrodite_config.model_config.hf_text_config
        parallel_config = aphrodite_config.parallel_config
        lora_config = aphrodite_config.lora_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        lora_vocab = (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
                aphrodite_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        from .utils import make_empty_intermediate_tensors_factory, make_layers

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True
        return loaded_local_expert

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual if residual is not None else hidden_states)
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

            if deepstack_input_embeds is not None and layer_idx in range(0, len(deepstack_input_embeds)):
                hidden_states = hidden_states + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
            ("in_proj_qkvz", "in_proj_z", 3),
            ("in_proj_ba", "in_proj_b", 0),
            ("in_proj_ba", "in_proj_a", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping() if hasattr(self.config, "num_experts") else []
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        num_experts = self.config.num_experts if hasattr(self.config, "num_experts") else 0
        for name, loaded_weight in weights:
            is_fused_expert = False
            if "rotary_emb.inv_freq" in name or name.startswith("mtp."):
                continue

            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self) or name not in params_dict:
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            loaded_chunks = loaded_weight.chunk(2, dim=-2)
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_chunks[0],
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_chunks[1],
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        if success:
                            name = name_mapped
                            break
                    else:
                        if (name_mapped.endswith(".bias") or name_mapped.endswith("_bias")) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        success = param.weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once("Parameter %s not found in params_dict, skip loading", name)
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5ForCausalLMBase(nn.Module, HasInnerState, SupportsEagle3, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__()
        config = aphrodite_config.model_config.hf_text_config
        quant_config = aphrodite_config.quant_config
        lora_config = aphrodite_config.lora_config
        cache_config = aphrodite_config.cache_config

        assert not cache_config.enable_prefix_caching, "Qwen3.5 currently does not support prefix caching"

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.model = Qwen3_5Model(aphrodite_config=aphrodite_config, prefix=maybe_prefix(prefix, "model"))

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    self.unpadded_vocab_size,
                    config.hidden_size,
                    org_num_embeddings=config.vocab_size,
                    padding_size=DEFAULT_VOCAB_PADDING_SIZE
                    if not lora_config
                    else lora_config.lora_vocab_padding_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size, config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(weights)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        aphrodite_config: "AphroditeConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            aphrodite_config.model_config.dtype,
            aphrodite_config.cache_config.mamba_cache_dtype,
            aphrodite_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        aphrodite_config: "AphroditeConfig",
        use_v1: bool = True,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = aphrodite_config.parallel_config
        hf_config = aphrodite_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = aphrodite_config.speculative_config.num_speculative_tokens if aphrodite_config.speculative_config else 0
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase):
    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__(aphrodite_config=aphrodite_config, prefix=prefix)

        self.expert_weights = []
        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is not None:
            self.num_moe_layers = len(self.moe_layers)
            self.num_expert_groups = 1
            self.num_shared_experts = 0
            self.num_logical_experts = example_layer.n_logical_experts
            self.num_physical_experts = example_layer.n_physical_experts
            self.num_local_physical_experts = example_layer.n_local_physical_experts
            self.num_routed_experts = example_layer.n_routed_experts
            self.num_redundant_experts = example_layer.n_redundant_experts

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.expert_weights = []
        for layer_idx, layer in enumerate(self.moe_layers):
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = "model"):
        super(Qwen3VLForConditionalGeneration, self).__init__()
        config: Qwen3_5Config = aphrodite_config.model_config.hf_config
        quant_config = aphrodite_config.quant_config
        multimodal_config = aphrodite_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        if not multimodal_config.get_limit_per_prompt("image") and not multimodal_config.get_limit_per_prompt("video"):
            self.visual = None
        else:
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
            )

        self.language_model = Qwen3_5ForCausalLM(
            aphrodite_config=aphrodite_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.packed_modules_mapping = self.packed_modules_mapping | self.language_model.packed_modules_mapping
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = len(config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        if self.use_deepstack and self.visual is not None:
            self.deepstack_input_embeds = [
                torch.zeros(
                    aphrodite_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
        else:
            self.deepstack_input_embeds = None
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

    def recompute_mrope_positions(self, *args, **kwargs):
        raise NotImplementedError("Qwen3.5 does not support multimodal pruning (EVS).")

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        aphrodite_config: "AphroditeConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen3_5ForCausalLM.get_mamba_state_dtype_from_config(aphrodite_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        aphrodite_config: "AphroditeConfig",
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen3_5ForCausalLM.get_mamba_state_shape_from_config(aphrodite_config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["mtp."]
        if self.visual is None:
            skip_prefixes.extend(["visual."])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_aphrodite_mapper)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = "model"):
        super(Qwen3VLForConditionalGeneration, self).__init__()
        config: Qwen3_5MoeConfig = aphrodite_config.model_config.hf_config
        quant_config = aphrodite_config.quant_config
        multimodal_config = aphrodite_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        if not multimodal_config.get_limit_per_prompt("image") and not multimodal_config.get_limit_per_prompt("video"):
            self.visual = None
        else:
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
            )

        self.language_model = Qwen3_5MoeForCausalLM(
            aphrodite_config=aphrodite_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.packed_modules_mapping = self.packed_modules_mapping | self.language_model.packed_modules_mapping
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = len(config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        if self.use_deepstack and self.visual is not None:
            self.deepstack_input_embeds = [
                torch.zeros(
                    aphrodite_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
        else:
            self.deepstack_input_embeds = None
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        aphrodite_config: "AphroditeConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen3_5MoeForCausalLM.get_mamba_state_dtype_from_config(aphrodite_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        aphrodite_config: "AphroditeConfig",
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen3_5MoeForCausalLM.get_mamba_state_shape_from_config(aphrodite_config)
