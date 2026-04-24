# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F

from aphrodite.compilation.decorators import support_torch_compile
from aphrodite.config import AphroditeConfig
from aphrodite.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from aphrodite.model_executor.layers.deepseek_v4_attention import (
    DeepseekV4Indexer,
    DeepseekV4MLAModules,
    DeepseekV4MultiHeadLatentAttentionWrapper,
)
from aphrodite.model_executor.layers.fused_moe import FusedMoE, GateLinear, SharedFusedMoE
from aphrodite.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from aphrodite.model_executor.layers.layernorm import RMSNorm
from aphrodite.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from aphrodite.model_executor.layers.logits_processor import LogitsProcessor
from aphrodite.model_executor.layers.quantization import QuantizationMethods
from aphrodite.model_executor.layers.quantization.fp8 import Fp8Config
from aphrodite.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from aphrodite.model_executor.layers.rotary_embedding import get_rope
from aphrodite.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from aphrodite.model_executor.model_loader.weight_utils import default_weight_loader
from aphrodite.model_executor.models.deepseek_v2 import DeepseekV2MLP
from aphrodite.platforms import current_platform
from aphrodite.sequence import IntermediateTensors
from aphrodite.utils.multi_stream_utils import AuxStreamType

from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)


class DeepseekV4FP8Config(Fp8Config):
    """FP8 config that routes MoE layers to MXFP4 quantization.

    DeepSeek V4 checkpoints use FP8 for linear/attention layers but
    MXFP4 for MoE expert weights. This config inherits standard FP8
    behavior and overrides only the MoE dispatch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_scale_e8m0: bool = True

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "deepseek_v4_fp8"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None) -> QuantizationMethods | None:
        if not (isinstance(hf_quant_cfg, dict) and hf_quant_cfg.get("quant_method") in ("fp8", "deepseek_v4_fp8")):
            return None
        model_type = getattr(hf_config, "model_type", None)
        if model_type == "deepseek_v4" or user_quant == "deepseek_v4_fp8":
            return "deepseek_v4_fp8"
        return None

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return Mxfp4MoEMethod(layer.moe_config)
        return super().get_quant_method(layer, prefix)

    def is_mxfp4_quant(self, prefix, layer):
        return isinstance(layer, FusedMoE)


class DeepseekV4MoE(nn.Module):
    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        config = aphrodite_config.model_config.hf_config
        quant_config = aphrodite_config.quant_config
        self.prefix = prefix

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size
        assert config.n_routed_experts % self.tp_size == 0

        self.n_routed_experts = config.n_routed_experts
        self.n_local_experts = config.n_routed_experts // self.tp_size
        self.n_activated_experts = config.num_experts_per_tok
        self.experts_start_idx = self.tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            bias=False,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers

        if is_hash_moe:
            # hash MoE doesn't use e_score_correction_bias
            # Use randint instead of empty to avoid garbage values causing
            # invalid memory access in dummy mode (--load-format="dummy")
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "sqrtsoftplus"),
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
        )

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        org_shape = hidden_states.shape

        _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            fused_moe_out = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
                input_ids=input_ids,
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            fused_moe_out = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )

        shared_output, final_hidden_states = fused_moe_out

        if self.shared_experts is None:
            assert shared_output is None

        if self.shared_experts is not None:
            assert shared_output is not None
            final_hidden_states += shared_output

        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)

        return final_hidden_states.view(org_shape)


class DeepseekV4Attention(nn.Module):
    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        config = aphrodite_config.model_config.hf_config
        quant_config = aphrodite_config.quant_config
        layer_id = extract_layer_index(prefix)

        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert self.n_heads % tp_size == 0

        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        # NOTE(zyongye) Compress ratio can't be 0
        # we do this for because MTP layer is not included
        # in the compress ratio list
        if layer_id < config.num_hidden_layers:
            self.compress_ratio = max(1, config.compress_ratios[layer_id])
        else:
            self.compress_ratio = 1
        self.eps = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings

        # Padded to min 64 heads for FlashMLA, initialized to -inf
        # (no sink effect). Weight loading fills the first n_local_heads slots.
        padded_heads = max(self.n_local_heads, 64)
        self.attn_sink = nn.Parameter(
            torch.full((padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )

        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )
        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = config.quantization_config["scale_fmt"]

        self.rope_parameters = config.rope_scaling

        # Initialize rotary embedding BEFORE DeepseekV4MLAModules (which needs it)
        rope_parameters = config.rope_parameters
        rope_parameters["rope_theta"] = config.compress_rope_theta if self.compress_ratio > 1 else config.rope_theta
        # TODO(yifan): double check this!
        # rope_parameters["rope_type"] = "deepseek_yarn"
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn" if config.rope_parameters.get("apply_yarn_scaling", True) else "deepseek_llama_scaling"
            )
        rope_parameters["mscale"] = 0  # Disable mscale
        rope_parameters["mscale_all_dim"] = 0  # Disable mscale
        rope_parameters["is_deepseek_v4"] = True
        rope_parameters["rope_dim"] = self.rope_head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=False,
            dtype=config.torch_dtype,
        )

        self.indexer = None
        if self.compress_ratio == 4:
            # Only C4A uses sparse attention and hence has indexer.
            self.indexer = DeepseekV4Indexer(
                aphrodite_config,
                config=config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=aphrodite_config.cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
            )

        mla_modules = DeepseekV4MLAModules(
            aphrodite_config=aphrodite_config,
            fused_wqa_wkv=self.fused_wqa_wkv,
            q_norm=self.q_norm,
            wq_b=self.wq_b,
            kv_norm=self.kv_norm,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            rotary_emb=self.rotary_emb,
            indexer=self.indexer,
            indexer_rotary_emb=self.rotary_emb,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream=aux_stream,
        )
        self.mla_attn = DeepseekV4MultiHeadLatentAttentionWrapper(
            hidden_size=self.hidden_size,
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.softmax_scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            v_head_dim=self.head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.head_dim,
            o_lora_rank=self.o_lora_rank,
            mla_modules=mla_modules,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            cache_config=aphrodite_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ):
        return self.mla_attn(positions, hidden_states, llama_4_scaling)


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        aphrodite_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream] | None = None,
    ):
        super().__init__()
        config = aphrodite_config.model_config.hf_config
        self.hidden_size = config.hidden_size

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = DeepseekV4Attention(
            aphrodite_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream=aux_stream_dict.get(AuxStreamType.Attention) if aux_stream_dict is not None else None,
        )
        self.ffn = DeepseekV4MoE(aphrodite_config, prefix=f"{prefix}.ffn")

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        # Lazy import to avoid top-level tilelang dependency.
        # Registers both torch.ops.aphrodite.mhc_pre and mhc_post,
        # so hc_post() doesn't need its own import.
        import aphrodite.model_executor.layers.mhc  # noqa: F401

        post_mix, res_mix, layer_input = torch.ops.aphrodite.mhc_pre(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
        )
        return layer_input, post_mix, res_mix

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return torch.ops.aphrodite.mhc_post(x, residual, post, comb)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


@support_torch_compile
class DeepseekV4Model(nn.Module):
    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__()

        config = aphrodite_config.model_config.hf_config
        quant_config = aphrodite_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        aux_stream_list = [torch.cuda.Stream() for _ in range(1)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
        }

        self.device = current_platform.device_type
        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            aphrodite_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                aphrodite_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_dict=self.aux_stream_dict,
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)

        self.hc_head_fn = nn.Parameter(
            torch.empty(
                self.hc_mult,
                self.hc_dim,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(
                self.hc_mult,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )

        # Pre-hc_head residual stream buffer for the MTP draft. Stable
        # address (outside the cudagraph pool) so the copy_ in forward()
        # refreshes it correctly across captured shapes.
        self._mtp_hidden_buffer = torch.empty(
            aphrodite_config.scheduler_config.max_num_batched_tokens,
            self.hc_dim,
            dtype=aphrodite_config.model_config.dtype,
            device=self.device,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.embed_input_ids(input_ids)
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(
                hidden_states,
                positions,
                input_ids,
            )

        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        hidden_states = hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                        success = weight_loader(
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
                    loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )


@torch.compile(backend=current_platform.simple_compile_backend)
def hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    x = hidden_states
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


class DeepseekV4ForCausalLM(nn.Module):
    model_cls = DeepseekV4Model

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex={
            # Routed MoE expert scales: experts.N.wX.scale -> .weight_scale
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            # Everything else (FP8 linear + shared experts): .scale -> .weight_scale_inv
            re.compile(r"\.scale$"): ".weight_scale_inv",
        },
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".attn.compressor.": ".attn.mla_attn.compressor.",
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__()

        config = aphrodite_config.model_config.hf_config
        self.config = config

        self.model = self.model_cls(aphrodite_config=aphrodite_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
