"""PyTorch MAMBA model."""
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn
from transformers import MambaConfig

from aphrodite.common.config import AphroditeConfig, CacheConfig
from aphrodite.common.sequence import IntermediateTensors
from aphrodite.common.utils import LayerBlockType
from aphrodite.distributed import get_tensor_model_parallel_world_size
from aphrodite.distributed.parallel_state import get_pp_group
from aphrodite.modeling.layers.layernorm import RMSNorm
from aphrodite.modeling.layers.logits_processor import LogitsProcessor
from aphrodite.modeling.layers.mamba.mamba_mixer import MambaMixer
from aphrodite.modeling.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from aphrodite.modeling.model_loader.weight_utils import default_weight_loader
from aphrodite.modeling.models.interfaces import (HasInnerState,
                                                  IsAttentionFree, SupportsPP,
                                                  SupportsV0Only)
from aphrodite.modeling.models.mamba_cache import (MambaCacheManager,
                                                   MambaCacheParams)
from aphrodite.modeling.sampling_metadata import SamplingMetadata
from aphrodite.quantization.base_config import QuantizationConfig

from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

KVCache = Tuple[torch.Tensor, torch.Tensor]


class MambaDecoderLayer(nn.Module):

    def __init__(self,
                 config: MambaConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 is_lora_enabled: Optional[bool] = False) -> None:
        super().__init__()
        self.config = config
        self.is_falcon_mamba = config.model_type == "falcon_mamba"
        self.is_lora_enabled = is_lora_enabled
        mixer_rms_eps = config.mixer_rms_eps if self.is_falcon_mamba else None
        self.mixer = MambaMixer(hidden_size=config.hidden_size,
                                ssm_state_size=config.state_size,
                                conv_kernel_size=config.conv_kernel,
                                intermediate_size=config.intermediate_size,
                                time_step_rank=config.time_step_rank,
                                use_conv_bias=config.use_conv_bias,
                                use_bias=config.use_bias,
                                use_rms_norm=self.is_falcon_mamba,
                                rms_norm_has_weight=not self.is_falcon_mamba,
                                rms_norm_eps=mixer_rms_eps,
                                activation=config.hidden_act,
                                is_lora_enabled=self.is_lora_enabled)

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer(hidden_states, mamba_cache_params)
        return hidden_states, residual


class MambaModel(nn.Module):

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__()

        config = aphrodite_config.model_config.hf_config
        cache_config = aphrodite_config.cache_config
        quant_config = aphrodite_config.quant_config
        lora_config = aphrodite_config.lora_config
        is_lora_enabled = bool(lora_config)

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embeddings = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MambaDecoderLayer(config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             is_lora_enabled=is_lora_enabled),
            prefix=f"{prefix}.layers")

        self.norm_f = RMSNorm(config.hidden_size,
                              eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                mamba_cache_params=mamba_cache_params.at_layer_idx(
                    i - self.start_layer))
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm_f(hidden_states, residual)

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "A_log" in name:
                name = name.replace("A_log", "A")
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
        return loaded_params


class MambaForCausalLM(nn.Module, HasInnerState, IsAttentionFree, SupportsPP,
                       SupportsV0Only):

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        config = aphrodite_config.model_config.hf_config
        cache_config = aphrodite_config.cache_config
        lora_config = aphrodite_config.lora_config
        self.scheduler_config = aphrodite_config.scheduler_config
        assert not cache_config.enable_prefix_caching, \
            "Mamba does not support prefix caching"

        super().__init__()
        self.config = config
        self.aphrodite_config = aphrodite_config
        self.model_config = aphrodite_config.model_config
        self.backbone = MambaModel(aphrodite_config=aphrodite_config,
                                   prefix=maybe_prefix(prefix, "backbone"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        if config.tie_word_embeddings:
            self.lm_head = self.backbone.embeddings
        else:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size,
            )

        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.backbone.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_input_embeddings(input_ids)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs):
        if self.mamba_cache is None:
            num_mamba_layers = self.model_config.get_num_layers_by_block_type(
                self.aphrodite_config.parallel_config, LayerBlockType.mamba)
            self.mamba_cache = MambaCacheManager(
                self.aphrodite_config, self.lm_head.weight.dtype, num_mamba_layers,
                *self._get_mamba_cache_shape())

        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        hidden_states = self.backbone(input_ids, positions, mamba_cache_params,
                                      intermediate_tensors, inputs_embeds)

        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_mamba_cache_shape(
            self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        conv_state_shape = (
            self.config.intermediate_size // world_size,
            self.config.conv_kernel - 1,
        )
        temporal_state_shape = (
            self.config.intermediate_size // world_size,
            self.config.state_size,
        )
        return conv_state_shape, temporal_state_shape

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
