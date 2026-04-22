# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
"""Inference-only HF format GLM-4 model compatible with THUDM weights."""

from aphrodite.config import AphroditeConfig
from aphrodite.model_executor.models.llama import LlamaForCausalLM

from .utils import PPMissingLayer


class GlmForCausalLM(LlamaForCausalLM):
    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        hf_config = aphrodite_config.model_config.hf_config
        hf_config.rope_parameters["partial_rotary_factor"] = 0.5
        super().__init__(aphrodite_config=aphrodite_config, prefix=prefix)
        # Hack Llama model to fit HF format GLM implementation
        # Attention difference between GLM and Llama:
        # 1. Half partial rotary_dim and no Neox style.
        # 2. There is no bias for o_proj in attention
        for layer in self.model.layers:
            if not isinstance(layer, PPMissingLayer):
                layer.self_attn.rotary_emb.is_neox_style = False
                layer.self_attn.o_proj.bias = None
                layer.self_attn.o_proj.skip_bias_add = True
