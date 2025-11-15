# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from ComfyUI: comfy/sdxl_clip.py

# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

import torch

from aphrodite.diffusion.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPTextConfig,
    TextEncoderConfig,
)
from aphrodite.diffusion.runtime.models.encoders.base import TextEncoder
from aphrodite.diffusion.runtime.models.encoders.clip import CLIPTextModel


class SDXLClipG(CLIPTextModel):
    """
    SDXL CLIP-G (Gigantic) text encoder.
    Based on ComfyUI's SDXLClipG implementation.
    
    CLIP-G uses the penultimate layer (layer_idx=-2) instead of the final layer.
    """
    
    def __init__(
        self,
        config: CLIPTextConfig,
    ) -> None:
        super().__init__(config)
        self.use_penultimate_layer = True  # Use penultimate layer like ComfyUI


class SDXLClipModel(TextEncoder):
    """
    SDXL dual CLIP text encoder model.
    Combines CLIP-L (large) and CLIP-G (gigantic) encoders.
    
    Based on ComfyUI's SDXLClipModel implementation.
    """
    
    def __init__(
        self,
        config: TextEncoderConfig | None = None,
        clip_l_config: CLIPTextConfig | None = None,
        clip_g_config: CLIPTextConfig | None = None,
    ) -> None:
        # Support both single config (from loader) and separate configs (from tests)
        if config is not None:
            # Extract clip_l_config and clip_g_config from SDXLClipConfig
            from aphrodite.diffusion.configs.models.encoders.sdxl_clip import (
                SDXLClipConfig,
                SDXLClipLConfig,
                SDXLClipGConfig,
            )
            if isinstance(config, SDXLClipConfig):
                # Create CLIP-L and CLIP-G configs from SDXLClipConfig
                clip_l_config = SDXLClipLConfig(
                    arch_config=config.clip_l_config,
                    prefix=config.prefix + ".clip_l" if hasattr(config, "prefix") else "clip_l",
                )
                clip_g_config = SDXLClipGConfig(
                    arch_config=config.clip_g_config,
                    prefix=config.prefix + ".clip_g" if hasattr(config, "prefix") else "clip_g",
                )
            else:
                # Fallback: create default configs
                clip_l_config = SDXLClipLConfig()
                clip_g_config = SDXLClipGConfig()
        
        if clip_l_config is None or clip_g_config is None:
            raise ValueError("Either config (SDXLClipConfig) or both clip_l_config and clip_g_config must be provided")
        
        super().__init__(clip_l_config)  # Use clip_l_config as base
        
        self.clip_l = CLIPTextModel(clip_l_config)
        self.clip_g = SDXLClipG(clip_g_config)
        
        # Store both configs
        self.clip_l_config = clip_l_config
        self.clip_g_config = clip_g_config
    
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        """
        Forward pass for dual CLIP encoding.
        
        Args:
            input_ids: Can be a dict with "l" and "g" keys, or a single tensor for both
            Other args: Standard CLIP forward arguments
        
        Returns:
            BaseEncoderOutput with concatenated embeddings and pooled output from clip_g
        """
        # Handle dual input_ids (dict with "l" and "g" keys)
        if isinstance(input_ids, dict):
            input_ids_l = input_ids.get("l")
            input_ids_g = input_ids.get("g")
        else:
            # If single tensor, use for both (they should be the same)
            input_ids_l = input_ids
            input_ids_g = input_ids
        
        # Encode with CLIP-L (also uses penultimate layer like ComfyUI)
        l_outputs: BaseEncoderOutput = self.clip_l(
            input_ids=input_ids_l,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,  # Always get hidden states for penultimate layer
            **kwargs,
        )
        
        # CLIP-L also uses penultimate layer (ComfyUI: layer="hidden", layer_idx=-2)
        if hasattr(l_outputs, 'hidden_states') and l_outputs.hidden_states is not None:
            if len(l_outputs.hidden_states) >= 3:
                # Get the second-to-last transformer layer (penultimate)
                l_out = l_outputs.hidden_states[-2]
            else:
                l_out = l_outputs.last_hidden_state
        else:
            l_out = l_outputs.last_hidden_state
        
        # Encode with CLIP-G (need hidden states to get penultimate layer)
        g_outputs: BaseEncoderOutput = self.clip_g(
            input_ids=input_ids_g,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,  # Always get hidden states for penultimate layer
            **kwargs,
        )
        
        # CLIP-G uses penultimate layer (before final layer norm)
        # ComfyUI: layer="hidden", layer_idx=-2
        # The encoder returns hidden_states as a list:
        # - hidden_states[0] = inputs_embeds
        # - hidden_states[1] = after layer 0
        # - ...
        # - hidden_states[-1] = after final layer (before final layer norm)
        # Then final_layer_norm is applied to get last_hidden_state
        # For penultimate layer (layer_idx=-2), we want the second-to-last transformer layer
        if hasattr(g_outputs, 'hidden_states') and g_outputs.hidden_states is not None:
            # hidden_states is a list from CLIPEncoder
            # The last element is after the final transformer layer (before final layer norm)
            # For penultimate, we want the second-to-last transformer layer output
            if len(g_outputs.hidden_states) >= 3:
                # Get the second-to-last transformer layer (penultimate)
                # hidden_states[-1] is final layer, hidden_states[-2] is penultimate
                g_out = g_outputs.hidden_states[-2]
            else:
                # Fallback to last hidden state if not enough layers
                g_out = g_outputs.last_hidden_state
        else:
            # Fallback if hidden_states not available
            g_out = g_outputs.last_hidden_state
        
        # Concatenate along feature dimension (dim=-1)
        # ComfyUI: cut_to = min(l_out.shape[1], g_out.shape[1])
        # then torch.cat([l_out[:,:cut_to], g_out[:,:cut_to]], dim=-1)
        cut_to = min(l_out.shape[1], g_out.shape[1])
        
        # Concatenate: [B, seq_len, l_dim + g_dim]
        concatenated = torch.cat([l_out[:, :cut_to], g_out[:, :cut_to]], dim=-1)
        
        # Return pooled output from CLIP-G (for ADM conditioning)
        pooled_output = g_outputs.pooler_output if hasattr(g_outputs, 'pooler_output') else None
        
        return BaseEncoderOutput(
            last_hidden_state=concatenated,
            pooler_output=pooled_output,
            hidden_states=(
                [l_outputs.hidden_states, g_outputs.hidden_states]
                if output_hidden_states and hasattr(l_outputs, 'hidden_states')
                else None
            ),
        )
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights for both CLIP-L and CLIP-G.
        
        Handles both diffusers format (conditioner.embedders.*) and
        Aphrodite format (clip_l.*, clip_g.*).
        """
        from aphrodite.diffusion.runtime.models.encoders.sdxl_clip_weight_utils import (
            process_sdxl_clip_weights,
        )
        
        # Convert weights from diffusers format if needed
        processed_weights = list(process_sdxl_clip_weights(weights))
        
        loaded_params: set[str] = set()
        clip_l_weights: list[tuple[str, torch.Tensor]] = []
        clip_g_weights: list[tuple[str, torch.Tensor]] = []
        
        for name, weight in processed_weights:
            if name.startswith("clip_l."):
                # Remove "clip_l." prefix
                clip_l_name = name[len("clip_l."):]
                clip_l_weights.append((clip_l_name, weight))
            elif name.startswith("clip_g."):
                # Remove "clip_g." prefix
                clip_g_name = name[len("clip_g."):]
                clip_g_weights.append((clip_g_name, weight))
            elif name.startswith("text_model."):
                # Fallback: assume clip_l if no prefix
                clip_l_weights.append((name, weight))
        
        # Load CLIP-L weights
        if clip_l_weights:
            l_loaded = self.clip_l.load_weights(clip_l_weights)
            loaded_params.update(f"clip_l.{name}" for name in l_loaded)
        
        # Load CLIP-G weights
        if clip_g_weights:
            g_loaded = self.clip_g.load_weights(clip_g_weights)
            loaded_params.update(f"clip_g.{name}" for name in g_loaded)
        
        return loaded_params


# Entry point for model registry
EntryClass = SDXLClipModel

