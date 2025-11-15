# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from ComfyUI: comfy/supported_models.py

# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from aphrodite.diffusion.configs.models import EncoderConfig, VAEConfig
from aphrodite.diffusion.configs.models.encoders import (
    BaseEncoderOutput,
    SDXLClipConfig,
)
from aphrodite.diffusion.configs.models.unet import SDXLUNetConfig
from aphrodite.diffusion.configs.models.vaes.base import (
    VAEArchConfig,
)
from aphrodite.diffusion.configs.models.vaes.base import (
    VAEConfig as BaseVAEConfig,
)
from aphrodite.diffusion.configs.pipelines.base import PipelineConfig


def sdxl_preprocess_text(text: str) -> str:
    """Preprocess text for SDXL dual CLIP encoding."""
    # SDXL uses standard CLIP tokenization
    return text


def sdxl_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    """
    Postprocess SDXL dual CLIP output.

    Returns the concatenated hidden states from CLIP-L and CLIP-G.
    """
    return outputs.last_hidden_state


@dataclass
class SDXLVAEConfig(BaseVAEConfig):
    """SDXL VAE configuration using standard AutoencoderKL."""

    arch_config: VAEArchConfig = field(default_factory=VAEArchConfig)

    def __post_init__(self):
        """Set SDXL-specific VAE defaults."""
        # SDXL uses standard 4-channel VAE with 8x scale factor
        if (
            not hasattr(self.arch_config, "spatial_compression_ratio")
            or self.arch_config.spatial_compression_ratio is None
        ):
            self.arch_config.spatial_compression_ratio = 8
        # SDXL VAE scaling factor (from ComfyUI: 0.13025)
        if not hasattr(self.arch_config, "scaling_factor") or self.arch_config.scaling_factor == 0:
            self.arch_config.scaling_factor = 0.13025


@dataclass
class SDXLPipelineConfig(PipelineConfig):
    """
    SDXL pipeline configuration.

    Based on ComfyUI's SDXL configuration:
    - UNet with ADM conditioning (2816 dims for base, 2560 for refiner)
    - Dual CLIP text encoder (CLIP-L + CLIP-G)
    - Standard AutoencoderKL VAE (4-channel latents)
    """

    # SDXL-specific parameters
    is_image_gen: bool = True
    embedded_cfg_scale: float = 6.0  # SDXL default guidance scale

    # UNet configuration (replaces dit_config for UNet-based models)
    unet_config: SDXLUNetConfig = field(default_factory=SDXLUNetConfig)
    unet_precision: str = "bf16"

    # VAE configuration
    vae_config: VAEConfig = field(default_factory=SDXLVAEConfig)
    vae_precision: str = "fp32"  # Use fp32 for VAE to prevent NaN (ComfyUI default)
    vae_tiling: bool = True
    vae_sp: bool = True

    # Text encoder configuration - single dual CLIP
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (SDXLClipConfig(),))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp16",))
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            dict(
                max_length=77,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
            ),
        ]
    )

    # Preprocess/postprocess functions for dual CLIP
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(default_factory=lambda: (sdxl_preprocess_text,))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = field(
        default_factory=lambda: (sdxl_postprocess_text,)
    )

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        """
        Prepare latent shape for SDXL.

        SDXL uses standard 4-channel latents with 8x VAE scale factor.
        For image generation, num_frames should be 1.
        """
        vae_scale_factor = self.vae_config.arch_config.spatial_compression_ratio
        height = batch.height // vae_scale_factor
        width = batch.width // vae_scale_factor

        # SDXL uses 4-channel latents: [B, 4, H/8, W/8]
        shape = (batch_size, 4, height, width)
        return shape

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        """Prepare positive conditioning kwargs including ADM conditioning for SDXL."""
        from aphrodite.diffusion.runtime.models.unet.adm_conditioning import encode_sdxl_adm

        # Get pooled output from text encoder (CLIP-G pooled output)
        # For SDXL, pooled output should be in batch.pooled_embeds
        pooled_output = None
        if hasattr(batch, "pooled_embeds") and batch.pooled_embeds is not None:
            if isinstance(batch.pooled_embeds, list) and len(batch.pooled_embeds) > 0:
                pooled_output = batch.pooled_embeds[0]
            elif not isinstance(batch.pooled_embeds, list):
                pooled_output = batch.pooled_embeds

        # Encode ADM conditioning
        # If pooled_output is None, create a zero tensor with correct shape
        height_val = batch.height if isinstance(batch.height, int) else batch.height[0]
        width_val = batch.width if isinstance(batch.width, int) else batch.width[0]

        if pooled_output is None:
            # Create zero tensor for CLIP pooled output [B, 1280]
            batch_size = 1  # Will be expanded in the denoising loop
            pooled_output = torch.zeros((batch_size, 1280), device=device, dtype=dtype)

        adm_cond = encode_sdxl_adm(
            clip_pooled=pooled_output,
            height=height_val,
            width=width_val,
            crop_w=0,
            crop_h=0,
            target_width=width_val,
            target_height=height_val,
        )

        return {"y": adm_cond}

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        """Prepare negative conditioning kwargs including ADM conditioning for SDXL."""
        from aphrodite.diffusion.runtime.models.unet.adm_conditioning import encode_sdxl_adm

        # For negative prompts, use same resolution but potentially different aesthetic score
        pooled_output = None
        if hasattr(batch, "neg_pooled_embeds") and batch.neg_pooled_embeds is not None:
            if isinstance(batch.neg_pooled_embeds, list) and len(batch.neg_pooled_embeds) > 0:
                pooled_output = batch.neg_pooled_embeds[0]
            elif not isinstance(batch.neg_pooled_embeds, list):
                pooled_output = batch.neg_pooled_embeds

        # Encode ADM conditioning for negative prompt
        height_val = batch.height if isinstance(batch.height, int) else batch.height[0]
        width_val = batch.width if isinstance(batch.width, int) else batch.width[0]

        if pooled_output is None:
            # Create zero tensor for CLIP pooled output [B, 1280]
            batch_size = 1  # Will be expanded in the denoising loop
            pooled_output = torch.zeros((batch_size, 1280), device=device, dtype=dtype)

        adm_cond = encode_sdxl_adm(
            clip_pooled=pooled_output,
            height=height_val,
            width=width_val,
            crop_w=0,
            crop_h=0,
            target_width=width_val,
            target_height=height_val,
        )

        return {"y": adm_cond}
