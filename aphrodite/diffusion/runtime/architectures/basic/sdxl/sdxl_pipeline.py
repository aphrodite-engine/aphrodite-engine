# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from ComfyUI: comfy/supported_models.py

# SPDX-License-Identifier: Apache-2.0
"""
SDXL image diffusion pipeline implementation.

This module contains an implementation of the SDXL (Stable Diffusion XL) pipeline
using the modular pipeline architecture.
"""

from aphrodite.diffusion.runtime.pipelines import ComposedPipelineBase
from aphrodite.diffusion.runtime.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from aphrodite.diffusion.runtime.server_args import ServerArgs
from aphrodite.logger import init_logger

logger = init_logger(__name__)


class SDXLPipeline(ComposedPipelineBase):
    """
    SDXL (Stable Diffusion XL) image generation pipeline.

    Based on ComfyUI's SDXL implementation:
    - Dual CLIP text encoder (CLIP-L + CLIP-G)
    - UNet with ADM conditioning
    - Standard AutoencoderKL VAE
    - 4-channel latent format
    """

    pipeline_name = "StableDiffusionXLPipeline"  # Matches diffusers _class_name

    _required_config_modules = [
        "text_encoder",  # SDXLClipModel (dual CLIP)
        "tokenizer",  # SDXL tokenizer
        "unet",  # SDXLUNetModel
        "vae",  # Standard AutoencoderKL
        "scheduler",  # Diffusion scheduler (e.g., EulerDiscreteScheduler)
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        # Input validation
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        # Text encoding with dual CLIP
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        # Conditioning stage (handles ADM conditioning)
        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage(),
        )

        # Timestep preparation
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
            ),
        )

        # Latent preparation (for UNet, we don't need transformer)
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=None,  # UNet doesn't use transformer
            ),
        )

        # Denoising stage with UNet
        # Note: DenoisingStage expects 'transformer' parameter, but we'll pass UNet
        # The stage should handle both transformer and UNet models
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("unet"),  # Pass UNet as transformer
                scheduler=self.get_module("scheduler"),
            ),
        )

        # VAE decoding
        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae")),
        )


EntryClass = SDXLPipeline
