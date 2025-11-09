# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Pipeline stages for diffusion models.

This package contains the various stages that can be composed to create
complete diffusion pipelines.
"""

from aphrodite.diffusion.runtime.pipelines.stages.base import PipelineStage
from aphrodite.diffusion.runtime.pipelines.stages.causal_denoising import (
    CausalDMDDenoisingStage,
)
from aphrodite.diffusion.runtime.pipelines.stages.conditioning import (
    ConditioningStage,
)
from aphrodite.diffusion.runtime.pipelines.stages.decoding import DecodingStage
from aphrodite.diffusion.runtime.pipelines.stages.denoising import DenoisingStage
from aphrodite.diffusion.runtime.pipelines.stages.denoising_dmd import (
    DmdDenoisingStage,
)
from aphrodite.diffusion.runtime.pipelines.stages.encoding import EncodingStage
from aphrodite.diffusion.runtime.pipelines.stages.image_encoding import (
    ImageEncodingStage,
    ImageVAEEncodingStage,
)
from aphrodite.diffusion.runtime.pipelines.stages.input_validation import (
    InputValidationStage,
)
from aphrodite.diffusion.runtime.pipelines.stages.latent_preparation import (
    LatentPreparationStage,
)
from aphrodite.diffusion.runtime.pipelines.stages.stepvideo_encoding import (
    StepvideoPromptEncodingStage,
)
from aphrodite.diffusion.runtime.pipelines.stages.text_encoding import (
    TextEncodingStage,
)
from aphrodite.diffusion.runtime.pipelines.stages.timestep_preparation import (
    TimestepPreparationStage,
)

__all__ = [
    "PipelineStage",
    "InputValidationStage",
    "TimestepPreparationStage",
    "LatentPreparationStage",
    "ConditioningStage",
    "DenoisingStage",
    "DmdDenoisingStage",
    "CausalDMDDenoisingStage",
    "EncodingStage",
    "DecodingStage",
    "ImageEncodingStage",
    "ImageVAEEncodingStage",
    "TextEncodingStage",
    "StepvideoPromptEncodingStage",
]
