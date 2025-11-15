# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from aphrodite.diffusion.configs.pipelines.base import (
    PipelineConfig,
    SlidingTileAttnConfig,
)
from aphrodite.diffusion.configs.pipelines.flux import FluxPipelineConfig
from aphrodite.diffusion.configs.pipelines.hunyuan import (
    FastHunyuanConfig,
    HunyuanConfig,
)
from aphrodite.diffusion.configs.pipelines.registry import (
    get_pipeline_config_cls_from_name,
)
from aphrodite.diffusion.configs.pipelines.sdxl import (
    SDXLPipelineConfig,
    SDXLVAEConfig,
)
from aphrodite.diffusion.configs.pipelines.stepvideo import StepVideoT2VConfig
from aphrodite.diffusion.configs.pipelines.wan import (
    SelfForcingWanT2V480PConfig,
    WanI2V480PConfig,
    WanI2V720PConfig,
    WanT2V480PConfig,
    WanT2V720PConfig,
)

__all__ = [
    "HunyuanConfig",
    "FastHunyuanConfig",
    "FluxPipelineConfig",
    "PipelineConfig",
    "SDXLPipelineConfig",
    "SDXLVAEConfig",
    "SlidingTileAttnConfig",
    "WanT2V480PConfig",
    "WanI2V480PConfig",
    "WanT2V720PConfig",
    "WanI2V720PConfig",
    "StepVideoT2VConfig",
    "SelfForcingWanT2V480PConfig",
    "get_pipeline_config_cls_from_name",
]
