# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from aphrodite.diffusion.configs.models.vaes.hunyuanvae import HunyuanVAEConfig
from aphrodite.diffusion.configs.models.vaes.stepvideovae import StepVideoVAEConfig
from aphrodite.diffusion.configs.models.vaes.wanvae import WanVAEConfig

__all__ = [
    "HunyuanVAEConfig",
    "WanVAEConfig",
    "StepVideoVAEConfig",
]
