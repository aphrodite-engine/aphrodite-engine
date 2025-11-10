# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from aphrodite.diffusion.configs.models.base import ModelConfig
from aphrodite.diffusion.configs.models.dits.base import DiTConfig
from aphrodite.diffusion.configs.models.encoders.base import EncoderConfig
from aphrodite.diffusion.configs.models.vaes.base import VAEConfig

__all__ = ["ModelConfig", "VAEConfig", "DiTConfig", "EncoderConfig"]
