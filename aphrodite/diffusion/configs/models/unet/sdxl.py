# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from ComfyUI: comfy/supported_models.py

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from aphrodite.diffusion.configs.models.base import ArchConfig, ModelConfig
from aphrodite.diffusion.runtime.layers.quantization import QuantizationConfig


@dataclass
class SDXLUNetArchConfig(ArchConfig):
    """
    SDXL UNet architecture configuration.
    
    Based on ComfyUI's SDXL unet_config:
    - model_channels: 320
    - transformer_depth: [0, 0, 2, 2, 10, 10]
    - context_dim: 2048 (CLIP-L 768 + CLIP-G 1280)
    - adm_in_channels: 2816 (CLIP pooled 1280 + 6×256 embeddings)
    - use_linear_in_transformer: True
    """
    
    # Core UNet parameters
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 320
    num_res_blocks: int = 2
    channel_mult: tuple[int, ...] = (1, 2, 4, 4)
    dropout: float = 0.0
    conv_resample: bool = True
    dims: int = 2
    
    # Attention parameters
    num_heads: int = -1  # Use num_head_channels instead
    num_head_channels: int = 64
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    
    # Transformer parameters
    use_spatial_transformer: bool = True
    transformer_depth: list[int] = field(default_factory=lambda: [0, 0, 2, 2, 10, 10])
    transformer_depth_middle: int | None = None
    transformer_depth_output: list[int] | None = None
    context_dim: int = 2048  # CLIP-L (768) + CLIP-G (1280)
    use_linear_in_transformer: bool = True
    
    # ADM conditioning
    adm_in_channels: int = 2816  # CLIP pooled (1280) + 6×256 embeddings
    
    # Other parameters
    use_checkpoint: bool = False
    num_classes: int | None = None


@dataclass
class SDXLUNetConfig(ModelConfig):
    """
    SDXL UNet model configuration.
    
    Wraps SDXLUNetArchConfig with Aphrodite-specific parameters.
    """
    
    arch_config: ArchConfig = field(default_factory=SDXLUNetArchConfig)
    
    # Aphrodite-specific parameters
    prefix: str = "unet"
    quant_config: QuantizationConfig | None = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Ensure arch_config is SDXLUNetArchConfig
        if not isinstance(self.arch_config, SDXLUNetArchConfig):
            # If a generic ArchConfig was provided, convert it
            if isinstance(self.arch_config, ArchConfig):
                # Create new SDXLUNetArchConfig with values from provided config
                arch_dict = {
                    field.name: getattr(self.arch_config, field.name, None)
                    for field in SDXLUNetArchConfig.__dataclass_fields__.values()
                    if hasattr(self.arch_config, field.name)
                }
                # Also check extra_attrs
                if hasattr(self.arch_config, "extra_attrs"):
                    for key, value in self.arch_config.extra_attrs.items():
                        if key in SDXLUNetArchConfig.__dataclass_fields__:
                            arch_dict[key] = value
                
                self.arch_config = SDXLUNetArchConfig(**arch_dict)
            else:
                self.arch_config = SDXLUNetArchConfig()

