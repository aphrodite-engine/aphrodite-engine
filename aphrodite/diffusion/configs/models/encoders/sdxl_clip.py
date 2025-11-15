# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from ComfyUI: comfy/sdxl_clip.py

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from aphrodite.diffusion.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)
from aphrodite.diffusion.configs.models.encoders.clip import (
    CLIPTextArchConfig,
    CLIPTextConfig,
)


def _is_transformer_layer(n: str, m) -> bool:
    return "layers" in n and str.isdigit(n.split(".")[-1])


def _is_embeddings(n: str, m) -> bool:
    return n.endswith("embeddings")


@dataclass
class SDXLClipLArchConfig(TextEncoderArchConfig):
    """
    SDXL CLIP-L (Large) architecture configuration.
    
    CLIP-L specifications:
    - hidden_size: 768
    - num_hidden_layers: 12
    - num_attention_heads: 12
    - intermediate_size: 3072
    """
    
    vocab_size: int = 49408
    hidden_size: int = 768
    intermediate_size: int = 3072
    projection_dim: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    dropout: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    pad_token_id: int = 1
    bos_token_id: int = 49406
    eos_token_id: int = 49407
    text_len: int = 77
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
    )
    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_transformer_layer, _is_embeddings])


@dataclass
class SDXLClipGArchConfig(TextEncoderArchConfig):
    """
    SDXL CLIP-G (Gigantic) architecture configuration.
    
    CLIP-G specifications:
    - hidden_size: 1280
    - num_hidden_layers: 32
    - num_attention_heads: 20
    - intermediate_size: 5120
    """
    
    vocab_size: int = 49408
    hidden_size: int = 1280
    intermediate_size: int = 5120
    projection_dim: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 20
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    dropout: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    pad_token_id: int = 1
    bos_token_id: int = 49406
    eos_token_id: int = 49407
    text_len: int = 77
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
    )
    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_transformer_layer, _is_embeddings])


@dataclass
class SDXLClipArchConfig(TextEncoderArchConfig):
    """
    SDXL dual CLIP architecture configuration.
    
    Combines CLIP-L and CLIP-G configurations.
    The actual model uses both encoders and concatenates their outputs.
    """
    
    # CLIP-L config
    clip_l_config: SDXLClipLArchConfig = field(default_factory=SDXLClipLArchConfig)
    
    # CLIP-G config
    clip_g_config: SDXLClipGArchConfig = field(default_factory=SDXLClipGArchConfig)
    
    # Combined output dimension (CLIP-L 768 + CLIP-G 1280 = 2048)
    hidden_size: int = 2048


@dataclass
class SDXLClipLConfig(TextEncoderConfig):
    """SDXL CLIP-L model configuration."""
    
    arch_config: TextEncoderArchConfig = field(default_factory=SDXLClipLArchConfig)
    
    num_hidden_layers_override: int | None = None
    require_post_norm: bool | None = None
    prefix: str = "clip_l"


@dataclass
class SDXLClipGConfig(TextEncoderConfig):
    """SDXL CLIP-G model configuration."""
    
    arch_config: TextEncoderArchConfig = field(default_factory=SDXLClipGArchConfig)
    
    num_hidden_layers_override: int | None = None
    require_post_norm: bool | None = None
    prefix: str = "clip_g"


@dataclass
class SDXLClipConfig(TextEncoderConfig):
    """
    SDXL dual CLIP model configuration.
    
    Wraps both CLIP-L and CLIP-G configurations.
    """
    
    arch_config: TextEncoderArchConfig = field(default_factory=SDXLClipArchConfig)
    
    # Individual encoder configs
    clip_l_config: SDXLClipLArchConfig = field(default_factory=SDXLClipLArchConfig)
    clip_g_config: SDXLClipGArchConfig = field(default_factory=SDXLClipGArchConfig)
    
    prefix: str = "sdxl_clip"
    
    # Set architectures to ensure SDXLClipModel is loaded
    architectures: list[str] = field(default_factory=lambda: ["SDXLClipModel"])
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Ensure arch_config is SDXLClipArchConfig
        if not isinstance(self.arch_config, SDXLClipArchConfig):
            if isinstance(self.arch_config, TextEncoderArchConfig):
                # Create new SDXLClipArchConfig
                self.arch_config = SDXLClipArchConfig(
                    clip_l_config=self.clip_l_config,
                    clip_g_config=self.clip_g_config,
                )
            else:
                self.arch_config = SDXLClipArchConfig()

