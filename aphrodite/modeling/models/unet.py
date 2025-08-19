"""UNet2DConditionModel implementation for Stable Diffusion."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aphrodite.config import AphroditeConfig
from aphrodite.modeling.layers.linear import (ColumnParallelLinear,
                                              RowParallelLinear)
# Attention will be implemented using basic PyTorch operations
from aphrodite.modeling.models.interfaces import SupportsQuant
from aphrodite.modeling.models.utils import (AutoWeightsLoader,
                                             default_weight_loader,
                                             maybe_prefix)
from aphrodite.common.sequence import IntermediateTensors


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    Create sinusoidal timestep embeddings.
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

        if act_fn == "silu":
            self.act = F.silu
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class Downsample2D(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True, padding: int = 1):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.padding = padding

        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3,
                                  stride=2, padding=padding)
        else:
            self.conv = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        else:
            hidden_states = F.avg_pool2d(hidden_states, kernel_size=2, stride=2)
        return hidden_states


class Upsample2D(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Upsample by factor of 2
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


class UNetResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int = 1280,
        groups: int = 32,
        eps: float = 1e-6,
        use_shortcut: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_shortcut = use_shortcut

        # First conv block
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection
        self.time_emb_proj = nn.Linear(time_embedding_dim, out_channels)

        # Second conv block
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Shortcut connection
        if self.use_shortcut and in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None

    def forward(self, hidden_states: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        # First conv block
        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        # Time embedding
        time_emb = F.silu(time_emb)
        time_emb = self.time_emb_proj(time_emb)
        # Reshape time embedding to match spatial dimensions
        time_emb = time_emb[:, :, None, None]
        hidden_states = hidden_states + time_emb

        # Second conv block
        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        # Shortcut connection
        if self.use_shortcut:
            if self.conv_shortcut is not None:
                residual = self.conv_shortcut(residual)
            hidden_states = hidden_states + residual

        return hidden_states


class UNetSelfAttention(nn.Module):
    """Self-attention block for UNet using Aphrodite's efficient attention."""

    def __init__(
        self,
        channels: int,
        num_attention_heads: int = 8,
        attention_head_dim: int = None,
        groups: int = 32,
        eps: float = 1e-6,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.channels = channels

        if attention_head_dim is None:
            attention_head_dim = channels // num_attention_heads

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(groups, channels, eps=eps)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1)

        # Use simple attention implementation for diffusion models
        # (no KV caching needed)

        # QKV projection
        self.to_qkv = ColumnParallelLinear(
            channels, 3 * self.inner_dim, bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "to_qkv"),
        )

        # Output projection
        self.to_out = nn.ModuleList([
            RowParallelLinear(
                self.inner_dim, channels, bias=True,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "to_out.0"),
            ),
            nn.Identity(),
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        batch_size, channels, height, width = hidden_states.shape

        # Normalization and conv input
        hidden_states = self.norm(hidden_states)
        hidden_states = self.conv_input(hidden_states)

        # Reshape to sequence format: (batch, seq_len, channels)
        hidden_states = hidden_states.view(batch_size, channels, height * width)
        hidden_states = hidden_states.transpose(1, 2)  # (batch, seq_len, channels)

        # Get QKV
        qkv = self.to_qkv(hidden_states)
        if isinstance(qkv, tuple):
            qkv = qkv[0]  # Handle parallel linear output

        seq_len = hidden_states.shape[1]

        # Split QKV
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_dim)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_dim)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_dim)

        # Apply attention using simple scaled dot-product (no KV cache needed)
        scale = 1.0 / math.sqrt(self.attention_head_dim)

        # Transpose for attention computation: (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back to sequence format
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.inner_dim)

        # Output projection
        attn_output = self.to_out[0](attn_output)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        attn_output = self.to_out[1](attn_output)

        # Reshape back to spatial format
        attn_output = attn_output.transpose(1, 2)  # (batch, channels, seq_len)
        attn_output = attn_output.view(batch_size, channels, height, width)

        return attn_output + residual


class UNetCrossAttention(nn.Module):
    """Cross-attention block for UNet with text conditioning."""

    def __init__(
        self,
        channels: int,
        cross_attention_dim: int = 768,
        num_attention_heads: int = 8,
        attention_head_dim: int = None,
        groups: int = 32,
        eps: float = 1e-6,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.channels = channels
        self.cross_attention_dim = cross_attention_dim

        if attention_head_dim is None:
            attention_head_dim = channels // num_attention_heads

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.LayerNorm(channels)

        # Query projection (from image features)
        self.to_q = ColumnParallelLinear(
            channels, self.inner_dim, bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "to_q"),
        )

        # Key and Value projections (from text features)
        self.to_k = ColumnParallelLinear(
            cross_attention_dim, self.inner_dim, bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "to_k"),
        )

        self.to_v = ColumnParallelLinear(
            cross_attention_dim, self.inner_dim, bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "to_v"),
        )

        # Output projection
        self.to_out = nn.ModuleList([
            RowParallelLinear(
                self.inner_dim, channels, bias=True,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "to_out.0"),
            ),
            nn.Identity(),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        batch_size, channels, height, width = hidden_states.shape

        # Reshape to sequence format
        hidden_states = hidden_states.view(batch_size, channels, height * width)
        hidden_states = hidden_states.transpose(1, 2)  # (batch, seq_len, channels)

        # Normalize
        hidden_states = self.norm(hidden_states)

        # Get query from image features
        q = self.to_q(hidden_states)
        if isinstance(q, tuple):
            q = q[0]

        # Get key and value from text features
        k = self.to_k(encoder_hidden_states)
        if isinstance(k, tuple):
            k = k[0]

        v = self.to_v(encoder_hidden_states)
        if isinstance(v, tuple):
            v = v[0]

        seq_len = hidden_states.shape[1]
        text_seq_len = encoder_hidden_states.shape[1]

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_dim)
        k = k.view(batch_size, text_seq_len, self.num_attention_heads, self.attention_head_dim)
        v = v.view(batch_size, text_seq_len, self.num_attention_heads, self.attention_head_dim)

        # Transpose for attention: (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute cross-attention
        scale = 1.0 / math.sqrt(self.attention_head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.inner_dim)

        # Output projection
        attn_output = self.to_out[0](attn_output)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        attn_output = self.to_out[1](attn_output)

        # Reshape back to spatial format
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.view(batch_size, channels, height, width)

        return attn_output + residual


class UNetAttentionBlock(nn.Module):
    """Combined self-attention and cross-attention block with feedforward."""

    def __init__(
        self,
        channels: int,
        cross_attention_dim: int = 768,
        num_attention_heads: int = 8,
        attention_head_dim: int = None,
        groups: int = 32,
        eps: float = 1e-6,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()

        self.norm_input = nn.GroupNorm(groups, channels, eps=eps)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1)

        # Self-attention
        self.norm1 = nn.LayerNorm(channels)
        self.attn1 = UNetSelfAttention(
            channels=channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            groups=groups,
            eps=eps,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "attn1"),
        )

        # Cross-attention
        self.norm2 = nn.LayerNorm(channels)
        self.attn2 = UNetCrossAttention(
            channels=channels,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            groups=groups,
            eps=eps,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "attn2"),
        )

        # Feedforward network (GeGLU)
        self.norm3 = nn.LayerNorm(channels)
        self.ff = nn.ModuleList([
            nn.Linear(channels, channels * 8),  # 4x expansion * 2 for GeGLU
            nn.Linear(channels * 4, channels),
        ])

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual_long = hidden_states
        batch_size, channels, height, width = hidden_states.shape

        # Input processing
        hidden_states = self.norm_input(hidden_states)
        hidden_states = self.conv_input(hidden_states)

        # Reshape for attention operations
        hidden_states_2d = hidden_states.view(batch_size, channels, height * width)
        hidden_states_2d = hidden_states_2d.transpose(1, 2)  # (batch, seq_len, channels)

        # Self-attention
        residual_short = hidden_states_2d
        hidden_states_2d = self.norm1(hidden_states_2d)

        # Apply self-attention on spatial format
        hidden_states_spatial = hidden_states_2d.transpose(1, 2).view(batch_size, channels, height, width)
        hidden_states_spatial = self.attn1(hidden_states_spatial)
        hidden_states_2d = hidden_states_spatial.view(batch_size, channels, height * width).transpose(1, 2)

        hidden_states_2d = hidden_states_2d + residual_short

        # Cross-attention
        residual_short = hidden_states_2d
        hidden_states_2d = self.norm2(hidden_states_2d)

        # Apply cross-attention on spatial format
        hidden_states_spatial = hidden_states_2d.transpose(1, 2).view(batch_size, channels, height, width)
        hidden_states_spatial = self.attn2(hidden_states_spatial, encoder_hidden_states)
        hidden_states_2d = hidden_states_spatial.view(batch_size, channels, height * width).transpose(1, 2)

        hidden_states_2d = hidden_states_2d + residual_short

        # Feedforward (GeGLU)
        residual_short = hidden_states_2d
        hidden_states_2d = self.norm3(hidden_states_2d)

        # GeGLU: split into gate and value, apply gelu to gate
        hidden_states_2d, gate = self.ff[0](hidden_states_2d).chunk(2, dim=-1)
        hidden_states_2d = hidden_states_2d * F.gelu(gate)
        hidden_states_2d = self.ff[1](hidden_states_2d)

        hidden_states_2d = hidden_states_2d + residual_short

        # Reshape back to spatial format
        hidden_states = hidden_states_2d.transpose(1, 2).view(batch_size, channels, height, width)
        hidden_states = self.conv_output(hidden_states)

        return hidden_states + residual_long


class UNet2DConditionModel(nn.Module, SupportsQuant):
    """
    UNet model for conditional image generation (Stable Diffusion).

    This is a 2D UNet model that takes a noisy sample, conditional state, and
    timestep to produce a denoised output.
    """

    is_unet_model = True

    def __init__(
        self,
        *,
        aphrodite_config: AphroditeConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = aphrodite_config.model_config.hf_config
        quant_config = aphrodite_config.quant_config

        # Configuration
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.block_out_channels = config.block_out_channels
        self.layers_per_block = config.layers_per_block
        self.attention_head_dim = config.attention_head_dim
        self.cross_attention_dim = config.cross_attention_dim
        self.norm_num_groups = config.norm_num_groups
        self.time_embedding_dim = config.time_embedding_dim

        # Time embedding
        time_embed_dim = self.time_embedding_dim
        self.time_proj = lambda x: get_timestep_embedding(x, 320, flip_sin_to_cos=True)
        self.time_embedding = TimestepEmbedding(320, time_embed_dim)

        # Input convolution
        self.conv_in = nn.Conv2d(
            self.in_channels, self.block_out_channels[0], kernel_size=3, padding=1
        )

        # Build encoder blocks (down-sampling path)
        self.down_blocks = nn.ModuleList([])
        output_channel = self.block_out_channels[0]

        for i, down_block_out_channels in enumerate(self.block_out_channels):
            input_channel = output_channel
            output_channel = down_block_out_channels
            is_final_block = i == len(self.block_out_channels) - 1
            has_attention = i < 3  # First 3 blocks have attention

            down_block = []

            # Add residual blocks
            for j in range(self.layers_per_block):
                down_block.append(
                    UNetResidualBlock(
                        in_channels=input_channel if j == 0 else output_channel,
                        out_channels=output_channel,
                        time_embedding_dim=time_embed_dim,
                        groups=self.norm_num_groups,
                    )
                )

                # Add attention block if needed
                if has_attention:
                    down_block.append(
                        UNetAttentionBlock(
                            channels=output_channel,
                            cross_attention_dim=self.cross_attention_dim,
                            num_attention_heads=output_channel // self.attention_head_dim,
                            attention_head_dim=self.attention_head_dim,
                            groups=self.norm_num_groups,
                            quant_config=quant_config,
                            prefix=maybe_prefix(prefix, f"down_blocks.{i}.attentions.{j}"),
                        )
                    )

                input_channel = output_channel

            # Add downsampling (except for final block)
            if not is_final_block:
                down_block.append(Downsample2D(output_channel))

            self.down_blocks.append(nn.ModuleList(down_block))

        # Middle block
        mid_block_channel = self.block_out_channels[-1]
        self.mid_block = nn.ModuleList([
            UNetResidualBlock(
                in_channels=mid_block_channel,
                out_channels=mid_block_channel,
                time_embedding_dim=time_embed_dim,
                groups=self.norm_num_groups,
            ),
            UNetAttentionBlock(
                channels=mid_block_channel,
                cross_attention_dim=self.cross_attention_dim,
                num_attention_heads=mid_block_channel // self.attention_head_dim,
                attention_head_dim=self.attention_head_dim,
                groups=self.norm_num_groups,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "mid_block.attentions.0"),
            ),
            UNetResidualBlock(
                in_channels=mid_block_channel,
                out_channels=mid_block_channel,
                time_embedding_dim=time_embed_dim,
                groups=self.norm_num_groups,
            ),
        ])

        # Build decoder blocks (up-sampling path)
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(self.block_out_channels))

        for i, up_block_out_channels in enumerate(reversed_block_out_channels):
            prev_output_channel = reversed_block_out_channels[i - 1] if i > 0 else reversed_block_out_channels[0]
            output_channel = up_block_out_channels
            input_channel = prev_output_channel + output_channel  # Skip connection
            is_final_block = i == len(self.block_out_channels) - 1
            has_attention = i < 3  # First 3 blocks have attention

            up_block = []

            # Add residual blocks
            for j in range(self.layers_per_block + 1):  # +1 for decoder
                up_block.append(
                    UNetResidualBlock(
                        in_channels=input_channel if j == 0 else output_channel,
                        out_channels=output_channel,
                        time_embedding_dim=time_embed_dim,
                        groups=self.norm_num_groups,
                    )
                )

                # Add attention block if needed
                if has_attention:
                    up_block.append(
                        UNetAttentionBlock(
                            channels=output_channel,
                            cross_attention_dim=self.cross_attention_dim,
                            num_attention_heads=output_channel // self.attention_head_dim,
                            attention_head_dim=self.attention_head_dim,
                            groups=self.norm_num_groups,
                            quant_config=quant_config,
                            prefix=maybe_prefix(prefix, f"up_blocks.{i}.attentions.{j}"),
                        )
                    )

                input_channel = output_channel

            # Add upsampling (except for final block)
            if not is_final_block:
                up_block.append(Upsample2D(output_channel))

            self.up_blocks.append(nn.ModuleList(up_block))

        # Output layers
        self.conv_norm_out = nn.GroupNorm(self.norm_num_groups, self.block_out_channels[0])
        self.conv_out = nn.Conv2d(
            self.block_out_channels[0], self.out_channels, kernel_size=3, padding=1
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            sample: (batch_size, in_channels, height, width) noisy samples
            timestep: (batch_size,) timesteps
            encoder_hidden_states: (batch_size, seq_len, cross_attention_dim) text embeddings

        Returns:
            (batch_size, out_channels, height, width) predicted noise
        """
        # Time embedding
        timesteps = timestep.flatten()
        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        # Input convolution
        sample = self.conv_in(sample)

        # Encoder (down-sampling)
        down_block_res_samples = [sample]
        for down_block in self.down_blocks:
            for layer in down_block:
                if isinstance(layer, UNetResidualBlock):
                    sample = layer(sample, emb)
                elif isinstance(layer, UNetAttentionBlock):
                    sample = layer(sample, encoder_hidden_states)
                else:  # Downsample2D
                    sample = layer(sample)
            down_block_res_samples.append(sample)

        # Middle block
        for layer in self.mid_block:
            if isinstance(layer, UNetResidualBlock):
                sample = layer(sample, emb)
            elif isinstance(layer, UNetAttentionBlock):
                sample = layer(sample, encoder_hidden_states)

        # Decoder (up-sampling)
        for up_block in self.up_blocks:
            res_sample = down_block_res_samples.pop()
            sample = torch.cat([sample, res_sample], dim=1)

            for layer in up_block:
                if isinstance(layer, UNetResidualBlock):
                    sample = layer(sample, emb)
                elif isinstance(layer, UNetAttentionBlock):
                    sample = layer(sample, encoder_hidden_states)
                else:  # Upsample2D
                    sample = layer(sample)

        # Output
        sample = self.conv_norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_out(sample)

        return sample

    def load_weights(self, weights: List[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from a diffusers checkpoint."""
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            else:
                print(f"DEBUG: Weight {name} not found in model parameters")

        return loaded_params
