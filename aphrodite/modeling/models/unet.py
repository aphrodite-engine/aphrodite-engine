"""UNet2DConditionModel implementation for Stable Diffusion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aphrodite.config import AphroditeConfig
from aphrodite.modeling.models.interfaces import SupportsQuant
from aphrodite.modeling.models.utils import maybe_prefix


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic
    Models.

    Create sinusoidal timestep embeddings.
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
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
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=padding)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
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


class UNetAttention(nn.Module):
    """Attention layer matching Diffusers structure exactly."""

    def __init__(
        self,
        channels: int,
        cross_attention_dim: int | None = None,
        num_attention_heads: int = 8,
        attention_head_dim: int = None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.channels = channels
        self.cross_attention_dim = cross_attention_dim or channels

        if attention_head_dim is None:
            attention_head_dim = channels // num_attention_heads

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.scale = 1.0 / math.sqrt(attention_head_dim)

        # Query projection (always from input features)
        self.to_q = nn.Linear(channels, self.inner_dim, bias=False)

        # Key and Value projections (from cross_attention_dim for cross-attn)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=False)

        # Output projection
        self.to_out = nn.ModuleList(
            [
                nn.Linear(self.inner_dim, channels),
                nn.Identity(),  # Dropout placeholder
            ]
        )

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, channels = hidden_states.shape

        # Use encoder_hidden_states for cross-attention, hidden_states for
        # self-attention
        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        # Get QKV
        q = self.to_q(hidden_states)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head attention
        q = q.view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_dim,
        )
        k = k.view(
            batch_size,
            context.shape[1],
            self.num_attention_heads,
            self.attention_head_dim,
        )
        v = v.view(
            batch_size,
            context.shape[1],
            self.num_attention_heads,
            self.attention_head_dim,
        )

        # Transpose for attention: (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.inner_dim)

        # Output projection
        attn_output = self.to_out[0](attn_output)
        attn_output = self.to_out[1](attn_output)

        return attn_output


class FeedForward(nn.Module):
    """GeGLU feedforward network matching Diffusers structure."""

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # GeGLU: project to 2x inner_dim, then split for gate and value
        self.net = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(dim, inner_dim * 2),  # proj layer (GeGLU splits this)  # noqa: E501
                    ]
                ),
                nn.Identity(),  # Placeholder for potential dropout
                nn.Linear(inner_dim, dim_out),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # GeGLU activation
        hidden_states, gate = self.net[0][0](hidden_states).chunk(2, dim=-1)
        hidden_states = hidden_states * F.gelu(gate)

        # Dropout (currently identity)
        hidden_states = self.net[1](hidden_states)

        # Final projection
        hidden_states = self.net[2](hidden_states)

        return hidden_states


class BasicTransformerBlock(nn.Module):
    """Basic transformer block matching Diffusers structure exactly."""

    def __init__(
        self,
        dim: int,
        cross_attention_dim: int = 768,
        num_attention_heads: int = 8,
        attention_head_dim: int = None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # Self-attention
        self.attn1 = UNetAttention(
            channels=dim,
            cross_attention_dim=None,  # Self-attention
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "attn1"),
        )

        # Cross-attention
        self.attn2 = UNetAttention(
            channels=dim,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "attn2"),
        )

        # Feedforward
        self.ff = FeedForward(
            dim=dim,
            # 4x expansion for GeGLU (4x * 2 = 8x total, matches 2560 output
            # for 320 input)
            mult=4,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "ff"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn1(hidden_states)
        hidden_states = hidden_states + residual

        # Cross-attention
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(hidden_states, encoder_hidden_states)
        hidden_states = hidden_states + residual

        # Feedforward
        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class Transformer2DModel(nn.Module):
    """2D Transformer model matching Diffusers structure."""

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        cross_attention_dim: int = 768,
        num_layers: int = 1,
        norm_num_groups: int = 32,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.in_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # Input/output projections
        self.norm = nn.GroupNorm(norm_num_groups, in_channels, eps=1e-6)
        self.proj_in = nn.Conv2d(in_channels, self.inner_dim, kernel_size=1)
        self.proj_out = nn.Conv2d(self.inner_dim, in_channels, kernel_size=1)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    cross_attention_dim=cross_attention_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, f"transformer_blocks.{i}"),
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = hidden_states.shape
        residual = hidden_states

        # Input projection
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)

        # Reshape to sequence format
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.view(
            batch_size,
            inner_dim,
            height * width,
        )
        hidden_states = hidden_states.transpose(1, 2)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)

        # Reshape back to spatial format
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.view(batch_size, inner_dim, height, width)

        # Output projection
        hidden_states = self.proj_out(hidden_states)

        return hidden_states + residual


class UNetAttentionBlock(nn.Module):
    """Attention block wrapper using Transformer2DModel to match Diffusers
    exactly."""

    def __init__(
        self,
        channels: int,
        cross_attention_dim: int = 768,
        num_attention_heads: int = 8,
        attention_head_dim: int = None,
        groups: int = 32,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()

        if attention_head_dim is None:
            attention_head_dim = channels // num_attention_heads

        # Use Transformer2DModel to match Diffusers structure exactly
        self.transformer = Transformer2DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=channels,
            cross_attention_dim=cross_attention_dim,
            num_layers=1,
            norm_num_groups=groups,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(hidden_states, encoder_hidden_states)


class CrossAttnDownBlock2D(nn.Module):
    """Cross-attention down block matching Diffusers structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        cross_attention_dim: int = 768,
        num_attention_heads: int = 8,
        attention_head_dim: int = None,
        num_layers: int = 2,
        add_downsample: bool = True,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()

        if attention_head_dim is None:
            attention_head_dim = out_channels // num_attention_heads

        # Residual blocks
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                UNetResidualBlock(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    time_embedding_dim=time_embedding_dim,
                )
            )

        # Attention blocks (one per resnet)
        self.attentions = nn.ModuleList()
        for i in range(num_layers):
            self.attentions.append(
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=out_channels,
                    cross_attention_dim=cross_attention_dim,
                    num_layers=1,
                    prefix=maybe_prefix(prefix, f"attentions.{i}"),
                )
            )

        # Downsampler
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels)])
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, time_emb)
            hidden_states = attn(hidden_states, encoder_hidden_states)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class DownBlock2D(nn.Module):
    """Down block without attention matching Diffusers structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        num_layers: int = 2,
        add_downsample: bool = True,
        prefix: str = "",
    ):
        super().__init__()

        # Residual blocks
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                UNetResidualBlock(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    time_embedding_dim=time_embedding_dim,
                )
            )

        # Downsampler
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels)])
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, time_emb)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    """Cross-attention up block matching Diffusers structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        time_embedding_dim: int,
        cross_attention_dim: int = 768,
        num_attention_heads: int = 8,
        attention_head_dim: int = None,
        num_layers: int = 3,
        add_upsample: bool = True,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()

        if attention_head_dim is None:
            attention_head_dim = out_channels // num_attention_heads

        # Residual blocks with exact Diffusers channel calculations
        self.resnets = nn.ModuleList()

        if out_channels == 1280:  # up_blocks[1]
            resnet_input_channels = [2560, 2560, 1920]
        elif out_channels == 640:  # up_blocks[2]
            resnet_input_channels = [1920, 1280, 960]
        elif out_channels == 320:  # up_blocks[3]
            resnet_input_channels = [960, 640, 640]
        else:
            # Fallback calculation
            skip_channels = self._get_skip_channels(out_channels, num_layers)
            resnet_input_channels = []
            for i in range(num_layers):
                if i == 0:
                    resnet_in_channels = prev_output_channel + skip_channels[i]
                else:
                    resnet_in_channels = out_channels + skip_channels[i]
                resnet_input_channels.append(resnet_in_channels)

        for i, resnet_in_channels in enumerate(resnet_input_channels):
            self.resnets.append(
                UNetResidualBlock(
                    in_channels=resnet_in_channels,
                    out_channels=out_channels,
                    time_embedding_dim=time_embedding_dim,
                )
            )

        # Attention blocks (one per resnet)
        self.attentions = nn.ModuleList()
        for i in range(num_layers):
            self.attentions.append(
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=out_channels,
                    cross_attention_dim=cross_attention_dim,
                    num_layers=1,
                    prefix=maybe_prefix(prefix, f"attentions.{i}"),
                )
            )

        # Upsampler
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels)])
        else:
            self.upsamplers = None

    def _get_skip_channels(
        self,
        out_channels: int,
        num_layers: int,
    ) -> list[int]:
        """Get the expected skip channel counts to match Diffusers exactly."""
        # These patterns are from exact Diffusers analysis
        if out_channels == 1280:  # up_blocks[1]
            return [1280, 1280, 640]
        elif out_channels == 640:  # up_blocks[2]
            return [1280, 640, 320]
        elif out_channels == 320:  # up_blocks[3]
            return [640, 320, 320]
        else:
            # Fallback for other configurations
            return [out_channels] * num_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: tuple[torch.Tensor, ...],
        time_emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Convert tuple to list for popping (matching Diffusers implementation)
        res_hidden_states_list = list(res_hidden_states_tuple)

        for resnet, attn in zip(self.resnets, self.attentions):
            # Pop skip connection from the end
            # (LIFO - matches Diffusers exactly)
            res_hidden_states = res_hidden_states_list.pop()

            # Concatenate input with skip connection
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # Apply ResNet and attention
            hidden_states = resnet(hidden_states, time_emb)
            hidden_states = attn(hidden_states, encoder_hidden_states)

        # Upsample at the end (matches Diffusers implementation)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states


class UpBlock2D(nn.Module):
    """Up block without attention matching Diffusers structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        time_embedding_dim: int,
        num_layers: int = 3,
        add_upsample: bool = True,
        prefix: str = "",
    ):
        super().__init__()

        # Residual blocks with exact Diffusers channel calculations for
        # UpBlock2D (up_blocks[0])
        self.resnets = nn.ModuleList()

        # up_blocks[0] has all resnets with 2560 input channels
        # (from Diffusers analysis)
        resnet_input_channels = [2560, 2560, 2560]

        for i, resnet_in_channels in enumerate(resnet_input_channels):
            self.resnets.append(
                UNetResidualBlock(
                    in_channels=resnet_in_channels,
                    out_channels=out_channels,
                    time_embedding_dim=time_embedding_dim,
                )
            )

        # Upsampler
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels)])
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: tuple[torch.Tensor, ...],
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Convert tuple to list for popping (matching Diffusers implementation)
        res_hidden_states_list = list(res_hidden_states_tuple)

        for resnet in self.resnets:
            # Pop skip connection from the end
            # (LIFO - matches Diffusers exactly)
            res_hidden_states = res_hidden_states_list.pop()

            # Concatenate input with skip connection
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # Apply ResNet
            hidden_states = resnet(hidden_states, time_emb)

        # Upsample at the end (matches Diffusers implementation)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states


class UNetMidBlock2DCrossAttn(nn.Module):
    """Mid block with cross-attention matching Diffusers structure."""

    def __init__(
        self,
        in_channels: int,
        time_embedding_dim: int,
        cross_attention_dim: int = 768,
        num_attention_heads: int = 8,
        attention_head_dim: int = None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()

        if attention_head_dim is None:
            attention_head_dim = in_channels // num_attention_heads

        # 2 resnets with 1 attention in between
        self.resnets = nn.ModuleList(
            [
                UNetResidualBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    time_embedding_dim=time_embedding_dim,
                ),
                UNetResidualBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    time_embedding_dim=time_embedding_dim,
                ),
            ]
        )

        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=in_channels,
                    cross_attention_dim=cross_attention_dim,
                    num_layers=1,
                    prefix=maybe_prefix(prefix, "attentions.0"),
                )
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, time_emb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, time_emb)

        return hidden_states


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
        # Time embedding dimension (Diffusers uses block_out_channels[0] * 4)
        self.time_embedding_dim = config.time_embedding_dim or (self.block_out_channels[0] * 4)

        # Time embedding
        time_embed_dim = self.time_embedding_dim
        self.time_proj = lambda x: get_timestep_embedding(
            x,
            320,
            flip_sin_to_cos=True,
        )
        self.time_embedding = TimestepEmbedding(320, time_embed_dim)

        # Input convolution
        self.conv_in = nn.Conv2d(
            self.in_channels,
            self.block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        # Build down blocks matching Diffusers structure exactly
        self.down_blocks = nn.ModuleList([])

        # down_block_types: ['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D',
        # 'CrossAttnDownBlock2D', 'DownBlock2D']
        down_block_types = config.down_block_types

        for i, (down_block_type, out_channels) in enumerate(zip(down_block_types, self.block_out_channels)):
            in_channels = self.block_out_channels[i - 1] if i > 0 else self.block_out_channels[0]
            is_final_block = i == len(self.block_out_channels) - 1

            if down_block_type == "CrossAttnDownBlock2D":
                down_block = CrossAttnDownBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_embedding_dim=time_embed_dim,
                    cross_attention_dim=self.cross_attention_dim,
                    num_attention_heads=out_channels // self.attention_head_dim,
                    attention_head_dim=self.attention_head_dim,
                    num_layers=self.layers_per_block,
                    add_downsample=not is_final_block,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, f"down_blocks.{i}"),
                )
            elif down_block_type == "DownBlock2D":
                down_block = DownBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_embedding_dim=time_embed_dim,
                    num_layers=self.layers_per_block,
                    add_downsample=not is_final_block,
                    prefix=maybe_prefix(prefix, f"down_blocks.{i}"),
                )
            else:
                raise ValueError(f"Unsupported down_block_type: {down_block_type}")

            self.down_blocks.append(down_block)

        # Middle block - UNetMidBlock2DCrossAttn
        mid_block_channel = self.block_out_channels[-1]
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=mid_block_channel,
            time_embedding_dim=time_embed_dim,
            cross_attention_dim=self.cross_attention_dim,
            num_attention_heads=mid_block_channel // self.attention_head_dim,
            attention_head_dim=self.attention_head_dim,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "mid_block"),
        )

        # Build up blocks matching Diffusers structure exactly
        self.up_blocks = nn.ModuleList([])

        # up_block_types: ['UpBlock2D', 'CrossAttnUpBlock2D',
        # 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D']
        up_block_types = config.up_block_types
        reversed_block_out_channels = list(reversed(self.block_out_channels))

        for i, (up_block_type, out_channels) in enumerate(zip(up_block_types, reversed_block_out_channels)):  # noqa: E501
            prev_output_channel = reversed_block_out_channels[i - 1] if i > 0 else mid_block_channel
            is_final_block = i == len(self.block_out_channels) - 1

            if up_block_type == "CrossAttnUpBlock2D":
                up_block = CrossAttnUpBlock2D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    prev_output_channel=prev_output_channel,
                    time_embedding_dim=time_embed_dim,
                    cross_attention_dim=self.cross_attention_dim,
                    num_attention_heads=out_channels // self.attention_head_dim,  # noqa: E501
                    attention_head_dim=self.attention_head_dim,
                    num_layers=self.layers_per_block + 1,  # 3 for up blocks
                    add_upsample=not is_final_block,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, f"up_blocks.{i}"),
                )
            elif up_block_type == "UpBlock2D":
                up_block = UpBlock2D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    prev_output_channel=prev_output_channel,
                    time_embedding_dim=time_embed_dim,
                    num_layers=self.layers_per_block + 1,  # 3 for up blocks
                    add_upsample=not is_final_block,
                    prefix=maybe_prefix(prefix, f"up_blocks.{i}"),
                )
            else:
                raise ValueError(f"Unsupported up_block_type: {up_block_type}")

            self.up_blocks.append(up_block)

        # Output layers
        self.conv_norm_out = nn.GroupNorm(
            self.norm_num_groups,
            self.block_out_channels[0],
        )
        self.conv_out = nn.Conv2d(
            self.block_out_channels[0],
            self.out_channels,
            kernel_size=3,
            padding=1,
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
            encoder_hidden_states: (batch_size, seq_len, cross_attention_dim)
                text embeddings

        Returns:
            (batch_size, out_channels, height, width) predicted noise
        """
        # Time embedding
        timesteps = timestep.flatten()
        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        # Input convolution
        sample = self.conv_in(sample)

        # Down blocks - collect skip connections including the initial sample
        # The initial sample is also a skip connection for the final up block
        all_skip_connections = [sample]  # Include initial sample as first skip
        # connection

        for down_block in self.down_blocks:
            if hasattr(down_block, "forward"):
                # CrossAttnDownBlock2D or DownBlock2D
                if isinstance(down_block, CrossAttnDownBlock2D):
                    sample, res_samples = down_block(
                        sample,
                        emb,
                        encoder_hidden_states,
                    )
                elif isinstance(down_block, DownBlock2D):
                    sample, res_samples = down_block(sample, emb)
                else:
                    raise ValueError(
                        f"Unknown down block type: {type(down_block)}",
                    )

                # Store skip connections for later use
                all_skip_connections.extend(res_samples)

        # Mid block
        sample = self.mid_block(sample, emb, encoder_hidden_states)

        # Up blocks - distribute skip connections properly
        # Skip connections are consumed in LIFO order
        # (last produced, first consumed)
        skip_stack = list(reversed(all_skip_connections))

        for up_block in self.up_blocks:
            # Get the required number of skip connections for this block
            num_res_layers = len(up_block.resnets)

            # Extract skip connections for this block
            block_skip_connections = []
            for _ in range(num_res_layers):
                if skip_stack:
                    block_skip_connections.append(skip_stack.pop(0))
                else:
                    raise ValueError(
                        f"Insufficient skip connections for {type(up_block).__name__}",
                    )

            # Convert to tuple and pass to the block
            res_samples = tuple(reversed(block_skip_connections))

            if isinstance(up_block, CrossAttnUpBlock2D):
                sample = up_block(
                    sample,
                    res_samples,
                    emb,
                    encoder_hidden_states,
                )
            elif isinstance(up_block, UpBlock2D):
                sample = up_block(sample, res_samples, emb)
            else:
                raise ValueError(f"Unknown up block type: {type(up_block)}")

        # Output
        sample = self.conv_norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_out(sample)

        return sample

    def load_weights(self, weights) -> set[str]:
        """Load weights from a diffusers checkpoint."""
        from aphrodite.modeling.models.utils import AutoWeightsLoader

        # Create weight mapping from diffusers to our structure
        weight_mapping = self._create_weight_mapping()

        # Apply weight mapping to the weights
        mapped_weights = []
        for name, loaded_weight in weights:
            mapped_name = weight_mapping.get(name, name)
            mapped_weights.append((mapped_name, loaded_weight))

        # Use AutoWeightsLoader for the actual loading
        loader = AutoWeightsLoader(self)
        return loader.load_weights(mapped_weights)

    def _create_weight_mapping(self) -> dict[str, str]:
        """Create mapping from diffusers weight names to our weight names."""
        mapping = {}

        # Time embedding (direct mapping)
        mapping.update(
            {
                "time_embedding.linear_1.weight": "time_embedding.linear_1.weight",
                "time_embedding.linear_1.bias": "time_embedding.linear_1.bias",
                "time_embedding.linear_2.weight": "time_embedding.linear_2.weight",
                "time_embedding.linear_2.bias": "time_embedding.linear_2.bias",
            }
        )

        # Conv layers (direct mapping)
        mapping.update(
            {
                "conv_in.weight": "conv_in.weight",
                "conv_in.bias": "conv_in.bias",
                "conv_out.weight": "conv_out.weight",
                "conv_out.bias": "conv_out.bias",
                "conv_norm_out.weight": "conv_norm_out.weight",
                "conv_norm_out.bias": "conv_norm_out.bias",
            }
        )

        # Down blocks
        for block_idx in range(4):
            self._map_down_block(mapping, block_idx)

        # Mid block
        self._map_mid_block(mapping)

        # Up blocks
        for block_idx in range(4):
            self._map_up_block(mapping, block_idx)

        return mapping

    def _map_down_block(self, mapping: dict[str, str], block_idx: int):
        """Map down block weights."""
        block_prefix = f"down_blocks.{block_idx}"

        # ResNet blocks (2 per down block)
        for resnet_idx in range(2):
            self._map_resnet_block(
                mapping,
                f"{block_prefix}.resnets.{resnet_idx}",
            )

        # Attention blocks (only for first 3 down blocks)
        if block_idx < 3:
            for attn_idx in range(2):
                self._map_attention_block(
                    mapping,
                    f"{block_prefix}.attentions.{attn_idx}",
                )

        # Downsampler (except for last block)
        if block_idx < 3:
            mapping.update(
                {
                    f"{block_prefix}.downsamplers.0.conv.weight": (f"{block_prefix}.downsamplers.0.conv.weight"),
                    f"{block_prefix}.downsamplers.0.conv.bias": (f"{block_prefix}.downsamplers.0.conv.bias"),
                }
            )

    def _map_up_block(self, mapping: dict[str, str], block_idx: int):
        """Map up block weights."""
        block_prefix = f"up_blocks.{block_idx}"

        # ResNet blocks (3 per up block)
        for resnet_idx in range(3):
            self._map_resnet_block(
                mapping,
                f"{block_prefix}.resnets.{resnet_idx}",
            )

        # Attention blocks (for blocks 1, 2, 3)
        if block_idx > 0:
            for attn_idx in range(3):
                self._map_attention_block(
                    mapping,
                    f"{block_prefix}.attentions.{attn_idx}",
                )

        # Upsampler (except for last block)
        if block_idx < 3:
            mapping.update(
                {
                    f"{block_prefix}.upsamplers.0.conv.weight": (f"{block_prefix}.upsamplers.0.conv.weight"),
                    f"{block_prefix}.upsamplers.0.conv.bias": (f"{block_prefix}.upsamplers.0.conv.bias"),
                }
            )

    def _map_mid_block(self, mapping: dict[str, str]):
        """Map mid block weights."""
        # 2 ResNet blocks
        for resnet_idx in range(2):
            self._map_resnet_block(mapping, f"mid_block.resnets.{resnet_idx}")

        # 1 Attention block
        self._map_attention_block(mapping, "mid_block.attentions.0")

    def _map_resnet_block(self, mapping: dict[str, str], prefix: str):
        """Map ResNet block weights."""
        mapping.update(
            {
                f"{prefix}.norm1.weight": f"{prefix}.norm1.weight",
                f"{prefix}.norm1.bias": f"{prefix}.norm1.bias",
                f"{prefix}.conv1.weight": f"{prefix}.conv1.weight",
                f"{prefix}.conv1.bias": f"{prefix}.conv1.bias",
                f"{prefix}.time_emb_proj.weight": f"{prefix}.time_emb_proj.weight",
                f"{prefix}.time_emb_proj.bias": f"{prefix}.time_emb_proj.bias",
                f"{prefix}.norm2.weight": f"{prefix}.norm2.weight",
                f"{prefix}.norm2.bias": f"{prefix}.norm2.bias",
                f"{prefix}.conv2.weight": f"{prefix}.conv2.weight",
                f"{prefix}.conv2.bias": f"{prefix}.conv2.bias",
            }
        )

        # Optional shortcut connection (when input/output channels differ)
        mapping.update(
            {
                f"{prefix}.conv_shortcut.weight": f"{prefix}.conv_shortcut.weight",
                f"{prefix}.conv_shortcut.bias": f"{prefix}.conv_shortcut.bias",
            }
        )

    def _map_attention_block(self, mapping: dict[str, str], prefix: str):
        """Map attention block weights (Transformer2DModel)."""
        # Outer projection layers
        mapping.update(
            {
                f"{prefix}.norm.weight": f"{prefix}.norm.weight",
                f"{prefix}.norm.bias": f"{prefix}.norm.bias",
                f"{prefix}.proj_in.weight": f"{prefix}.proj_in.weight",
                f"{prefix}.proj_in.bias": f"{prefix}.proj_in.bias",
                f"{prefix}.proj_out.weight": f"{prefix}.proj_out.weight",
                f"{prefix}.proj_out.bias": f"{prefix}.proj_out.bias",
            }
        )

        # Transformer blocks (usually just 1)
        for transformer_idx in range(1):
            # SD 1.5 has 1 transformer block per attention
            self._map_transformer_block(
                mapping,
                f"{prefix}.transformer_blocks.{transformer_idx}",
            )

    def _map_transformer_block(self, mapping: dict[str, str], prefix: str):
        """Map BasicTransformerBlock weights."""
        # Layer norms
        mapping.update(
            {
                f"{prefix}.norm1.weight": f"{prefix}.norm1.weight",
                f"{prefix}.norm1.bias": f"{prefix}.norm1.bias",
                f"{prefix}.norm2.weight": f"{prefix}.norm2.weight",
                f"{prefix}.norm2.bias": f"{prefix}.norm2.bias",
                f"{prefix}.norm3.weight": f"{prefix}.norm3.weight",
                f"{prefix}.norm3.bias": f"{prefix}.norm3.bias",
            }
        )

        # Self-attention (attn1)
        mapping.update(
            {
                f"{prefix}.attn1.to_q.weight": f"{prefix}.attn1.to_q.weight",
                f"{prefix}.attn1.to_k.weight": f"{prefix}.attn1.to_k.weight",
                f"{prefix}.attn1.to_v.weight": f"{prefix}.attn1.to_v.weight",
                f"{prefix}.attn1.to_out.0.weight": (f"{prefix}.attn1.to_out.0.weight"),
                f"{prefix}.attn1.to_out.0.bias": f"{prefix}.attn1.to_out.0.bias",
            }
        )

        # Cross-attention (attn2)
        mapping.update(
            {
                f"{prefix}.attn2.to_q.weight": f"{prefix}.attn2.to_q.weight",
                f"{prefix}.attn2.to_k.weight": f"{prefix}.attn2.to_k.weight",
                f"{prefix}.attn2.to_v.weight": f"{prefix}.attn2.to_v.weight",
                f"{prefix}.attn2.to_out.0.weight": (f"{prefix}.attn2.to_out.0.weight"),
                f"{prefix}.attn2.to_out.0.bias": f"{prefix}.attn2.to_out.0.bias",
            }
        )

        # Feedforward (GeGLU)
        mapping.update(
            {
                f"{prefix}.ff.net.0.proj.weight": (f"{prefix}.ff.net.0.0.weight"),  # GeGLU proj layer
                f"{prefix}.ff.net.0.proj.bias": f"{prefix}.ff.net.0.0.bias",
                f"{prefix}.ff.net.2.weight": f"{prefix}.ff.net.2.weight",
                f"{prefix}.ff.net.2.bias": f"{prefix}.ff.net.2.bias",
            }
        )
