"""
Implementation of AutoencoderKL (VAE) for Stable Diffusion in Aphrodite.
"""
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aphrodite.common.logger import logger
from aphrodite.common.sequence import IntermediateTensors
from aphrodite.config import AphroditeConfig
from aphrodite.modeling.model_loader.weight_utils import default_weight_loader
from aphrodite.modeling.models.interfaces import SupportsQuant


class Downsample2D(nn.Module):
    """Downsampling layer matching diffusers structure."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Asymmetric padding for downsampling (matches diffusers)
        x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class Upsample2D(nn.Module):
    """Upsampling layer matching diffusers structure."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class VAEResidualBlock(nn.Module):
    """
    Residual block used in VAE encoder/decoder, matching diffusers structure.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
    ):
        super().__init__()
        # Match diffusers naming exactly
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.nonlinearity = nn.SiLU()

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
            )
        else:
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return x + residual


class VAEAttentionBlock(nn.Module):
    """
    Self-attention block used in VAE encoder/decoder, matching diffusers
    structure.
    """

    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        # Match diffusers naming exactly
        self.group_norm = nn.GroupNorm(groups, channels)

                # Use Linear layers to match diffusers exactly (with bias)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)

        # Diffusers uses ModuleList for to_out: [Linear, Dropout]
        self.to_out = nn.ModuleList([
            nn.Linear(channels, channels, bias=True),
            nn.Identity(),  # Dropout placeholder (usually disabled)
        ])

        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.group_norm(x)

        B, C, H, W = x.shape

        # Reshape for linear layers: (B, C, H, W) -> (B, H*W, C)
        x = x.view(B, C, H * W).transpose(1, 2)

        # Get q, k, v
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Compute attention
        attn_weights = torch.softmax(
            q @ k.transpose(-1, -2) * self.scale,
            dim=-1,
        )
        attn_output = attn_weights @ v

        # Apply output projection (ModuleList: [Linear, Dropout])
        attn_output = self.to_out[0](attn_output)  # Linear layer
        attn_output = self.to_out[1](attn_output)  # Dropout (Identity)

        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        attn_output = attn_output.transpose(1, 2).view(B, C, H, W)

        return attn_output + residual


class VAEDownEncoderBlock(nn.Module):
    """Down encoder block matching diffusers structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        add_downsample: bool = True,
    ):
        super().__init__()

        # Residual layers (called "resnets" in diffusers)
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(VAEResidualBlock(in_ch, out_channels))

        # Downsampler (matching diffusers structure)
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels)
            ])
        else:
            self.downsamplers = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply residual blocks
        for resnet in self.resnets:
            x = resnet(x)

        # Apply downsampling if present
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)

        return x


class VAEUpDecoderBlock(nn.Module):
    """Up decoder block matching diffusers structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        add_upsample: bool = True,
    ):
        super().__init__()

        # Residual layers (called "resnets" in diffusers)
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(VAEResidualBlock(in_ch, out_channels))

        # Upsampler (matching diffusers structure)
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(out_channels)
            ])
        else:
            self.upsamplers = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply residual blocks
        for resnet in self.resnets:
            x = resnet(x)

        # Apply upsampling if present
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)

        return x


class VAEMidBlock(nn.Module):
    """Mid block with attention matching diffusers structure."""

    def __init__(self, channels: int):
        super().__init__()

        self.resnets = nn.ModuleList([
            VAEResidualBlock(channels, channels),
            VAEResidualBlock(channels, channels),
        ])

        self.attentions = nn.ModuleList([
            VAEAttentionBlock(channels),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First resnet
        x = self.resnets[0](x)

        # Attention
        x = self.attentions[0](x)

        # Second resnet
        x = self.resnets[1](x)

        return x


class VAEEncoder(nn.Module):
    """VAE Encoder that converts images to latent space."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: list[int] = None,
        layers_per_block: int = 2,
    ):
        if block_out_channels is None:
            block_out_channels = [128, 256, 512, 512]
        super().__init__()

        # Initial conv
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        # Downsampling blocks (matching diffusers structure)
        self.down_blocks = nn.ModuleList()
        in_ch = block_out_channels[0]

        for i, out_ch in enumerate(block_out_channels):
            # All blocks except the last have downsamplers
            add_downsample = (i < len(block_out_channels) - 1)

            block = VAEDownEncoderBlock(
                in_ch,
                out_ch,
                layers_per_block,
                add_downsample,
            )
            self.down_blocks.append(block)
            in_ch = out_ch

        # Middle block with attention
        self.mid_block = VAEMidBlock(in_ch)

        # Output layers
        self.conv_norm_out = nn.GroupNorm(32, in_ch)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            in_ch, latent_channels * 2, kernel_size=3, padding=1,
        )  # *2 for mean and logvar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.conv_in(x)

        # Downsampling blocks
        for block in self.down_blocks:
            x = block(x)

        # Middle block
        x = self.mid_block(x)

        # Output
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x


class VAEDecoder(nn.Module):
    """VAE Decoder that converts latents back to images."""

    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 3,
        block_out_channels: list[int] = None,
        layers_per_block: int = 2,
    ):
        if block_out_channels is None:
            block_out_channels = [128, 256, 512, 512]
        super().__init__()

        # Decoder starts with the highest channel count
        # block_out_channels for decoder: [512, 512, 256, 128]
        decoder_channels = list(reversed(block_out_channels))

        # Initial conv
        self.conv_in = nn.Conv2d(
            latent_channels, decoder_channels[0], kernel_size=3, padding=1,
        )

        # Middle block with attention
        self.mid_block = VAEMidBlock(decoder_channels[0])

        # Upsampling blocks (matching diffusers structure)
        self.up_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]

        for i, out_ch in enumerate(decoder_channels):
            # All blocks except the last have upsampling
            add_upsample = (i < len(decoder_channels) - 1)

            # All decoder blocks have 3 resnets
            num_layers = 3

            block = VAEUpDecoderBlock(in_ch, out_ch, num_layers, add_upsample)
            self.up_blocks.append(block)
            in_ch = out_ch

        # Output layers
        self.conv_norm_out = nn.GroupNorm(32, in_ch)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.conv_in(x)

        # Middle block
        x = self.mid_block(x)

        # Upsampling blocks
        for block in self.up_blocks:
            x = block(x)

        # Output
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x


class AutoencoderKL(nn.Module, SupportsQuant):
    """
    Variational Autoencoder (VAE) model for Stable Diffusion.

    This implementation is compatible with diffusers AutoencoderKL and supports
    efficient inference with quantization and memory optimizations.
    """

    # Mark this as a VAE model (not pooling)
    is_vae_model = True

    def __init__(
        self,
        *,
        aphrodite_config: AphroditeConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = aphrodite_config.model_config.hf_config

        # Extract config parameters
        self.in_channels = getattr(config, 'in_channels', 3)
        self.out_channels = getattr(config, 'out_channels', 3)
        self.latent_channels = getattr(config, 'latent_channels', 4)
        self.block_out_channels = getattr(
            config, 'block_out_channels', [128, 256, 512, 512],
        )
        self.layers_per_block = getattr(config, 'layers_per_block', 2)
        self.scaling_factor = getattr(config, 'scaling_factor', 0.18215)

        # Build encoder and decoder
        self.encoder = VAEEncoder(
            in_channels=self.in_channels,
            latent_channels=self.latent_channels,
            block_out_channels=self.block_out_channels,
            layers_per_block=self.layers_per_block,
        )

        self.decoder = VAEDecoder(
            latent_channels=self.latent_channels,
            out_channels=self.out_channels,
            block_out_channels=self.block_out_channels,
            layers_per_block=self.layers_per_block,
        )

        # Quantization layers for latent distribution
        self.quant_conv = nn.Conv2d(
            self.latent_channels * 2, self.latent_channels * 2, kernel_size=1,
        )
        self.post_quant_conv = nn.Conv2d(
            self.latent_channels, self.latent_channels, kernel_size=1,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.

        Args:
            x: Input images of shape (B, C, H, W)

        Returns:
            Latent representations of shape (B, latent_channels, H//8, W//8)
        """
        # Encode to latent distribution parameters
        h = self.encoder(x)
        h = self.quant_conv(h)

        # Split into mean and logvar
        mean, logvar = torch.chunk(h, 2, dim=1)

        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, -30.0, 20.0)

        # Sample from the distribution (using reparameterization trick)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(mean)

        # Apply scaling factor
        z = z * self.scaling_factor

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to images.

        Args:
            z: Latent representations of shape (B, latent_channels, H, W)

        Returns:
            Reconstructed images of shape (B, out_channels, H*8, W*8)
        """
        # Remove scaling factor
        z = z / self.scaling_factor

        # Apply post-quantization conv
        z = self.post_quant_conv(z)

        # Decode to images
        x = self.decoder(z)

        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for compatibility with Aphrodite model interface.
        For VAE, we expect the input to be images for encoding.
        """
        if inputs_embeds is not None:
            # If inputs_embeds is provided, treat it as image data
            return self.encode(inputs_embeds)
        else:
            raise NotImplementedError(
                "VAE requires image data via inputs_embeds parameter",
            )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load weights from diffusers checkpoint format."""

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Handle old diffusers naming convention for attention layers
            mapped_name = name
            if "attentions.0.key." in name:
                mapped_name = name.replace(
                    "attentions.0.key.", "attentions.0.to_k.",
                )
            elif "attentions.0.query." in name:
                mapped_name = name.replace(
                    "attentions.0.query.", "attentions.0.to_q.",
                )
            elif "attentions.0.value." in name:
                mapped_name = name.replace(
                    "attentions.0.value.", "attentions.0.to_v.",
                )
            elif "attentions.0.proj_attn." in name:
                mapped_name = name.replace(
                    "attentions.0.proj_attn.", "attentions.0.to_out.0.",
                )

            if mapped_name in params_dict:
                param = params_dict[mapped_name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader,
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped_name)
            elif name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader,
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            else:
                logger.debug(
                    f"Weight {name} (mapped to {mapped_name}) not found in "
                    "model parameters",
                )

        unloaded = set(params_dict.keys()) - loaded_params
        if unloaded:
            logger.debug(f"Unloaded parameters: {list(unloaded)[:10]}")

        return loaded_params

    def _create_weight_mapping(self) -> dict[str, str]:
        """Create mapping from diffusers weight names to our weight names."""
        return {}
