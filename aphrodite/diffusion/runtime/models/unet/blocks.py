# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from ComfyUI: comfy/ldm/modules/diffusionmodules/openaimodel.py

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from aphrodite.diffusion.runtime.layers.linear import ReplicatedLinear
from aphrodite.diffusion.runtime.models.unet.utils import avg_pool_nd, checkpoint


class TimestepBlock(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""
    
    @torch.jit.export
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class Upsample(nn.Module):
    """An upsampling layer with an optional convolution."""
    
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: int | None = None,
        padding: int = 1,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        
        if use_conv:
            if dims == 2:
                self.conv = nn.Conv2d(
                    self.channels, self.out_channels, 3, padding=padding, dtype=dtype, device=device
                )
            elif dims == 3:
                self.conv = nn.Conv3d(
                    self.channels, self.out_channels, 3, padding=padding, dtype=dtype, device=device
                )
            else:
                raise ValueError(f"unsupported dimensions: {dims}")
        else:
            self.conv = None
    
    def forward(self, x: torch.Tensor, output_shape: tuple | None = None) -> torch.Tensor:
        assert x.shape[1] == self.channels
        
        if self.dims == 3:
            shape = [x.shape[2], x.shape[3] * 2, x.shape[4] * 2]
            if output_shape is not None:
                shape[1] = output_shape[3]
                shape[2] = output_shape[4]
        else:
            shape = [x.shape[2] * 2, x.shape[3] * 2]
            if output_shape is not None:
                shape[0] = output_shape[2]
                shape[1] = output_shape[3]
        
        x = F.interpolate(x, size=shape, mode="nearest")
        if self.use_conv and self.conv is not None:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """A downsampling layer with an optional convolution."""
    
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: int | None = None,
        padding: int = 1,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        
        stride = 2 if dims != 3 else (1, 2, 2)
        
        if use_conv:
            if dims == 2:
                self.op = nn.Conv2d(
                    self.channels, self.out_channels, 3, stride=stride, padding=padding,
                    dtype=dtype, device=device
                )
            elif dims == 3:
                self.op = nn.Conv3d(
                    self.channels, self.out_channels, 3, stride=stride, padding=padding,
                    dtype=dtype, device=device
                )
            else:
                raise ValueError(f"unsupported dimensions: {dims}")
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    """
    
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: int | None = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        kernel_size: int = 3,
        exchange_temb_dims: bool = False,
        skip_t_emb: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims
        
        if isinstance(kernel_size, list):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2
        
        # Input layers
        if dims == 2:
            conv_cls = nn.Conv2d
        elif dims == 3:
            conv_cls = nn.Conv3d
        else:
            raise ValueError(f"unsupported dimensions: {dims}")
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels, dtype=dtype, device=device),
            nn.SiLU(),
            conv_cls(channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device),
        )
        
        # Upsample/downsample
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        
        # Timestep embedding layers
        self.skip_t_emb = skip_t_emb
        if not self.skip_t_emb:
            emb_out_dim = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
            emb_linear = ReplicatedLinear(
                emb_channels, emb_out_dim, bias=True, params_dtype=dtype
            )
            self.emb_layers = nn.ModuleList([nn.SiLU(), emb_linear])
        else:
            self.emb_layers = None
        
        # Output layers
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_cls(self.out_channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device),
        )
        
        # Skip connection
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_cls(
                channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device
            )
        else:
            self.skip_connection = conv_cls(
                channels, self.out_channels, 1, dtype=dtype, device=device
            )
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)
    
    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # Check for NaN in inputs
        if torch.isnan(x).any():
            raise ValueError(f"NaN in ResBlock input x!")
        if torch.isnan(emb).any():
            raise ValueError(f"NaN in ResBlock timestep embedding emb!")
        
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        # Check for NaN after in_layers
        if torch.isnan(h).any():
            raise ValueError(f"NaN after ResBlock in_layers!")
        
        emb_out = None
        if not self.skip_t_emb and self.emb_layers is not None:
            # Apply SiLU then linear
            emb_silu = self.emb_layers[0](emb)
            emb_out, _ = self.emb_layers[1](emb_silu)
            # Check for NaN after emb processing
            if torch.isnan(emb_out).any():
                raise ValueError(f"NaN after ResBlock emb_layers! emb stats: min={emb.min().item():.6f}, max={emb.max().item():.6f}")
            emb_out = emb_out.type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            if emb_out is not None:
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = h * (1 + scale) + shift
            h = out_rest(h)
        else:
            if emb_out is not None:
                if self.exchange_temb_dims:
                    emb_out = emb_out.movedim(1, 2)
                h = h + emb_out
            h = self.out_layers(h)
        
        # Check for NaN after out_layers
        if torch.isnan(h).any():
            raise ValueError(f"NaN after ResBlock out_layers!")
        
        skip = self.skip_connection(x)
        result = skip + h
        
        # Check for NaN in final output
        if torch.isnan(result).any():
            raise ValueError(f"NaN in ResBlock final output! skip stats: min={skip.min().item():.6f}, max={skip.max().item():.6f}, h stats: min={h.min().item():.6f}, max={h.max().item():.6f}")
        
        return result

