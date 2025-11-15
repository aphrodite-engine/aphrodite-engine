# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from ComfyUI: comfy/ldm/modules/attention.py

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import logging

from aphrodite.diffusion.runtime.layers.layernorm import LayerNorm
from aphrodite.diffusion.runtime.layers.linear import ReplicatedLinear

logger = logging.getLogger(__name__)


class CrossAttention(nn.Module):
    """Cross-attention layer for UNet."""
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.to_q = ReplicatedLinear(query_dim, inner_dim, bias=False, params_dtype=dtype)
        self.to_k = ReplicatedLinear(context_dim, inner_dim, bias=False, params_dtype=dtype)
        self.to_v = ReplicatedLinear(context_dim, inner_dim, bias=False, params_dtype=dtype)
        
        self.to_out = nn.ModuleList([
            ReplicatedLinear(inner_dim, query_dim, bias=True, params_dtype=dtype),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention.
        Based on ComfyUI's CrossAttention implementation.
        """
        h = self.heads
        
        # Check for NaN in inputs
        if torch.isnan(x).any():
            raise ValueError(f"NaN in CrossAttention input x! x shape: {x.shape}")
        if context is not None and torch.isnan(context).any():
            raise ValueError(f"NaN in CrossAttention context! context shape: {context.shape}")
        
        # Check input x before projections
        x_max_abs = x.abs().max().item()
        if x_max_abs > 1000:
            logger.warning(f"Large input x to CrossAttention! x_max_abs={x_max_abs:.2f}, x stats: min={x.min().item():.6f}, max={x.max().item():.6f}")
            # Clamp x to prevent extreme values
            x = torch.clamp(x, min=-1000.0, max=1000.0)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning(f"NaN/inf in CrossAttention input x! x stats: min={x.min().item():.6f}, max={x.max().item():.6f}")
            x = torch.clamp(x, min=-1e4, max=1e4)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        
        q, _ = self.to_q(x)
        context = context if context is not None else x
        
        # Check context values (text embeddings) - they might be extreme
        if context is not None and context is not x:
            context_max_abs = context.abs().max().item()
            if context_max_abs > 1000:
                logger.warning(f"Large context values in CrossAttention! context_max_abs={context_max_abs:.2f}, "
                             f"context stats: min={context.min().item():.6f}, max={context.max().item():.6f}")
                # Clamp context to prevent extreme values
                context = torch.clamp(context, min=-1000.0, max=1000.0)
        
        k, _ = self.to_k(context)
        v, _ = self.to_v(context)
        
        # Check for NaN/inf after QKV projections and log stats
        if torch.isnan(q).any():
            raise ValueError(f"NaN in q after to_q projection!")
        if torch.isnan(k).any():
            raise ValueError(f"NaN in k after to_k projection! context shape: {context.shape}")
        if torch.isnan(v).any():
            raise ValueError(f"NaN in v after to_v projection!")
        
        # Log Q/K/V stats to understand magnitude
        if torch.isinf(q).any() or torch.isinf(k).any():
            logger.warning(f"Inf in Q/K after projections! q stats: min={q.min().item():.6f}, max={q.max().item():.6f}, "
                         f"k stats: min={k.min().item():.6f}, max={k.max().item():.6f}, "
                         f"input x stats: min={x.min().item():.6f}, max={x.max().item():.6f}")
            # Clamp Q/K to prevent inf propagation
            q = torch.clamp(q, min=-1e4, max=1e4)
            k = torch.clamp(k, min=-1e4, max=1e4)
            v = torch.clamp(v, min=-1e4, max=1e4)
        
        # Reshape for attention (ComfyUI format: [B, N, H*D] -> [B*H, N, D])
        # Based on ComfyUI's attention_basic function
        b, n, _ = q.shape
        q = q.view(b, n, h, self.dim_head).permute(0, 2, 1, 3).reshape(b * h, n, self.dim_head).contiguous()
        k = k.view(b, -1, h, self.dim_head).permute(0, 2, 1, 3).reshape(b * h, -1, self.dim_head).contiguous()
        v = v.view(b, -1, h, self.dim_head).permute(0, 2, 1, 3).reshape(b * h, -1, self.dim_head).contiguous()
        
        # Scaled dot-product attention (ComfyUI style)
        # Force cast to float32 to avoid overflowing (ComfyUI does this conditionally)
        # Compute attention scores: [B*H, N, D] x [B*H, M, D]^T -> [B*H, N, M]
        # Always use float32 for attention computation to match ComfyUI's behavior when attn_precision is float32
        q_fp32 = q.float()
        k_fp32 = k.float()
        # Check for NaN/inf before einsum
        if torch.isnan(q_fp32).any() or torch.isnan(k_fp32).any():
            raise ValueError(f"NaN in q or k after float conversion!")
        if torch.isinf(q_fp32).any() or torch.isinf(k_fp32).any():
            # Clamp q and k to prevent inf values (these shouldn't happen, but handle gracefully)
            logger.warning(f"Inf values in q or k before einsum! q stats: min={q_fp32.min().item():.6f}, max={q_fp32.max().item():.6f}, k stats: min={k_fp32.min().item():.6f}, max={k_fp32.max().item():.6f}")
            q_fp32 = torch.clamp(q_fp32, min=-1e4, max=1e4)
            k_fp32 = torch.clamp(k_fp32, min=-1e4, max=1e4)
        
        # Log Q/K stats before einsum to understand magnitude
        q_max_abs = q_fp32.abs().max().item()
        k_max_abs = k_fp32.abs().max().item()
        if q_max_abs > 1000 or k_max_abs > 1000:
            # Check weight values to see if they're reasonable
            to_q_weight_max = self.to_q.weight.abs().max().item() if hasattr(self.to_q, 'weight') else 'N/A'
            to_k_weight_max = self.to_k.weight.abs().max().item() if hasattr(self.to_k, 'weight') else 'N/A'
            
            logger.warning(f"Large Q/K values before einsum! q_max_abs={q_max_abs:.2e}, k_max_abs={k_max_abs:.2e}, "
                         f"q stats: min={q_fp32.min().item():.6e}, max={q_fp32.max().item():.6e}, "
                         f"k stats: min={k_fp32.min().item():.6e}, max={k_fp32.max().item():.6e}, "
                         f"to_q_weight_max={to_q_weight_max}, to_k_weight_max={to_k_weight_max}, "
                         f"input x_max_abs={x_max_abs:.2f}, context_max_abs={context.abs().max().item() if context is not None else 'N/A'}")
            
            # Clamp Q/K aggressively to prevent overflow
            q_fp32 = torch.clamp(q_fp32, min=-1000.0, max=1000.0)
            k_fp32 = torch.clamp(k_fp32, min=-1000.0, max=1000.0)
        
        sim = torch.einsum('b i d, b j d -> b i j', q_fp32, k_fp32) * self.scale
        
        # Check for NaN/inf after einsum and clamp if needed
        if torch.isnan(sim).any():
            raise ValueError(f"NaN in sim after einsum! q stats: min={q.min().item():.6f}, max={q.max().item():.6f}, k stats: min={k.min().item():.6f}, max={k.max().item():.6f}, scale={self.scale}")
        if torch.isinf(sim).any():
            # Clamp sim to prevent inf values (this can happen if q/k are very large)
            logger.warning(f"Inf values in sim after einsum! q_max_abs={q_max_abs:.2f}, k_max_abs={k_max_abs:.2f}, "
                         f"sim_max_before_clamp={sim[~torch.isinf(sim)].max().item() if not torch.all(torch.isinf(sim)) else 'all_inf'}")
            sim = torch.clamp(sim, min=-500.0, max=500.0)
        
        if mask is not None:
            if mask.dtype == torch.bool:
                max_neg_value = -torch.finfo(sim.dtype).max
                mask_expanded = mask.unsqueeze(0).expand(b * h, -1, -1)
                sim.masked_fill_(~mask_expanded, max_neg_value)
            else:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                mask = mask.expand(b * h, -1, -1)
                sim = sim + mask
        
        # Check for NaN before softmax
        if torch.isnan(sim).any():
            raise ValueError(f"NaN in sim before softmax!")
        
        # Attention weights (ComfyUI does softmax directly without clamping/subtracting max)
        # The float32 casting above should be sufficient to prevent overflow
        attn = sim.softmax(dim=-1)
        
        # Check for NaN after softmax
        if torch.isnan(attn).any():
            raise ValueError(f"NaN in attn after softmax! sim stats: min={sim.min().item():.6f}, max={sim.max().item():.6f}, sim has inf: {torch.isinf(sim).any().item()}")
        
        out = torch.einsum('b i j, b j d -> b i d', attn.to(v.dtype), v)
        
        # Check for NaN/inf after final einsum
        if torch.isnan(out).any():
            raise ValueError(f"NaN in out after final einsum!")
        if torch.isinf(out).any():
            # Clamp inf values to prevent propagation
            out = torch.clamp(out, min=-1e6, max=1e6)
        
        # Reshape back: [B*H, N, D] -> [B, N, H*D]
        out = out.view(b, h, n, self.dim_head).permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        
        # Check for NaN/inf after reshape
        if torch.isnan(out).any() or torch.isinf(out).any():
            # Clamp again after reshape
            out = torch.clamp(out, min=-1e6, max=1e6)
        
        # Apply output projection
        out_proj, _ = self.to_out[0](out)
        out = self.to_out[1](out_proj)
        
        return out


class BasicTransformerBlock(nn.Module):
    """Basic transformer block with self-attention and cross-attention."""
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: int | None = None,
        gated_ff: bool = True,
        checkpoint: bool = True,
        disable_self_attn: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.disable_self_attn = disable_self_attn
        
        # Self-attention
        if not disable_self_attn:
            self.attn1 = CrossAttention(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                dtype=dtype,
                device=device,
            )
            self.norm1 = LayerNorm(dim, eps=1e-6)
        
        # Cross-attention
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            dtype=dtype,
            device=device,
        )
        self.norm2 = LayerNorm(dim, eps=1e-6)
        
        # Feed-forward
        ff_mult = 4
        inner_dim = int(dim * ff_mult)
        if gated_ff:
            # GEGLU
            self.ff_0 = ReplicatedLinear(dim, inner_dim * 2, bias=True, params_dtype=dtype)
            self.ff_act = nn.GELU()
            self.ff_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.ff_2 = ReplicatedLinear(inner_dim, dim, bias=True, params_dtype=dtype)
            self.ff_gated = True
        else:
            self.ff_0 = ReplicatedLinear(dim, inner_dim, bias=True, params_dtype=dtype)
            self.ff_act = nn.GELU()
            self.ff_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.ff_2 = ReplicatedLinear(inner_dim, dim, bias=True, params_dtype=dtype)
            self.ff_gated = False
        self.norm3 = LayerNorm(dim, eps=1e-6)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, context, use_reentrant=False)
        return self._forward(x, context)
    
    def _forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Check for NaN/inf in input
        if torch.isnan(x).any():
            raise ValueError(f"NaN in BasicTransformerBlock input x! x shape: {x.shape}")
        if torch.isinf(x).any():
            # Clamp inf values at the input to prevent propagation
            x = torch.clamp(x, min=-1e6, max=1e6)
            if torch.isnan(x).any():
                raise ValueError(f"NaN after clamping inf in BasicTransformerBlock input!")
        
        # Self-attention
        if not self.disable_self_attn:
            x = x.contiguous()  # Ensure contiguous for LayerNorm
            x_norm = self.norm1(x)
            # Check for NaN after norm1
            if torch.isnan(x_norm).any():
                raise ValueError(f"NaN after BasicTransformerBlock norm1! x stats before norm1: min={x.min().item():.6f}, max={x.max().item():.6f}, has_inf={torch.isinf(x).any().item()}")
            # Handle ReplicatedLinear tuple return
            attn1_out = self.attn1(x_norm)
            if isinstance(attn1_out, tuple):
                attn1_out = attn1_out[0]
            # Check for NaN/inf after attn1
            if torch.isnan(attn1_out).any():
                raise ValueError(f"NaN after BasicTransformerBlock attn1!")
            if torch.isinf(attn1_out).any():
                # Clamp inf values to prevent propagation
                attn1_out = torch.clamp(attn1_out, min=-1e6, max=1e6)
            x = x + attn1_out
            # Check for NaN/inf after adding attn1
            if torch.isnan(x).any() or torch.isinf(x).any():
                # Clamp to prevent inf propagation
                x = torch.clamp(x, min=-1e6, max=1e6)
                if torch.isnan(x).any():
                    raise ValueError(f"NaN after adding attn1 output!")
        
        # Cross-attention
        x = x.contiguous()  # Ensure contiguous for LayerNorm
        x_norm = self.norm2(x)
        # Check for NaN after norm2
        if torch.isnan(x_norm).any():
            raise ValueError(f"NaN after BasicTransformerBlock norm2!")
        attn2_out = self.attn2(x_norm, context)
        if isinstance(attn2_out, tuple):
            attn2_out = attn2_out[0]
        # Check for NaN/inf after attn2
        if torch.isnan(attn2_out).any():
            raise ValueError(f"NaN after BasicTransformerBlock attn2!")
        if torch.isinf(attn2_out).any():
            # Clamp inf values to prevent propagation
            attn2_out = torch.clamp(attn2_out, min=-1e6, max=1e6)
        x = x + attn2_out
        # Check for NaN/inf after adding attn2
        if torch.isnan(x).any() or torch.isinf(x).any():
            # Clamp to prevent inf propagation
            x = torch.clamp(x, min=-1e6, max=1e6)
            if torch.isnan(x).any():
                raise ValueError(f"NaN after adding attn2 output!")
        
        # Feed-forward
        x = x.contiguous()  # Ensure contiguous for LayerNorm
        # Check for NaN before norm3
        if torch.isnan(x).any():
            raise ValueError(f"NaN in x before BasicTransformerBlock norm3! x stats: min={x.min().item():.6f}, max={x.max().item():.6f}")
        x_norm = self.norm3(x)
        # Check for NaN after norm3
        if torch.isnan(x_norm).any():
            raise ValueError(f"NaN after BasicTransformerBlock norm3! x stats: min={x.min().item():.6f}, max={x.max().item():.6f}, x_norm stats: min={x_norm.min().item():.6f}, max={x_norm.max().item():.6f}")
        ff_out, _ = self.ff_0(x_norm)
        # Check for NaN after ff_0
        if torch.isnan(ff_out).any():
            raise ValueError(f"NaN after BasicTransformerBlock ff_0!")
        if self.ff_gated:
            # GEGLU: split and gate
            ff_out, gate = ff_out.chunk(2, dim=-1)
            ff_out = ff_out * self.ff_act(gate)
        else:
            ff_out = self.ff_act(ff_out)
        # Check for NaN after activation
        if torch.isnan(ff_out).any():
            raise ValueError(f"NaN after BasicTransformerBlock activation!")
        ff_out = self.ff_dropout(ff_out)
        ff_out_before_ff2 = ff_out  # Save for debugging
        # Clamp inf values before ff_2 to prevent overflow
        if torch.isinf(ff_out_before_ff2).any():
            logger.warning(f"Inf values detected before ff_2, clamping to safe range. Stats: min={ff_out_before_ff2.min().item():.6f}, max={ff_out_before_ff2.max().item():.6f}")
            ff_out_before_ff2 = torch.clamp(ff_out_before_ff2, min=-1e4, max=1e4)
        ff_out, _ = self.ff_2(ff_out_before_ff2)
        # Check for NaN/inf after ff_2
        if torch.isnan(ff_out).any():
            logger.error(f"NaN after BasicTransformerBlock ff_2! Input stats: min={ff_out_before_ff2.min().item():.6f}, max={ff_out_before_ff2.max().item():.6f}, "
                        f"ff_2 weight max: {self.ff_2.weight.abs().max().item() if hasattr(self.ff_2, 'weight') else 'N/A'}")
            # Try to recover by clamping
            ff_out = torch.clamp(ff_out, min=-1e6, max=1e6)
            ff_out = torch.nan_to_num(ff_out, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isinf(ff_out).any():
            ff_out = torch.clamp(ff_out, min=-1e6, max=1e6)
        x = x + ff_out
        # Check for NaN after adding ff_out
        if torch.isnan(x).any():
            raise ValueError(f"NaN after adding ff_out!")
        
        return x


class SpatialTransformer(nn.Module):
    """Spatial transformer block for UNet."""
    
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: int | None = None,
        use_linear: bool = False,
        use_checkpoint: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.n_heads = n_heads
        self.d_head = d_head
        
        # Project input to inner dimension
        if use_linear:
            self.proj_in = ReplicatedLinear(in_channels, inner_dim, params_dtype=dtype)
            self.proj_in_linear = True
        else:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device
            )
            self.proj_in_linear = False
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=inner_dim,
                n_heads=n_heads,
                d_head=d_head,
                dropout=dropout,
                context_dim=context_dim,
                checkpoint=use_checkpoint,
                dtype=dtype,
                device=device,
            )
            for _ in range(depth)
        ])
        
        # Project output back to input dimension
        if use_linear:
            self.proj_out = ReplicatedLinear(inner_dim, in_channels, params_dtype=dtype)
            self.proj_out_linear = True
        else:
            self.proj_out = nn.Conv2d(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device
            )
            self.proj_out_linear = False
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        
        # Check for NaN in input
        if torch.isnan(x).any():
            raise ValueError(f"NaN in SpatialTransformer input x! x shape: {x.shape}")
        
        # Project input
        if not self.proj_in_linear:
            x_in = self.proj_in(x)
            # Check for NaN after proj_in (conv)
            if torch.isnan(x_in).any():
                raise ValueError(f"NaN after SpatialTransformer proj_in (conv)!")
        else:
            x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            # Check for NaN before proj_in
            if torch.isnan(x_flat).any():
                raise ValueError(f"NaN in x_flat before proj_in!")
            x_in, _ = self.proj_in(x_flat)
            # Check for NaN after proj_in (linear)
            if torch.isnan(x_in).any():
                raise ValueError(f"NaN after SpatialTransformer proj_in (linear)! x_flat shape: {x_flat.shape}, x_in shape: {x_in.shape}")
            x_in = x_in.transpose(1, 2).view(b, -1, h, w)
            # Check for NaN after reshape
            if torch.isnan(x_in).any():
                raise ValueError(f"NaN after proj_in reshape! Expected shape: [B, {self.n_heads * self.d_head}, H, W], got: {x_in.shape}")
        
        # Flatten spatial dimensions
        x_in = x_in.flatten(2).transpose(1, 2)  # [B, H*W, inner_dim]
        
        # Check for NaN before transformer blocks
        if torch.isnan(x_in).any():
            raise ValueError(f"NaN in x_in before transformer blocks!")
        
        # Apply transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            x_in = block(x_in, context)
            # Handle tuple returns
            if isinstance(x_in, tuple):
                x_in = x_in[0]
            # Check for NaN after each transformer block
            if torch.isnan(x_in).any():
                raise ValueError(f"NaN after transformer block {idx}!")
        
        # Reshape back
        x_in = x_in.transpose(1, 2).view(b, -1, h, w)  # [B, inner_dim, H, W]
        
        # Check for NaN after reshape back
        if torch.isnan(x_in).any():
            raise ValueError(f"NaN after reshape back!")
        
        # Project output
        if not self.proj_out_linear:
            x_out = self.proj_out(x_in)
            # Check for NaN after proj_out (conv)
            if torch.isnan(x_out).any():
                raise ValueError(f"NaN after SpatialTransformer proj_out (conv)!")
        else:
            x_flat = x_in.flatten(2).transpose(1, 2)  # [B, H*W, inner_dim]
            x_out, _ = self.proj_out(x_flat)
            # Check for NaN after proj_out (linear)
            if torch.isnan(x_out).any():
                raise ValueError(f"NaN after SpatialTransformer proj_out (linear)!")
            x_out = x_out.transpose(1, 2).view(b, c, h, w)
        
        result = x + x_out
        
        # Check for NaN in final output
        if torch.isnan(result).any():
            raise ValueError(f"NaN in SpatialTransformer final output! x stats: min={x.min().item():.6f}, max={x.max().item():.6f}, x_out stats: min={x_out.min().item():.6f}, max={x_out.max().item():.6f}")
        
        return result

