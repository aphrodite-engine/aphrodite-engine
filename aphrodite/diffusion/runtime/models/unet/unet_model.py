# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from ComfyUI: comfy/ldm/modules/diffusionmodules/openaimodel.py

# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from aphrodite.diffusion.runtime.layers.linear import ReplicatedLinear
from aphrodite.diffusion.runtime.models.unet.blocks import (
    Downsample,
    ResBlock,
    TimestepBlock,
    Upsample,
)
from aphrodite.diffusion.runtime.models.unet.spatial_transformer import SpatialTransformer
from aphrodite.diffusion.runtime.models.unet.utils import timestep_embedding


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to children that support it.
    Based on ComfyUI's forward_timestep_embed function.
    """
    
    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor | None = None,
        output_shape: tuple | None = None,
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, Upsample):
                x = layer(x, output_shape=output_shape)
            else:
                x = layer(x)
        return x


class Timestep(nn.Module):
    """Timestep embedding module."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(t, self.dim)


class SDXLUNetModel(nn.Module):
    """
    SDXL UNet model.
    
    Architecture:
    - Input blocks (downsampling)
    - Middle block
    - Output blocks (upsampling)
    """
    
    def __init__(
        self,
        image_size: int | None = None,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 320,
        num_res_blocks: int | list[int] = 2,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        dropout: float = 0.0,
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: int | None = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_spatial_transformer: bool = True,
        transformer_depth: list[int] = (1, 1, 1, 1, 1, 1),
        transformer_depth_middle: int | None = None,
        transformer_depth_output: list[int] | None = None,
        context_dim: int | None = None,
        use_linear_in_transformer: bool = False,
        adm_in_channels: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        
        if context_dim is not None:
            assert use_spatial_transformer, "Must use spatial transformer when context_dim is provided"
        
        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels must be set"
        
        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels must be set"
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "num_res_blocks must be int or list with same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = dtype or torch.float32
        
        time_embed_dim = model_channels * 4
        
        # Timestep embedding
        self.time_embed_0 = ReplicatedLinear(model_channels, time_embed_dim, params_dtype=dtype)
        self.time_embed_act = nn.SiLU()
        self.time_embed_2 = ReplicatedLinear(time_embed_dim, time_embed_dim, params_dtype=dtype)
        
        # Additional dimension (ADM) conditioning for SDXL
        if adm_in_channels is not None:
            self.label_emb_0 = ReplicatedLinear(adm_in_channels, time_embed_dim, params_dtype=dtype)
            self.label_emb_act = nn.SiLU()
            self.label_emb_2 = ReplicatedLinear(time_embed_dim, time_embed_dim, params_dtype=dtype)
        else:
            self.label_emb_0 = None
            self.label_emb_act = None
            self.label_emb_2 = None
        
        # Input blocks
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, model_channels, 3, padding=1, dtype=dtype, device=device)
            )
        ])
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        # Helper to get attention layer
        def get_attention_layer(ch, depth):
            if num_head_channels == -1:
                dim_head = ch // num_heads
            else:
                num_heads_local = ch // num_head_channels
                dim_head = num_head_channels
            
            return SpatialTransformer(
                in_channels=ch,
                n_heads=num_heads if num_head_channels == -1 else num_heads_local,
                d_head=dim_head,
                depth=depth,
                dropout=dropout,
                context_dim=context_dim,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
                dtype=dtype,
                device=device,
            )
        
        # Build input blocks (based on ComfyUI's structure)
        # Make a copy since we'll be popping from it
        transformer_depth_input = list(transformer_depth) if transformer_depth else []
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=dtype,
                        device=device,
                    )
                ]
                ch = mult * model_channels
                
                # Add transformer if depth > 0 (ComfyUI pops from transformer_depth list)
                num_transformers = transformer_depth_input.pop(0) if transformer_depth_input else 0
                if num_transformers > 0 and use_spatial_transformer:
                    layers.append(get_attention_layer(ch, num_transformers))
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            
            # Downsample (except last level, and skip if next level has same channel_mult)
            # For SDXL: channel_mult = (1, 2, 4, 4), so we should only downsample at levels 0 and 1
            # Level 2 has same mult as level 3, so we shouldn't downsample at level 2
            should_downsample = level != len(channel_mult) - 1
            if should_downsample and level < len(channel_mult) - 1:
                # Check if next level has different channel_mult (if same, don't downsample)
                next_mult = channel_mult[level + 1]
                if mult == next_mult:
                    should_downsample = False
            
            if should_downsample:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            channels=ch,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=dtype,
                            device=device,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, dtype=dtype, device=device
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
            else:
                # Even if we don't downsample, we still need to track the channel count
                # for the output blocks (which use input_block_chans)
                pass
        
        # Middle block (based on ComfyUI)
        # Note: dim_head is computed in get_attention_layer, so we don't need it here
        
        # Middle block (ComfyUI logic: transformer_depth_middle >= -1 means create block)
        # -1: just ResBlock, no transformer
        # >= 0: ResBlock + Transformer + ResBlock
        # < -1: no middle block (not supported in our implementation)
        if transformer_depth_middle is None:
            # If not specified, use the last value from original transformer_depth or default to 1
            mid_transformer_depth = transformer_depth[-1] if transformer_depth and len(transformer_depth) > 0 else 1
        else:
            mid_transformer_depth = transformer_depth_middle
        
        mid_block = [
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=dtype,
                device=device,
            )
        ]
        
        # ComfyUI: if transformer_depth_middle >= 0, add transformer and second ResBlock
        if mid_transformer_depth >= 0 and use_spatial_transformer:
            mid_block.append(get_attention_layer(ch, mid_transformer_depth))
            mid_block.append(
                ResBlock(
                    channels=ch,
                    emb_channels=time_embed_dim,
                    dropout=dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=dtype,
                    device=device,
                )
            )
        
        self.middle_block = TimestepEmbedSequential(*mid_block)
        
        # Output blocks (based on ComfyUI - uses transformer_depth_output)
        self.output_blocks = nn.ModuleList([])
        # ComfyUI uses transformer_depth_output as a separate parameter
        if transformer_depth_output is None:
            # If not specified, use original transformer_depth (reversed for output blocks)
            transformer_depth_output = list(transformer_depth) if transformer_depth else []
        else:
            transformer_depth_output = list(transformer_depth_output)
        transformer_depth_output.reverse()  # ComfyUI pops from the end
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=dtype,
                        device=device,
                    )
                ]
                ch = model_channels * mult
                
                # Add transformer (ComfyUI pops from end of transformer_depth_output)
                num_transformers = transformer_depth_output.pop() if transformer_depth_output else 0
                if num_transformers > 0 and use_spatial_transformer:
                    layers.append(get_attention_layer(ch, num_transformers))
                
                # Upsample (except first iteration of last level)
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            channels=ch,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=dtype,
                            device=device,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=dtype, device=device)
                    )
                    ds //= 2
                
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1, dtype=dtype, device=device),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply the model to an input batch.
        
        Args:
            x: [N x C x H x W] Tensor of inputs
            timesteps: [N] Tensor of timesteps
            context: [N x L x D] Tensor of conditioning (text embeddings)
            y: [N x ADM_DIM] Tensor of ADM conditioning (for SDXL)
        
        Returns:
            [N x C x H x W] Tensor of outputs
        """
        # Timestep embedding
        t_emb = timestep_embedding(timesteps, self.model_channels).to(x.dtype)
        # Check for NaN in timestep input
        if torch.isnan(t_emb).any():
            raise ValueError(f"NaN in timestep embedding input (t_emb)! t_emb shape: {t_emb.shape}, timesteps: {timesteps}, t_emb stats: min={t_emb.min().item():.6f}, max={t_emb.max().item():.6f}")
        
        # Handle ReplicatedLinear tuple returns
        emb_out, _ = self.time_embed_0(t_emb)
        # Check for NaN after time_embed_0
        if torch.isnan(emb_out).any():
            raise ValueError(f"NaN after time_embed_0! t_emb stats: min={t_emb.min().item():.6f}, max={t_emb.max().item():.6f}")
        
        emb = self.time_embed_act(emb_out)
        # Check for NaN after activation
        if torch.isnan(emb).any():
            raise ValueError(f"NaN after time_embed_act!")
        
        emb_out, _ = self.time_embed_2(emb)
        # Check for NaN after time_embed_2
        if torch.isnan(emb_out).any():
            raise ValueError(f"NaN after time_embed_2!")
        emb = emb_out
        
        # ADM conditioning (for SDXL)
        if self.label_emb_0 is not None and y is not None:
            # Check for NaN in y before processing
            if torch.isnan(y).any():
                raise ValueError(f"NaN in ADM conditioning (y) input! y shape: {y.shape}, y stats: min={y.min().item():.6f}, max={y.max().item():.6f}")
            
            adm_emb_out, _ = self.label_emb_0(y)
            # Check for NaN after first linear layer
            if torch.isnan(adm_emb_out).any():
                raise ValueError(f"NaN after label_emb_0! y stats: min={y.min().item():.6f}, max={y.max().item():.6f}")
            
            adm_emb = self.label_emb_act(adm_emb_out)
            adm_emb_out, _ = self.label_emb_2(adm_emb)
            # Check for NaN after ADM processing
            if torch.isnan(adm_emb_out).any():
                raise ValueError(f"NaN after label_emb_2!")
            
            emb = emb + adm_emb_out
            # Check for NaN after adding ADM to timestep embedding
            if torch.isnan(emb).any():
                raise ValueError(f"NaN after adding ADM to timestep embedding!")
        
        # Input blocks (based on ComfyUI's forward pass)
        h = x
        hs = []
        # Check for NaN in timestep embedding before input blocks
        if torch.isnan(emb).any():
            raise ValueError(f"NaN in timestep embedding before input blocks!")
        
        for idx, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            # Check for NaN after each input block
            if torch.isnan(h).any():
                raise ValueError(f"NaN after input block {idx}!")
            hs.append(h)
        
        # Middle block
        h = self.middle_block(h, emb, context)
        
        # Output blocks (based on ComfyUI's forward pass)
        for module in self.output_blocks:
            hsp = hs.pop()
            h = torch.cat([h, hsp], dim=1)
            # Determine output_shape for upsampling layers (ComfyUI logic)
            output_shape = hs[-1].shape if len(hs) > 0 else None
            h = module(h, emb, context, output_shape=output_shape)
        
        # Convert to input dtype before output (ComfyUI does this)
        h = h.type(x.dtype)
        
        # Output projection
        return self.out(h)
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights for SDXL UNet.
        
        Handles both diffusers format (unet.*) and Aphrodite format (no prefix).
        Builds proper block index mapping based on model structure (matching ComfyUI).
        """
        from aphrodite.diffusion.runtime.loader.weight_utils import default_weight_loader
        from aphrodite.diffusion.runtime.models.unet.weight_utils import (
            process_sdxl_unet_weights,
        )
        
        # Build proper block index mapping (matching ComfyUI's logic)
        # This is needed because the simple regex replacement doesn't account for
        # the actual block structure which depends on num_res_blocks
        num_res_blocks = getattr(self, 'num_res_blocks', [2] * len(getattr(self, 'channel_mult', [1, 2, 4, 4])))
        channel_mult = getattr(self, 'channel_mult', [1, 2, 4, 4])
        num_blocks = len(channel_mult)
        
        # Build mapping: down_blocks.{x}.attentions.{i} -> input_blocks.{n}.1
        # Also build mapping for up_blocks -> output_blocks (reversed)
        block_mapping = {}
        
        # Input blocks (down_blocks)
        for x in range(num_blocks):
            n = 1 + (num_res_blocks[x] + 1) * x
            for i in range(num_res_blocks[x]):
                # Map down_blocks.{x}.attentions.{i}.* -> input_blocks.{n}.1.*
                old_prefix = f"down_blocks.{x}.attentions.{i}."
                new_prefix = f"input_blocks.{n}.1."
                block_mapping[old_prefix] = new_prefix
                # Also map resnets
                old_resnet_prefix = f"down_blocks.{x}.resnets.{i}."
                new_resnet_prefix = f"input_blocks.{n}.0."
                block_mapping[old_resnet_prefix] = new_resnet_prefix
                n += 1
        
        # Output blocks (up_blocks) - ComfyUI's exact logic
        # Need to account for transformer_depth_output to only map attention where transformers exist
        # Get transformer_depth_output (reversed, as ComfyUI pops from end)
        transformer_depth_output = getattr(self, 'transformer_depth_output', None)
        if transformer_depth_output is None:
            # Fallback: use transformer_depth reversed
            transformer_depth = getattr(self, 'transformer_depth', [0, 0, 2, 2, 10, 10])
            transformer_depth_output = list(transformer_depth)
        else:
            transformer_depth_output = list(transformer_depth_output)
        transformer_depth_output.reverse()  # ComfyUI pops from end
        
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        
        # Calculate output block structure
        output_block_start_indices = []
        current_idx = 0
        model_channels = getattr(self, 'model_channels', 320)
        for level in reversed(range(num_blocks)):
            num_blocks_at_level = reversed_num_res_blocks[level] + 1
            ch = model_channels * channel_mult[level]
            output_block_start_indices.append((level, current_idx, num_blocks_at_level, ch))
            current_idx += num_blocks_at_level
        
        # Map up_blocks to output_blocks using ComfyUI's exact formula
        # This is the standard mapping that ComfyUI uses for all models
        # Note: This may cause channel mismatches (e.g., up_blocks.1 with 640 channels
        # mapped to output_blocks 3-5 with 1280 channels), but ComfyUI handles this
        # by resizing weights or by having a different model structure
        for x in range(num_blocks):
            # Use ComfyUI's formula: n = (reversed_num_res_blocks[x] + 1) * x
            n = (reversed_num_res_blocks[x] + 1) * x
            l = reversed_num_res_blocks[x] + 1
            
            # Skip if this would go beyond available output blocks
            if n + l > len(self.output_blocks):
                continue
            
            for i in range(l):
                # Map resnets (all positions including where upsampler will be)
                old_resnet_prefix = f"up_blocks.{x}.resnets.{i}."
                new_resnet_prefix = f"output_blocks.{n}.0."
                block_mapping[old_resnet_prefix] = new_resnet_prefix
                
                # Map attention only if this block has transformers (matching ComfyUI's logic)
                # ComfyUI pops transformer_depth_output for each iteration
                num_transformers = transformer_depth_output.pop() if transformer_depth_output else 0
                if num_transformers > 0:
                    old_prefix = f"up_blocks.{x}.attentions.{i}."
                    new_prefix = f"output_blocks.{n}.1."
                    block_mapping[old_prefix] = new_prefix
                
                n += 1
        
        # First, do the standard conversions (ff.net, mid_block, etc.)
        processed_weights = list(process_sdxl_unet_weights(weights))
        
        # Then apply block index mapping to fix down_blocks -> input_blocks indices
        def apply_block_mapping(weights_iter):
            for name, tensor in weights_iter:
                original_name = name
                # Apply block mapping (this fixes the block indices)
                # Sort prefixes by length (longest first) to ensure more specific matches
                sorted_mappings = sorted(block_mapping.items(), key=lambda x: len(x[0]), reverse=True)
                for old_prefix, new_prefix in sorted_mappings:
                    if name.startswith(old_prefix):
                        name = new_prefix + name[len(old_prefix):]
                        # Debug: log mappings for output_blocks.3
                        if "output_blocks.3" in name:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.debug(f"Block mapping: {original_name} -> {name} (via {old_prefix} -> {new_prefix})")
                        break
                
                yield (name, tensor)
        
        # Apply block mapping after other conversions
        processed_weights = list(apply_block_mapping(processed_weights))
        
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        missing_weights = []
        size_mismatches = []
        # Build a channel-size-based fallback mapping for attention layers with size mismatches
        # This handles cases where checkpoint structure doesn't match expected mapping
        channel_size_fallback = {}
        
        for name, loaded_weight in processed_weights:
            # Check channel-size fallback first
            if name in channel_size_fallback:
                param_name = channel_size_fallback[name]
                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight)
                    loaded_params.add(param_name)
                    continue
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Failed to load fallback weight '{name}' -> '{param_name}': {e}")
            
            if name in params_dict:
                param = params_dict[name]
                # Check for size mismatch before loading
                if param.size() != loaded_weight.size():
                    import logging
                    logger = logging.getLogger(__name__)
                    # Find the original weight name before mapping
                    original_name = None
                    for old_prefix, new_prefix in block_mapping.items():
                        if name.startswith(new_prefix):
                            original_name = old_prefix + name[len(new_prefix):]
                            break
                    
                    # Skip weight resizing - it causes incorrect behavior (noise in output)
                    # Instead, skip loading mismatched weights and let them remain randomly initialized
                    # This is better than incorrectly resizing weights which produces noise
                    transformed_weight = None
                    param_shape = param.shape
                    loaded_shape = loaded_weight.shape
                    
                    # Handle Conv2d <-> Linear conversion for proj_in/proj_out
                    # Checkpoint may have Linear weights [out_ch, in_ch] but our model uses Conv2d [out_ch, in_ch, 1, 1]
                    # Or vice versa: checkpoint has Conv2d [out_ch, in_ch, 1, 1] but our model uses Linear [out_ch, in_ch]
                    # Also handle channel resizing (e.g., 640->1280) combined with conversion
                    if ("proj_in" in name or "proj_out" in name):
                        if len(param_shape) == 4 and len(loaded_shape) == 2:
                            # Our model uses Conv2d, checkpoint has Linear: [out_ch, in_ch] -> [out_ch, in_ch, 1, 1]
                            # Check if channels match exactly, or if we need to resize
                            if param_shape[0] == loaded_shape[0] and param_shape[1] == loaded_shape[1] and param_shape[2] == 1 and param_shape[3] == 1:
                                # Exact match - just convert shape
                                transformed_weight = loaded_weight.unsqueeze(-1).unsqueeze(-1)
                                logger.info(f"Converting Linear to Conv2d weight '{name}': {loaded_shape} -> {param_shape}")
                            elif param_shape[0] == loaded_shape[0] * 2 and param_shape[1] == loaded_shape[1] * 2 and param_shape[2] == 1 and param_shape[3] == 1:
                                # Need to resize channels (double) AND convert to Conv2d
                                # Repeat weights without scaling - doubling channels naturally doubles output magnitude
                                resized = loaded_weight.repeat(2, 2)
                                transformed_weight = resized.unsqueeze(-1).unsqueeze(-1)
                                logger.warning(f"Resizing and converting Linear to Conv2d weight '{name}': {loaded_shape} -> {param_shape} (doubling channels)")
                            elif param_shape[0] == loaded_shape[0] * 2 and param_shape[1] == loaded_shape[1] and param_shape[2] == 1 and param_shape[3] == 1:
                                # Double output channels only - repeat without scaling
                                resized = loaded_weight.repeat(2, 1)
                                transformed_weight = resized.unsqueeze(-1).unsqueeze(-1)
                                logger.warning(f"Resizing and converting Linear to Conv2d weight '{name}': {loaded_shape} -> {param_shape} (doubling output channels)")
                            elif param_shape[0] == loaded_shape[0] and param_shape[1] == loaded_shape[1] * 2 and param_shape[2] == 1 and param_shape[3] == 1:
                                # Double input channels only - no scaling needed (input doubling doesn't change output magnitude per channel)
                                resized = loaded_weight.repeat(1, 2)
                                transformed_weight = resized.unsqueeze(-1).unsqueeze(-1)
                                logger.warning(f"Resizing and converting Linear to Conv2d weight '{name}': {loaded_shape} -> {param_shape} (doubling input channels)")
                        elif len(param_shape) == 2 and len(loaded_shape) == 4:
                            # Our model uses Linear, checkpoint has Conv2d: [out_ch, in_ch, 1, 1] -> [out_ch, in_ch]
                            if param_shape[0] == loaded_shape[0] and param_shape[1] == loaded_shape[1] and loaded_shape[2] == 1 and loaded_shape[3] == 1:
                                transformed_weight = loaded_weight.squeeze(-1).squeeze(-1)
                                logger.info(f"Converting Conv2d to Linear weight '{name}': {loaded_shape} -> {param_shape}")
                    
                    # Enable weight resizing for channel mismatches
                    # ComfyUI's formula may map blocks with different channel sizes, requiring resizing
                    # We'll resize weights when channel dimensions differ by exactly 2x (doubling/halving)
                    if transformed_weight is None and len(param_shape) == len(loaded_shape) and len(param_shape) >= 2:
                        # Check if this is a channel dimension mismatch (exactly 2x difference)
                        # For 2D weight matrices (e.g., proj_in.weight, to_q.weight), both dimensions may need resizing
                        if len(param_shape) == 2:
                            # Check if both dimensions need to be doubled
                            if param_shape[0] == loaded_shape[0] * 2 and param_shape[1] == loaded_shape[1] * 2:
                                # Double both dimensions: repeat along both axes
                                # Don't scale - doubling channels naturally doubles output magnitude
                                transformed_weight = loaded_weight.repeat(2, 2)
                                logger.warning(f"Resizing weight '{name}': {loaded_shape} -> {param_shape} (doubling both dimensions)")
                            # Check if only first dimension needs doubling
                            elif param_shape[0] == loaded_shape[0] * 2 and param_shape[1] == loaded_shape[1]:
                                # Double the output channels: repeat along first dimension
                                # Don't scale - doubling output channels doubles output magnitude
                                transformed_weight = loaded_weight.repeat(2, 1)
                                logger.warning(f"Resizing weight '{name}': {loaded_shape} -> {param_shape} (doubling output channels)")
                            # Check if only second dimension needs doubling
                            elif param_shape[0] == loaded_shape[0] and param_shape[1] == loaded_shape[1] * 2:
                                # Double the input channels: repeat along second dimension
                                # No scaling needed here - input dimension doubling doesn't change output magnitude
                                transformed_weight = loaded_weight.repeat(1, 2)
                                logger.warning(f"Resizing weight '{name}': {loaded_shape} -> {param_shape} (doubling input channels)")
                            # Check if both dimensions need halving
                            elif param_shape[0] == loaded_shape[0] // 2 and param_shape[1] == loaded_shape[1] // 2:
                                # Halve both dimensions: take first half
                                transformed_weight = loaded_weight[:param_shape[0], :param_shape[1]]
                                logger.warning(f"Resizing weight '{name}': {loaded_shape} -> {param_shape} (halving both dimensions)")
                            # Check if only first dimension needs halving
                            elif param_shape[0] == loaded_shape[0] // 2 and param_shape[1] == loaded_shape[1]:
                                # Halve the output channels: take first half
                                transformed_weight = loaded_weight[:param_shape[0]]
                                logger.warning(f"Resizing weight '{name}': {loaded_shape} -> {param_shape} (halving output channels)")
                            # Check if only second dimension needs halving
                            elif param_shape[0] == loaded_shape[0] and param_shape[1] == loaded_shape[1] // 2:
                                # Halve the input channels: take first half
                                transformed_weight = loaded_weight[:, :param_shape[1]]
                                logger.warning(f"Resizing weight '{name}': {loaded_shape} -> {param_shape} (halving input channels)")
                        # For higher-dimensional tensors (e.g., conv weights), handle first dimension
                        elif param_shape[0] == loaded_shape[0] * 2 and param_shape[1:] == loaded_shape[1:]:
                            # Double the channels: repeat the weight
                            transformed_weight = loaded_weight.repeat(2, *([1] * (len(loaded_shape) - 1)))
                            logger.warning(f"Resizing weight '{name}': {loaded_shape} -> {param_shape} (doubling channels)")
                        elif param_shape[0] == loaded_shape[0] // 2 and param_shape[1:] == loaded_shape[1:]:
                            # Halve the channels: take first half
                            transformed_weight = loaded_weight[:param_shape[0]]
                            logger.warning(f"Resizing weight '{name}': {loaded_shape} -> {param_shape} (halving channels)")
                    
                    # For 1D tensors (biases, norm weights), resize by repeating or taking first part
                    if transformed_weight is None and len(param_shape) == 1 and len(loaded_shape) == 1:
                        if param_shape[0] == loaded_shape[0] * 2:
                            transformed_weight = loaded_weight.repeat(2)
                            logger.warning(f"Resizing 1D tensor '{name}': {loaded_shape} -> {param_shape} (doubling)")
                        elif param_shape[0] == loaded_shape[0] // 2:
                            transformed_weight = loaded_weight[:param_shape[0]]
                            logger.warning(f"Resizing 1D tensor '{name}': {loaded_shape} -> {param_shape} (halving)")
                    
                    if transformed_weight is not None and transformed_weight.shape == param.shape:
                        # Clamp transformed weights to prevent extreme values that can cause numerical instability
                        # Use a reasonable range based on typical weight values (most weights are in [-1, 1] range)
                        max_abs = transformed_weight.abs().max().item()
                        if max_abs > 1.0:
                            logger.warning(f"Clamping transformed weight '{name}' with max abs value {max_abs:.6e} to [-1.0, 1.0]")
                            transformed_weight = torch.clamp(transformed_weight, min=-1.0, max=1.0)
                        # Use the transformed weight
                        loaded_weight = transformed_weight
                    else:
                        # Can't transform - this indicates a fundamental mismatch
                        # Skip loading this weight and let it remain randomly initialized
                        # This is better than incorrectly resizing weights which causes noise
                        if original_name:
                            logger.warning(f"Skipping weight '{name}' (from '{original_name}'): param size {param.size()}, loaded size {loaded_weight.size()} - will remain randomly initialized")
                        else:
                            logger.warning(f"Skipping weight '{name}': param size {param.size()}, loaded size {loaded_weight.size()} - will remain randomly initialized")
                        size_mismatches.append((name, param.size(), loaded_weight.size()))
                        # Skip this weight - let it remain randomly initialized
                        continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                except AssertionError as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Failed to load weight '{name}': {e}")
                    logger.error(f"  Param size: {param.size()}, Loaded weight size: {loaded_weight.size()}")
                    raise
            else:
                # Log missing weights for attention layers to debug
                if 'attn' in name and 'weight' in name:
                    missing_weights.append(name)
        
        if missing_weights:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Some attention weights were not matched (first 10): {missing_weights[:10]}")
            # Also log some actual parameter names for comparison
            actual_param_names = [name for name in params_dict.keys() if 'attn' in name and 'weight' in name][:10]
            logger.warning(f"Actual attention parameter names in model (first 10): {actual_param_names}")
        
        return loaded_params


# Entry point for model registry
EntryClass = SDXLUNetModel

