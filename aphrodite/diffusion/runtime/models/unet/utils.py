# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from ComfyUI: comfy/ldm/modules/diffusionmodules/util.py

# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.

    Returns:
        an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    Args:
        func: the function to evaluate.
        inputs: a tuple of inputs to pass into `func`.
        params: a tuple of parameters that `func` depends on.
        flag: if False, disable gradient checkpointing and just call `func`.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return torch.utils.checkpoint.checkpoint(func, *args)
    else:
        return func(*inputs)


class AlphaBlender(nn.Module):
    """Alpha blending for temporal/spatial mixing."""

    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        assert merge_strategy in self.strategies, f"merge_strategy needs to be in {self.strategies}"

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy in ["learned", "learned_with_images"]:
            self.register_parameter("mix_factor", nn.Parameter(torch.Tensor([alpha])))
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: torch.Tensor | None, device) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor.to(device)
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor.to(device))
        elif self.merge_strategy == "learned_with_images":
            if image_only_indicator is None:
                alpha = torch.sigmoid(self.mix_factor.to(device)).unsqueeze(-1)
            else:
                alpha = torch.where(
                    image_only_indicator.bool(),
                    torch.ones(1, 1, device=image_only_indicator.device),
                    torch.sigmoid(self.mix_factor.to(image_only_indicator.device)).unsqueeze(-1),
                )
        else:
            raise NotImplementedError()
        return alpha

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator, x_spatial.device)
        x = alpha.to(x_spatial.dtype) * x_spatial + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        return x
