# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from ComfyUI: comfy/model_base.py

# SPDX-License-Identifier: Apache-2.0

import torch

from aphrodite.diffusion.runtime.models.unet.utils import timestep_embedding


class ADMConditioning:
    """
    Additional Dimension (ADM) conditioning for SDXL.

    Encodes resolution, crop, and aesthetic score information into embeddings
    that are concatenated with CLIP pooled output for UNet conditioning.

    Based on ComfyUI's SDXL encode_adm implementation.
    """

    def __init__(self, embed_dim: int = 256):
        """
        Initialize ADM conditioning.

        Args:
            embed_dim: Dimension for each value embedding (default 256 for SDXL)
        """
        self.embed_dim = embed_dim

    def encode_sdxl_base(
        self,
        clip_pooled: torch.Tensor,
        height: int = 768,
        width: int = 768,
        crop_h: int = 0,
        crop_w: int = 0,
        target_height: int | None = None,
        target_width: int | None = None,
    ) -> torch.Tensor:
        """
        Encode ADM conditioning for SDXL base model.

        Args:
            clip_pooled: CLIP-G pooled output [B, 1280]
            height: Image height
            width: Image width
            crop_h: Crop top offset
            crop_w: Crop left offset
            target_height: Target height (defaults to height)
            target_width: Target width (defaults to width)

        Returns:
            ADM conditioning tensor [B, 2816]
            - CLIP pooled: 1280 dims
            - Height embedding: 256 dims
            - Width embedding: 256 dims
            - Crop H embedding: 256 dims
            - Crop W embedding: 256 dims
            - Target height embedding: 256 dims
            - Target width embedding: 256 dims
            Total: 1280 + 6 * 256 = 2816 dims
        """
        if target_height is None:
            target_height = height
        if target_width is None:
            target_width = width

        batch_size = clip_pooled.shape[0]
        device = clip_pooled.device

        # Embed each value using timestep embedding
        # ComfyUI uses Timestep(256) embedder
        embeddings = []
        embeddings.append(timestep_embedding(torch.tensor([height], device=device), self.embed_dim))
        embeddings.append(timestep_embedding(torch.tensor([width], device=device), self.embed_dim))
        embeddings.append(timestep_embedding(torch.tensor([crop_h], device=device), self.embed_dim))
        embeddings.append(timestep_embedding(torch.tensor([crop_w], device=device), self.embed_dim))
        embeddings.append(timestep_embedding(torch.tensor([target_height], device=device), self.embed_dim))
        embeddings.append(timestep_embedding(torch.tensor([target_width], device=device), self.embed_dim))

        # Concatenate all embeddings and flatten
        # ComfyUI: torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        flat = torch.flatten(torch.cat(embeddings, dim=1)).unsqueeze(dim=0).repeat(batch_size, 1)

        # Concatenate with CLIP pooled output
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)

    def encode_sdxl_refiner(
        self,
        clip_pooled: torch.Tensor,
        height: int = 768,
        width: int = 768,
        crop_h: int = 0,
        crop_w: int = 0,
        aesthetic_score: float = 6.0,
        is_negative: bool = False,
    ) -> torch.Tensor:
        """
        Encode ADM conditioning for SDXL refiner model.

        Args:
            clip_pooled: CLIP-G pooled output [B, 1280]
            height: Image height
            width: Image width
            crop_h: Crop top offset
            crop_w: Crop left offset
            aesthetic_score: Aesthetic score (default 6.0 for positive, 2.5 for negative)
            is_negative: Whether this is for negative prompt (uses different default score)

        Returns:
            ADM conditioning tensor [B, 2560]
            - CLIP pooled: 1280 dims
            - Height embedding: 256 dims
            - Width embedding: 256 dims
            - Crop H embedding: 256 dims
            - Crop W embedding: 256 dims
            - Aesthetic score embedding: 256 dims
            Total: 1280 + 5 * 256 = 2560 dims
        """
        if is_negative and aesthetic_score == 6.0:
            # Default for negative prompts
            aesthetic_score = 2.5

        batch_size = clip_pooled.shape[0]
        device = clip_pooled.device

        # Embed each value
        embeddings = []
        embeddings.append(timestep_embedding(torch.tensor([height], device=device), self.embed_dim))
        embeddings.append(timestep_embedding(torch.tensor([width], device=device), self.embed_dim))
        embeddings.append(timestep_embedding(torch.tensor([crop_h], device=device), self.embed_dim))
        embeddings.append(timestep_embedding(torch.tensor([crop_w], device=device), self.embed_dim))
        embeddings.append(timestep_embedding(torch.tensor([aesthetic_score], device=device), self.embed_dim))

        # Concatenate and flatten
        flat = torch.flatten(torch.cat(embeddings, dim=1)).unsqueeze(dim=0).repeat(batch_size, 1)

        # Concatenate with CLIP pooled output
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


def encode_sdxl_adm(
    clip_pooled: torch.Tensor,
    height: int = 768,
    width: int = 768,
    crop_h: int = 0,
    crop_w: int = 0,
    target_height: int | None = None,
    target_width: int | None = None,
    aesthetic_score: float | None = None,
    is_refiner: bool = False,
    is_negative: bool = False,
) -> torch.Tensor:
    """
    Convenience function to encode SDXL ADM conditioning.

    Args:
        clip_pooled: CLIP-G pooled output [B, 1280]
        height: Image height
        width: Image width
        crop_h: Crop top offset
        crop_w: Crop left offset
        target_height: Target height (for base model, defaults to height)
        target_width: Target width (for base model, defaults to width)
        aesthetic_score: Aesthetic score (for refiner model)
        is_refiner: Whether this is for refiner model (uses aesthetic_score)
        is_negative: Whether this is for negative prompt (affects aesthetic_score default)

    Returns:
        ADM conditioning tensor [B, 2816] for base or [B, 2560] for refiner
    """
    adm = ADMConditioning(embed_dim=256)

    if is_refiner:
        return adm.encode_sdxl_refiner(
            clip_pooled=clip_pooled,
            height=height,
            width=width,
            crop_h=crop_h,
            crop_w=crop_w,
            aesthetic_score=aesthetic_score if aesthetic_score is not None else 6.0,
            is_negative=is_negative,
        )
    else:
        return adm.encode_sdxl_base(
            clip_pooled=clip_pooled,
            height=height,
            width=width,
            crop_h=crop_h,
            crop_w=crop_w,
            target_height=target_height,
            target_width=target_width,
        )
