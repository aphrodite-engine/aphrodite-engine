# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from aphrodite.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from aphrodite.platforms import current_platform
from aphrodite.utils.deep_gemm import fp8_einsum, is_deep_gemm_supported


def _expand_block_scales(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        from aphrodite.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        scale = _upcast_e8m0_to_fp32(scale)
    else:
        scale = scale.float()
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = (rows + row_blocks - 1) // row_blocks
    col_block = (cols + col_blocks - 1) // col_blocks
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    return torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]


def _bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """Portable inverse-RoPE and dequantized WO_A fallback."""
    half_rope = rope_dim // 2
    rope = o[..., -rope_dim:].float()
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    cos = cos_sin_cache[positions, :half_rope].unsqueeze(1)
    sin = cos_sin_cache[positions, half_rope:].unsqueeze(1)
    inv_rope = torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1).flatten(-2)
    o_ref = torch.cat((o[..., :-rope_dim].float(), inv_rope), dim=-1)
    o_ref = o_ref.view(o.shape[0], n_groups, -1)

    weight = getattr(wo_a, "_dsv4_wo_a_bf16", None)
    if weight is None:
        hidden_dim = heads_per_group * o.shape[-1]
        weight = wo_a.weight.view(n_groups, o_lora_rank, hidden_dim).float()
        weight_scale = wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
        weight_scale = weight_scale.view(n_groups, -1, weight_scale.shape[-1])
        weight = (weight * _expand_block_scales(weight_scale, o_lora_rank, hidden_dim)).to(torch.bfloat16)
        wo_a._dsv4_wo_a_bf16 = weight
    return torch.einsum("tgd,grd->tgr", o_ref.to(torch.bfloat16), weight)


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    """
    if not is_deep_gemm_supported():
        z = _bf16_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
        )
        return wo_b(z.flatten(1))

    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    weight_scale = wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (wo_a.weight, weight_scale),
        z,
        recipe=einsum_recipe,
    )
    return wo_b(z.flatten(1))
