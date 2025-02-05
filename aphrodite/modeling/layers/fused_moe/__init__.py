from aphrodite.modeling.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from aphrodite.triton_utils import HAS_TRITON

__all__ = [
    "FusedMoE",
    "FusedMoEMethodBase",
    "FusedMoeWeightScaleSupported",
]

if HAS_TRITON:
    from aphrodite.modeling.layers.fused_moe.fused_marlin_moe import (
        fused_marlin_moe, single_marlin_moe)
    from aphrodite.modeling.layers.fused_moe.fused_moe import (
        fused_experts, fused_moe, fused_topk, get_config_file_name,
        grouped_topk)

    __all__ += [
        "fused_marlin_moe",
        "single_marlin_moe",
        "fused_moe",
        "fused_topk",
        "fused_experts",
        "get_config_file_name",
        "grouped_topk",
    ]
