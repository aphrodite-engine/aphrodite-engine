# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V3.2 (``deepseek_v32``) model — hardware-isolated entry point.

DeepSeek V3.2 introduced the DeepSeek Sparse Attention (DSA) architecture:
MLA + a "lightning indexer" that selects the top-k tokens for a sparse MLA
attend. The same model code serves any DSA checkpoint, including GLM-5.2
(``glm_moe_dsa``), which reuses this architecture.

The optimized kernels under ``nvidia/`` target the Blackwell (SM100) family.
Every other platform — ROCm, XPU, pre-SM100 CUDA (e.g. H100), CPU — falls back
to the generic ``deepseek_v2`` implementation, which already handles the DSA
(index_topk) architecture and is ``torch.compile``-friendly there. This matches
main's behavior on those platforms (no hard failure).
"""

from importlib import import_module

from aphrodite.platforms import current_platform

_use_nvidia = current_platform.is_cuda() and current_platform.is_device_capability_family(100)
_model_module = import_module(
    ".nvidia.model" if _use_nvidia else "aphrodite.model_executor.models.deepseek_v2",
    __package__,
)
_mtp_module = import_module(
    ".nvidia.mtp" if _use_nvidia else "aphrodite.model_executor.models.deepseek_mtp",
    __package__,
)

DeepseekV32ForCausalLM = getattr(
    _model_module,
    "DeepseekV32ForCausalLM" if _use_nvidia else "DeepseekV3ForCausalLM",
)
DeepseekV32MTP = getattr(_mtp_module, "DeepseekV32MTP" if _use_nvidia else "DeepSeekMTP")
GlmMoeDsaForCausalLM = DeepseekV32ForCausalLM if _use_nvidia else _model_module.GlmMoeDsaForCausalLM

__all__ = [
    "DeepseekV32ForCausalLM",
    "DeepseekV32MTP",
    "GlmMoeDsaForCausalLM",
]
