# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for ktransformers hybrid MoE execution."""

from pydantic import Field, model_validator

from aphrodite.config.utils import config, get_hash_factors, hash_factors


@config
class KTransformersMoEConfig:
    """Configuration for ktransformers CPU+GPU hybrid MoE execution."""

    kt_weight_path: str | None = None
    """Path to CPU expert weights.
    This may point at either a ktransformers-prepacked expert checkpoint or a
    regular HF safetensors checkpoint. If unset, hybrid MoE execution is
    disabled unless the launcher fills it from a local model path.
    """

    kt_method: str = "AUTO"
    """CPU expert weight format. Use AUTO to infer it from safetensors."""

    kt_cpuinfer: int | None = Field(default=None, ge=1)
    """Number of CPUInfer threads used by ktransformers."""

    kt_threadpool_count: int = Field(default=2, ge=1)
    """Number of ktransformers CPU thread pools, typically one per NUMA node."""

    kt_num_gpu_experts: int | None = Field(default=None, ge=1)
    """Number of routed experts kept on GPU. Remaining experts run on CPU."""

    kt_max_deferred_experts_per_token: int | None = Field(default=None, ge=0)
    """Maximum number of experts per token deferred to CPU."""

    @property
    def is_enabled(self) -> bool:
        return self.kt_weight_path is not None or self.kt_num_gpu_experts is not None

    @model_validator(mode="after")
    def validate_ktransformers_moe_config(self) -> "KTransformersMoEConfig":
        self.kt_method = self.kt_method.upper()
        supported_methods = {
            "AUTO",
            "AMXINT4",
            "AMXINT8",
            "RAWINT4",
            "FP8",
            "FP8_AVX2",
            "FP8_PERCHANNEL",
            "BF16",
            "BF16_AVX2",
            "BF16_AVX512",
            "GPTQ_INT4",
        }
        if self.kt_method not in supported_methods:
            raise ValueError(
                "Unsupported kt_method "
                f"{self.kt_method!r}; expected one of {sorted(supported_methods)}."
            )
        if self.is_enabled and self.kt_num_gpu_experts is None:
            raise ValueError("kt_num_gpu_experts must be set when ktransformers hybrid MoE is enabled.")
        return self

    def compute_hash(self) -> str:
        factors = get_hash_factors(self, ignored_factors=set())
        return hash_factors(factors)
