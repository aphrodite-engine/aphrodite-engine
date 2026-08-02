# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code=call-arg

from typing import Annotated, Any, Literal

from pydantic import Field, GetPydanticSchema, ValidationInfo, field_validator
from pydantic_core import core_schema

from aphrodite.config.utils import config
from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt8StaticChannelSym,
    kMxfp4Dynamic,
    kMxfp6E2m3Dynamic,
    kMxfp6E2m3Static,
    kMxfp6E3m2Dynamic,
    kMxfp6E3m2Static,
    kMxfp8Dynamic,
    kNvfp4Static,
)

# User-facing names addressable from quantization_config.
QUANT_KEY_NAMES: dict[str, QuantKey] = {
    "fp8_per_tensor_static": kFp8StaticTensorSym,
    "fp8_per_tensor_dynamic": kFp8DynamicTensorSym,
    "fp8_per_token": kFp8DynamicTokenSym,
    "fp8_per_channel_static": kFp8StaticChannelSym,
    "fp8_per_block_static": kFp8Static128BlockSym,
    "fp8_per_block_dynamic": kFp8Dynamic128Sym,
    "mxfp8": kMxfp8Dynamic,
    "mxfp4": kMxfp4Dynamic,
    "mxfp6_e2m3": kMxfp6E2m3Static,
    "mxfp6_e2m3_dynamic": kMxfp6E2m3Dynamic,
    "mxfp6_e3m2": kMxfp6E3m2Static,
    "mxfp6_e3m2_dynamic": kMxfp6E3m2Dynamic,
    "int8_per_channel_static": kInt8StaticChannelSym,
}


def _coerce_quant_key(v: Any) -> QuantKey | None:
    if v is None or isinstance(v, QuantKey):
        return v
    if not isinstance(v, str):
        raise TypeError(f"expected str or QuantKey, got {type(v).__name__}")
    try:
        return QUANT_KEY_NAMES[v]
    except KeyError:
        raise ValueError(f"unknown quantization name {v!r}; expected one of {sorted(QUANT_KEY_NAMES)}") from None


# Stop pydantic from introspecting QuantKey: it transitively contains a
# NamedTuple with `ClassVar[GroupShape]` declarations that pydantic refuses.
QuantKeyField = Annotated[
    QuantKey | None,
    GetPydanticSchema(lambda _src, _handler: core_schema.no_info_plain_validator_function(_coerce_quant_key)),
]


@config
class QuantSpec:
    """Quantization spec for one layer kind (linear or MoE).

    `None` on either side means the method class falls back to its own default
    (typically inherited from the checkpoint, or unquantized for online).
    """

    weight: QuantKeyField = None
    """Weight quantization key, or a name from QUANT_KEY_NAMES."""

    activation: QuantKeyField = None
    """Activation quantization key, or a name from QUANT_KEY_NAMES."""


def _coerce_override_weight(v: Any) -> Any:
    if v in (None, "bf16"):
        return v
    return _coerce_quant_key(v)


OverrideWeightField = Annotated[
    QuantKey | Literal["bf16"] | None,
    GetPydanticSchema(lambda _src, _handler: core_schema.no_info_plain_validator_function(_coerce_override_weight)),
]


@config
class QuantOverride:
    """Ordered module-level override for online quantization."""

    pattern: str = ""
    """Exact module prefix or ``re:`` regular expression."""

    weight: OverrideWeightField = None
    """Replacement weight format; ``bf16`` leaves the module unquantized."""

    activation: QuantKeyField = None
    """Replacement activation format; omitted fields inherit the base spec."""


@config
class QuantizationConfigArgs:
    """User-facing quantization configuration.

    See `docs/features/quantization/online.md` for the schema and shorthand
    string forms accepted on `linear` and `moe`.
    """

    linear: QuantSpec | None = None
    """Spec applied to ``LinearBase`` layers."""

    moe: QuantSpec | None = None
    """Spec applied to ``FusedMoE`` layers."""

    ignore: list[str] = Field(default_factory=list)
    """Layers to skip quantization for."""

    overrides: list[QuantOverride] = Field(default_factory=list)
    """Ordered module precision overrides. Later matching rules win."""

    @field_validator("linear", "moe", mode="before")
    @classmethod
    def _coerce_spec(cls, v: Any, info: ValidationInfo) -> Any:
        if not isinstance(v, str):
            return v
        field_name = info.field_name
        assert field_name is not None
        if v in _ONLINE_SHORTHANDS:
            spec = getattr(_ONLINE_SHORTHANDS[v], field_name)
            if spec is None:
                raise ValueError(f"online shorthand {v!r} does not define a {field_name} spec")
            return spec
        return QuantSpec(weight=_coerce_quant_key(v))


# CLI shorthands accepted by `--quantization`. Each desugars to a full
# QuantizationConfigArgs; activation overrides go through quantization_config.
_ONLINE_SHORTHANDS: dict[str, QuantizationConfigArgs] = {
    "fp8_per_tensor": QuantizationConfigArgs(
        linear=QuantSpec(weight=kFp8StaticTensorSym),
        moe=QuantSpec(weight=kFp8StaticTensorSym),
    ),
    "fp8_per_block": QuantizationConfigArgs(
        linear=QuantSpec(weight=kFp8Static128BlockSym),
        moe=QuantSpec(weight=kFp8Static128BlockSym),
    ),
    # Per-output-channel weight scale + dynamic per-token activation.
    # Same shape as llmcompressor's FP8_DYNAMIC recipe.
    "fp8_per_channel": QuantizationConfigArgs(
        linear=QuantSpec(weight=kFp8StaticChannelSym),
        moe=QuantSpec(weight=kFp8StaticChannelSym),
    ),
    "mxfp8": QuantizationConfigArgs(
        linear=QuantSpec(weight=kMxfp8Dynamic),
        moe=QuantSpec(weight=kMxfp8Dynamic),
    ),
    "mxfp6": QuantizationConfigArgs(
        linear=QuantSpec(weight=kMxfp6E2m3Static, activation=kMxfp8Dynamic),
        moe=QuantSpec(weight=kMxfp6E2m3Static, activation=kMxfp8Dynamic),
        overrides=[
            QuantOverride(
                pattern=r"re:(^|.*\.)(gate|router|shared_expert_gate|lm_head)$",
                weight="bf16",
            )
        ],
    ),
    # INT8 weight-only on MoE; linear stays unquantized (no `linear` field).
    "int8_per_channel_weight_only": QuantizationConfigArgs(
        moe=QuantSpec(weight=kInt8StaticChannelSym),
    ),
    # Online NVFP4 on MoE with per-token dynamic activation scales (Blackwell +
    # FlashInfer TRTLLM only); linear stays unquantized (no `linear` field).
    "nvfp4_per_token": QuantizationConfigArgs(
        moe=QuantSpec(weight=kNvfp4Static),
    ),
}


# Names accepted by `--quantization`; "online" means "use quantization_config".
ONLINE_QUANT_SHORTHAND_NAMES: tuple[str, ...] = (
    *_ONLINE_SHORTHANDS.keys(),
    "online",
)


def resolve_quantization_config(
    quantization: str | None,
    quantization_config: dict[str, Any] | QuantizationConfigArgs | None,
) -> QuantizationConfigArgs | None:
    """Resolve `--quantization` shorthand and `--quantization-config` into a
    QuantizationConfigArgs.

    `quantization` is a CLI shorthand that desugars into a base config via
    `_ONLINE_SHORTHANDS`. `quantization_config` is a dict or pre-built args
    object. When both are given, fields explicitly set in `quantization_config`
    take precedence over the shorthand.
    """
    if quantization is not None and quantization not in ONLINE_QUANT_SHORTHAND_NAMES:
        if quantization_config is not None:
            raise ValueError(
                f"quantization_config is only supported when quantization is "
                f"one of {sorted(ONLINE_QUANT_SHORTHAND_NAMES)}, "
                f"got quantization={quantization!r}"
            )
        return None

    base = _ONLINE_SHORTHANDS.get(quantization) if quantization else None

    if quantization_config is None:
        return base

    if isinstance(quantization_config, dict):
        quantization_config = QuantizationConfigArgs(**quantization_config)

    if base is None:
        return quantization_config

    return QuantizationConfigArgs(
        linear=quantization_config.linear or base.linear,
        moe=quantization_config.moe or base.moe,
        ignore=quantization_config.ignore or base.ignore,
        overrides=[*base.overrides, *quantization_config.overrides],
    )
