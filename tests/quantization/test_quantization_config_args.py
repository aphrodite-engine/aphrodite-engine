# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for QuantizationConfigArgs parsing."""

import pytest

from aphrodite.config.quantization import (
    QUANT_KEY_NAMES,
    QuantizationConfigArgs,
    QuantOverride,
    QuantSpec,
    resolve_quantization_config,
)
from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kInt8StaticChannelSym,
    kMxfp6E2m3Dynamic,
    kMxfp6E2m3Static,
    kMxfp8Dynamic,
)

# ---- QuantSpec ------------------------------------------------------------


def test_quant_spec_resolves_string_to_quant_key():
    spec = QuantSpec(weight="mxfp8", activation="fp8_per_token")
    assert spec.weight == kMxfp8Dynamic
    assert spec.activation == kFp8DynamicTokenSym


def test_quant_spec_accepts_quant_key_directly():
    spec = QuantSpec(weight=kFp8StaticTensorSym)
    assert spec.weight is kFp8StaticTensorSym
    assert spec.activation is None


def test_quant_spec_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown quantization name"):
        QuantSpec(weight="not_a_real_format")


# ---- QuantizationConfigArgs string shorthand on linear/moe ----------------


def test_args_linear_string_resolves_via_quant_key_names():
    # A bare QUANT_KEY_NAMES entry desugars to QuantSpec(weight=<key>).
    args = QuantizationConfigArgs(linear="fp8_per_block_static")
    assert args.linear == QuantSpec(weight=kFp8Static128BlockSym)
    assert args.moe is None


def test_args_moe_string_resolves_via_online_shorthand():
    # An online-shorthand name pulls the matching slot from _ONLINE_SHORTHANDS
    # (so `linear: "fp8_per_block"` and `moe: "fp8_per_block"` produce the
    # same per-layer-kind spec the `--quantization fp8_per_block` shorthand
    # would).
    args = QuantizationConfigArgs(moe="fp8_per_block")
    assert args.moe == QuantSpec(weight=kFp8Static128BlockSym)


def test_args_string_shorthand_missing_slot_raises():
    # int8_per_channel_weight_only sets only `moe`; using it on `linear`
    # has no defined spec and should raise rather than silently no-op.
    with pytest.raises(ValueError, match="does not define a linear spec"):
        QuantizationConfigArgs(linear="int8_per_channel_weight_only")


def test_args_accepts_dict_form():
    args = QuantizationConfigArgs(moe={"activation": "mxfp8"})
    assert args.moe == QuantSpec(weight=None, activation=kMxfp8Dynamic)


# ---- resolve_quantization_config -----------------------------------------


def test_resolve_shorthand_only_populates_both_slots():
    args = resolve_quantization_config("fp8_per_block", None)
    assert args.linear == QuantSpec(weight=kFp8Static128BlockSym)
    assert args.moe == QuantSpec(weight=kFp8Static128BlockSym)


def test_resolve_int8_shorthand_leaves_linear_unset():
    # int8_per_channel_weight_only is MoE-only; linear stays None so that
    # OnlineQuantizationConfig leaves Linear layers in full precision.
    args = resolve_quantization_config("int8_per_channel_weight_only", None)
    assert args.linear is None
    assert args.moe == QuantSpec(weight=kInt8StaticChannelSym)


def test_resolve_quantization_config_only():
    # When only `quantization_config` is given (e.g. for an already-quantized
    # checkpoint that needs an activation override), it's returned as-is.
    args = resolve_quantization_config(None, {"moe": {"activation": "mxfp8"}})
    assert args.linear is None
    assert args.moe == QuantSpec(weight=None, activation=kMxfp8Dynamic)


def test_resolve_merges_explicit_over_shorthand():
    # Explicit linear in quantization_config wins; moe falls back to the
    # shorthand's slot.
    args = resolve_quantization_config(
        "fp8_per_tensor",
        {"linear": "fp8_per_block"},
    )
    assert args.linear == QuantSpec(weight=kFp8Static128BlockSym)
    assert args.moe == QuantSpec(weight=kFp8StaticTensorSym)


def test_mxfp6_shorthand_uses_w6a8_and_preserves_routers():
    args = resolve_quantization_config("mxfp6", None)
    expected = QuantSpec(weight=kMxfp6E2m3Static, activation=kMxfp8Dynamic)
    assert args.linear == expected
    assert args.moe == expected
    assert args.overrides == [
        QuantOverride(
            pattern=r"re:(^|.*\.)(gate|router|shared_expert_gate|lm_head)$",
            weight="bf16",
        )
    ]


def test_mxfp6_w6a6_and_ordered_overrides():
    args = resolve_quantization_config(
        "mxfp6",
        {
            "linear": {
                "weight": "mxfp6_e2m3",
                "activation": "mxfp6_e2m3_dynamic",
            },
            "overrides": [
                {"pattern": "re:.*\\.gate$", "weight": "mxfp6_e2m3"},
                {"pattern": "model.layers.0.mlp.gate", "weight": "bf16"},
            ],
        },
    )
    assert args.linear == QuantSpec(
        weight=kMxfp6E2m3Static,
        activation=kMxfp6E2m3Dynamic,
    )
    assert [rule.weight for rule in args.overrides] == [
        "bf16",
        kMxfp6E2m3Static,
        "bf16",
    ]


def test_resolve_rejects_quantization_config_with_non_shorthand_quant():
    # If --quantization names something other than an online shorthand,
    # quantization_config is not allowed via this path (checkpoint quant
    # paths read it directly off ModelConfig instead).
    with pytest.raises(ValueError, match="quantization_config is only supported"):
        resolve_quantization_config("gptq", {"linear": "fp8_per_block"})


# ---- QUANT_KEY_NAMES coverage --------------------------------------------


def test_quant_key_names_round_trip():
    # Every advertised name should round-trip through QuantSpec without error
    # and produce the same QuantKey it maps to.
    for name, expected in QUANT_KEY_NAMES.items():
        assert QuantSpec(weight=name).weight == expected, name
        assert QuantSpec(activation=name).activation == expected, name


def test_static_block_weight_paired_with_dynamic_block_activation():
    # The block-FP8 shorthand pair: 128x128 static weights + 1x128 dynamic
    # activations. Pinning this so renames in QUANT_KEY_NAMES don't quietly
    # rewire the kernel dispatch.
    spec = QuantSpec(weight="fp8_per_block_static", activation="fp8_per_block_dynamic")
    assert spec.weight == kFp8Static128BlockSym
    assert spec.activation == kFp8Dynamic128Sym
