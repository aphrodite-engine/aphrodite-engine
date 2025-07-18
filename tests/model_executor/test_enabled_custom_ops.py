import pytest

from aphrodite.common.config import CompilationConfig, AphroditeConfig, set_current_aphrodite_config
from aphrodite.modeling.custom_op import CustomOp
from aphrodite.modeling.layers.activation import (GeluAndMul,
                                                   ReLUSquaredActivation,
                                                   SiluAndMul)
from aphrodite.modeling.layers.fused_moe.fused_moe import (
    dispatch_fused_experts_func, dispatch_topk_func,
    torch_aphrodite_inplace_fused_experts, torch_aphrodite_outplace_fused_experts,
    aphrodite_topk_softmax)
from aphrodite.modeling.layers.fused_moe.rocm_aiter_fused_moe import (
    is_rocm_aiter_moe_enabled)
from aphrodite.modeling.layers.layernorm import (
    RMSNorm, dispatch_cuda_rmsnorm_func, fused_add_rms_norm, rms_norm,
    rocm_aiter_fused_add_rms_norm, rocm_aiter_rms_norm)
from aphrodite.platforms import current_platform


# Registered subclass for test
@CustomOp.register("relu3")
class Relu3(ReLUSquaredActivation):
    pass


@pytest.mark.parametrize(
    "env, torch_level, ops_enabled, default_on",
    [
        # Default values based on compile level
        ("", 0, [True] * 4, True),
        ("", 1, [True] * 4, True),
        ("", 2, [True] * 4, True),  # All by default
        ("", 3, [False] * 4, False),
        ("", 4, [False] * 4, False),  # None by default
        # Explicitly enabling/disabling
        #
        # Default: all
        #
        # All but SiluAndMul
        ("+rms_norm,-silu_and_mul", 0, [1, 0, 1, 1], True),
        # Only ReLU3
        ("none,-rms_norm,+relu3", 0, [0, 0, 0, 1], False),
        # All but SiluAndMul
        ("all,-silu_and_mul", 1, [1, 0, 1, 1], True),
        # All but ReLU3 (even if ReLU2 is on)
        ("-relu3,relu2", 1, [1, 1, 1, 0], True),
        # GeluAndMul and SiluAndMul
        ("none,-relu3,+gelu_and_mul,+silu_and_mul", 2, [0, 1, 1, 0], False),
        # All but RMSNorm
        ("-rms_norm", 2, [0, 1, 1, 1], True),
        #
        # Default: none
        #
        # Only ReLU3
        ("-silu_and_mul,+relu3", 3, [0, 0, 0, 1], False),
        # All but RMSNorm
        ("all,-rms_norm", 4, [0, 1, 1, 1], True),
    ])
def test_enabled_ops(env: str, torch_level: int, ops_enabled: list[int],
                     default_on: bool):
    aphrodite_config = AphroditeConfig(compilation_config=CompilationConfig(
        level=torch_level, custom_ops=env.split(",")))
    with set_current_aphrodite_config(aphrodite_config):
        assert CustomOp.default_on() == default_on

        ops_enabled = [bool(x) for x in ops_enabled]

        assert RMSNorm(1024).enabled() == ops_enabled[0]
        assert CustomOp.op_registry["rms_norm"].enabled() == ops_enabled[0]

        assert SiluAndMul().enabled() == ops_enabled[1]
        assert CustomOp.op_registry["silu_and_mul"].enabled() == ops_enabled[1]

        assert GeluAndMul().enabled() == ops_enabled[2]
        assert CustomOp.op_registry["gelu_and_mul"].enabled() == ops_enabled[2]

        # If registered, subclasses should follow their own name
        assert Relu3().enabled() == ops_enabled[3]
        assert CustomOp.op_registry["relu3"].enabled() == ops_enabled[3]

        # Unregistered subclass
        class SiluAndMul2(SiluAndMul):
            pass

        # Subclasses should not require registration
        assert SiluAndMul2().enabled() == SiluAndMul().enabled()


@pytest.mark.parametrize(
    "env", ["all,none", "all,+rms_norm,all", "+rms_norm,-rms_norm"])
def test_enabled_ops_invalid(env: str):
    with pytest.raises(Exception):  # noqa
        aphrodite_config = AphroditeConfig(compilation_config=CompilationConfig(
            custom_ops=env.split(",")))
        with set_current_aphrodite_config(aphrodite_config):
            RMSNorm(1024).enabled()


@pytest.mark.parametrize("use_rocm_aiter", ["0", "1"])
def test_topk_dispatch(use_rocm_aiter: str, monkeypatch):
    monkeypatch.setenv("APHRODITE_ROCM_USE_AITER", use_rocm_aiter)
    topk_func = dispatch_topk_func()
    is_rocm_aiter_moe_enabled.cache_clear()
    if current_platform.is_rocm() and int(use_rocm_aiter):
        from aphrodite.modeling.layers.fused_moe.rocm_aiter_fused_moe import (
            rocm_aiter_topk_softmax)
        assert topk_func == rocm_aiter_topk_softmax
    else:
        assert topk_func == aphrodite_topk_softmax


@pytest.mark.parametrize("use_rocm_aiter", ["0", "1"])
@pytest.mark.parametrize("inplace", [True, False])
def test_fused_experts_dispatch(use_rocm_aiter: str, inplace: bool,
                                monkeypatch):

    monkeypatch.setenv("APHRODITE_ROCM_USE_AITER", use_rocm_aiter)
    is_rocm_aiter_moe_enabled.cache_clear()
    fused_experts_func = dispatch_fused_experts_func(inplace)
    if current_platform.is_rocm() and int(use_rocm_aiter):
        from aphrodite.modeling.layers.fused_moe.rocm_aiter_fused_moe import (
            rocm_aiter_fused_experts)
        assert fused_experts_func == rocm_aiter_fused_experts
    elif inplace:
        assert fused_experts_func == torch_aphrodite_inplace_fused_experts
    else:
        assert fused_experts_func == torch_aphrodite_outplace_fused_experts


@pytest.mark.parametrize("add_residual", [True, False])
@pytest.mark.parametrize("use_rocm_aiter", ["0", "1"])
@pytest.mark.parametrize("use_rocm_aiter_norm", ["0", "1"])
@pytest.mark.skipif(not current_platform.is_rocm(),
                    reason="AITER is a feature exclusive for ROCm")
def test_rms_norm_dispatch(add_residual: bool, use_rocm_aiter: str,
                           use_rocm_aiter_norm: str, monkeypatch):
    monkeypatch.setenv("APHRODITE_ROCM_USE_AITER", use_rocm_aiter)
    monkeypatch.setenv("APHRODITE_ROCM_USE_AITER_RMSNORM", use_rocm_aiter_norm)
    rms_norm_func = dispatch_cuda_rmsnorm_func(add_residual)

    if not add_residual:
        if current_platform.is_rocm() and int(use_rocm_aiter) and int(
                use_rocm_aiter_norm):
            assert rms_norm_func == rocm_aiter_rms_norm
        else:
            assert rms_norm_func == rms_norm
    elif current_platform.is_rocm() and int(use_rocm_aiter) and int(
            use_rocm_aiter_norm):
        assert rms_norm_func == rocm_aiter_fused_add_rms_norm
    else:
        assert rms_norm_func == fused_add_rms_norm
