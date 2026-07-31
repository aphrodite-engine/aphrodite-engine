# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import aphrodite
import aphrodite.ir.ops
from aphrodite.compilation.passes.fusion.add_rms_fusion import AddRMSNormFusionPass, RMSNormReshapeFusionPass
from aphrodite.compilation.passes.fx_utils import find_op_nodes, is_func
from aphrodite.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from aphrodite.compilation.passes.utility.post_cleanup import PostCleanupPass
from aphrodite.config import AphroditeConfig, CompilationConfig, CompilationMode
from aphrodite.platforms import current_platform
from tests.compile.backend import TestBackend

pytestmark = pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Requires CUDA or ROCm")


class RMSNormModel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = aphrodite.ir.ops.rms_norm(x, self.weight, 1e-6)
        return rms.reshape(-1, rms.shape[-1])


class AddRMSNormModel(torch.nn.Module):
    def __init__(self, hidden_size: int, residual_first: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size))
        self.residual_first = residual_first

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = residual + x if self.residual_first else x + residual
        rms = aphrodite.ir.ops.rms_norm(residual, self.weight, 1e-6)
        return rms.reshape(-1, rms.shape[-1]), residual

    def ops_in_model_before(self):
        return [torch.ops.aten.add, torch.ops.aphrodite_ir.rms_norm]

    def ops_in_model_after(self):
        return [torch.ops.aphrodite_ir.fused_add_rms_norm]


def _run_fusion_test(model, config, passes, *inputs):
    backend = TestBackend(NoOpEliminationPass(config), *passes, PostCleanupPass(config))
    outputs_unfused = model(*inputs)
    outputs_fused = torch.compile(model, backend=backend)(*inputs)
    torch.testing.assert_close(outputs_fused, outputs_unfused)
    return backend


def _is_reshape(node):
    return isinstance(node, torch.fx.Node) and is_func(node, torch.ops.aten.reshape.default)


@pytest.fixture
def aphrodite_config():
    config = AphroditeConfig(compilation_config=CompilationConfig(mode=CompilationMode.APHRODITE_COMPILE))
    with aphrodite.config.set_current_aphrodite_config(config):
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(0)
        yield config


def test_rmsnorm_reshape_fusion(aphrodite_config):
    fusion_pass = RMSNormReshapeFusionPass(aphrodite_config)
    model = RMSNormModel(hidden_size=32)
    x = torch.randn(2, 7, 32)
    backend = _run_fusion_test(model, aphrodite_config, [fusion_pass], x)

    assert fusion_pass.matched_count == 1
    (rms_node,) = find_op_nodes(torch.ops.aphrodite_ir.rms_norm, backend.graph_post_pass)
    assert _is_reshape(rms_node.args[0])


@pytest.mark.parametrize("residual_first", [True, False])
def test_add_rmsnorm_reshape_fusion(aphrodite_config, residual_first):
    add_fusion = AddRMSNormFusionPass(aphrodite_config)
    reshape_fusion = RMSNormReshapeFusionPass(aphrodite_config)
    model = AddRMSNormModel(hidden_size=32, residual_first=residual_first)
    x = torch.randn(2, 7, 32)
    residual = torch.randn_like(x)
    backend = _run_fusion_test(model, aphrodite_config, [add_fusion, reshape_fusion], x, residual)

    assert add_fusion.matched_count == 1
    assert reshape_fusion.matched_count == 1
    backend.check_before_ops(model.ops_in_model_before())
    backend.check_after_ops(model.ops_in_model_after())
