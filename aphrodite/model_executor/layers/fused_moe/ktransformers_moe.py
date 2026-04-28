# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ktransformers CPU+GPU hybrid execution for routed MoE experts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from aphrodite.distributed import get_tensor_model_parallel_rank
from aphrodite.logger import init_logger
from aphrodite.model_executor.custom_op import CustomOp
from aphrodite.model_executor.layers.fused_moe.activation import MoEActivation
from aphrodite.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from aphrodite.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from aphrodite.platforms import current_platform

if TYPE_CHECKING:
    from aphrodite.config.ktransformers import KTransformersMoEConfig
    from aphrodite.model_executor.layers.fused_moe.layer import FusedMoE

logger = init_logger(__name__)


def _try_import_kt_moe_wrapper() -> type[Any]:
    try:
        from aphrodite.kernels.kt_kernel import KTMoEWrapper
    except ImportError as err:
        raise ImportError(
            "Aphrodite was not built with vendored kt_kernel support. Rebuild Aphrodite or remove --kt-weight-path."
        ) from err
    return KTMoEWrapper


@CustomOp.register("ktransformers_moe_wrapper")
class KTransformersMoEWrapperMethod(FusedMoEMethodBase, CustomOp):
    """Wrap a GPU MoE method and defer high-numbered experts to ktransformers."""

    is_ktransformers_moe_wrapper = True

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTransformersMoEConfig,
        *,
        layer_idx: int,
        num_layers: int | None,
    ) -> None:
        super().__init__(gpu_method.moe)
        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        assert kt_config.kt_num_gpu_experts is not None
        self.num_gpu_experts = kt_config.kt_num_gpu_experts
        self.tp_rank = get_tensor_model_parallel_rank()
        self.wrapper: Any | None = None

    @property
    def supports_internal_mk(self) -> bool:
        # Avoid replacing this wrapper with FusedMoEModularMethod. The wrapped
        # method owns its own kernel state and is invoked explicitly in apply().
        return True

    @property
    def supports_eplb(self) -> bool:
        return False

    @property
    def supports_shared_expert_overlap(self) -> bool:
        return self.gpu_method.supports_shared_expert_overlap

    @property
    def method_name(self) -> str:
        return f"ktransformers_{self.gpu_method.method_name}"

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return self.gpu_method.topk_indices_dtype

    @property
    def skip_forward_padding(self) -> bool:
        return self.gpu_method.skip_forward_padding

    @property
    def is_monolithic(self) -> bool:
        return False

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        return self.gpu_method.maybe_roundup_sizes(
            hidden_size,
            intermediate_size_per_partition,
            act_dtype,
            moe_parallel_config,
        )

    def uses_weight_scale_2_pattern(self) -> bool:
        return self.gpu_method.uses_weight_scale_2_pattern()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        self._validate_layer(layer, num_experts)
        self._install_gpu_expert_map(layer, num_experts)

        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        if self.tp_rank != 0:
            return

        KTMoEWrapper = _try_import_kt_moe_wrapper()
        capture_sizes = layer.aphrodite_config.compilation_config.cudagraph_capture_sizes
        if capture_sizes and hasattr(KTMoEWrapper, "set_capture_batch_sizes"):
            KTMoEWrapper.set_capture_batch_sizes(capture_sizes)

        max_deferred = self.kt_config.kt_max_deferred_experts_per_token or 0
        if self.num_layers is not None and self.layer_idx == self.num_layers - 1:
            max_deferred = 0

        self.wrapper = KTMoEWrapper(
            layer_idx=self.layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=layer.top_k,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size_per_partition * layer.tp_size,
            num_gpu_experts=self.num_gpu_experts,
            cpuinfer_threads=self.kt_config.kt_cpuinfer,
            threadpool_count=self.kt_config.kt_threadpool_count,
            weight_path=self.kt_config.kt_weight_path,
            chunked_prefill_size=layer.aphrodite_config.scheduler_config.max_num_batched_tokens,
            method=self.kt_config.kt_method,
            activation=layer.activation.value,
            max_deferred_experts_per_token=max_deferred,
        )

    def _install_gpu_expert_map(self, layer: torch.nn.Module, num_experts: int) -> None:
        expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
        expert_map[: self.num_gpu_experts] = torch.arange(
            self.num_gpu_experts,
            dtype=torch.int32,
        )
        if "_expert_map" not in layer._buffers:
            delattr(layer, "_expert_map")
            layer.register_buffer("_expert_map", expert_map)
        else:
            layer._buffers["_expert_map"] = expert_map

    def _validate_layer(self, layer: torch.nn.Module, num_experts: int) -> None:
        if not current_platform.is_cuda():
            raise NotImplementedError("ktransformers hybrid MoE is currently supported only on CUDA.")
        if layer.tp_size != 1:
            raise NotImplementedError("ktransformers hybrid MoE currently supports tensor_parallel_size=1 only.")
        if layer.use_ep or layer.enable_eplb:
            raise NotImplementedError("ktransformers hybrid MoE does not support EP or EPLB yet.")
        if layer.activation not in (MoEActivation.SILU, MoEActivation.GELU) or not layer.moe_config.is_act_and_mul:
            raise NotImplementedError("ktransformers hybrid MoE currently supports only gated SiLU/GELU experts.")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError("ktransformers hybrid MoE does not support apply_router_weight_on_input yet.")
        if not 0 < self.num_gpu_experts < num_experts:
            raise ValueError(
                "kt_num_gpu_experts must be greater than 0 and smaller than "
                f"the local expert count ({num_experts}); got {self.num_gpu_experts}."
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.gpu_method.process_weights_after_loading(layer)
        self.moe_quant_config = self.gpu_method.moe_quant_config
        self.moe_kernel = self.gpu_method.moe_kernel

        if self.tp_rank == 0 and self.wrapper is not None:
            torch.cuda.synchronize()
            physical_to_logical_map = torch.arange(
                layer.global_num_experts,
                dtype=torch.int64,
            ).contiguous()
            self.wrapper.load_weights(physical_to_logical_map)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return self.gpu_method.get_fused_moe_quant_config(layer)

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.tp_rank == 0:
            assert self.wrapper is not None
            self.wrapper.submit_forward(
                x,
                topk_ids,
                topk_weights,
                torch.cuda.current_stream(x.device).cuda_stream,
            )

        gpu_output = self.gpu_method.apply(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts_input=shared_experts_input,
        )

        if self.tp_rank != 0:
            return gpu_output

        assert self.wrapper is not None
        cpu_output = self.wrapper.sync_forward(
            x,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        return gpu_output + cpu_output

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("ktransformers hybrid MoE uses the non-monolithic routed expert path.")
