from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter, UninitializedParameter

from aphrodite import _custom_ops as ops
from aphrodite.distributed import (get_tensor_model_parallel_rank,
                                   get_tensor_model_parallel_world_size,
                                   tensor_model_parallel_all_gather)
from aphrodite.modeling.layers.linear import LinearBase, LinearMethodBase
from aphrodite.modeling.parameter import ModelWeightParameter
from aphrodite.modeling.utils import set_weight_attrs
from aphrodite.quantization.base_config import QuantizationConfig


def make_group_map(q_groups, num_qrows):
    gr = q_groups.tolist()
    group_map = []
    num_groups = len(gr) // 2

    for i in range(num_groups):
        bits = gr[i * 2]
        if i < num_groups - 1:
            qrows = gr[i * 2 + 3] - gr[i * 2 + 1]
        else:
            qrows = num_qrows - gr[i * 2 + 1]
        rows = qrows * 32 // bits
        for j in range(rows):
            group_map += [i]
            group_map += [rows - j]
    return torch.tensor(group_map, dtype=torch.short, device=q_groups.device)


class Exl2Config(QuantizationConfig):
    """Config class for Exl2."""

    def __repr__(self) -> str:
        return "Exl2Config()"

    @classmethod
    def get_name(cls) -> str:
        return "exl2"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Exl2Config":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["Exl2LinearMethod"]:
        if isinstance(layer, LinearBase):
            return Exl2LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    def merge_weight(self) -> bool:
        return False

    def quant_vocab(self) -> List[bool]:
        return [False, True]

    def support_fused_moe(self) -> bool:
        return False

    def rope_style(self) -> Optional[bool]:
        return None


class Exl2LinearMethod(LinearMethodBase):
    """Linear method for Exl2.

    Args:
        quant_config: The Exl2 quantization config.
    """

    def __init__(self, quant_config: Exl2Config):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        # The shape of weight is unknown until load state dict
        # q_groups, q_invperm, q_scale, q_scale_max, q_weight, q_groups
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.exllama_state = 0
        
        # Use empty tensors with proper parameter classes for deferred loading
        output_size_per_partition = sum(output_partition_sizes)
        
        # Create uninitialized parameters that will be materialized during weight loading
        
        # q_weight has output dimension that needs sharding
        qweight = UninitializedParameter(requires_grad=False)
        set_weight_attrs(qweight, {
            "output_dim": 1,
            "ignore_warning": True
        })
        layer.register_parameter("q_weight", qweight)
        
        # q_scale has output dimension that needs sharding  
        qscale = UninitializedParameter(requires_grad=False)
        set_weight_attrs(qscale, {
            "output_dim": 1,
            "ignore_warning": True
        })
        layer.register_parameter("q_scale", qscale)
        
        # q_groups and q_invperm don't need special sharding, but q_scale_max might
        for name in ["q_groups", "q_invperm", "q_scale_max"]:
            param = UninitializedParameter(requires_grad=False)
            set_weight_attrs(param, {
                "ignore_warning": True
            })
            layer.register_parameter(name, param)
        
        # Set weight loaders after registration (following FP6/DeepSpeedFP pattern)
        custom_weight_loader = weight_loader if weight_loader else self._default_weight_loader
        
        # Set weight loader for all parameters
        for param_name in ["q_weight", "q_scale", "q_groups", "q_invperm", "q_scale_max"]:
            param = getattr(layer, param_name)
            set_weight_attrs(param, {"weight_loader": custom_weight_loader})

    def _default_weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """Default weight loader that handles UninitializedParameter materialization and copying."""
        
        # Handle UninitializedParameter materialization first
        if isinstance(param, UninitializedParameter):
            # Get output_dim for potential sharding
            output_dim = getattr(param, "output_dim", None)
            
            # Calculate the final shape after potential sharding
            final_shape = list(loaded_weight.shape)
            if output_dim is not None:
                # This parameter needs output dimension sharding
                from aphrodite.distributed import get_tensor_model_parallel_world_size
                tp_size = get_tensor_model_parallel_world_size()
                if tp_size > 1:
                    assert final_shape[output_dim] % tp_size == 0, (
                        f"Cannot shard dimension {output_dim} of size {final_shape[output_dim]} "
                        f"across {tp_size} GPUs"
                    )
                    final_shape[output_dim] = final_shape[output_dim] // tp_size
            
            # Materialize with the calculated shape
            param.materialize(final_shape, dtype=loaded_weight.dtype)
        
        # Handle sharding for materialized parameters
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None:
            from aphrodite.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            
            if tp_size > 1:
                # Narrow the loaded weight to the appropriate shard
                shard_size = param.data.shape[output_dim]
                start_idx = tp_rank * shard_size
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        
        # Copy the (potentially sharded) weight data
        assert param.data.shape == loaded_weight.shape, (
            f"Shape mismatch: param {param.data.shape} vs loaded {loaded_weight.shape}"
        )
        param.data.copy_(loaded_weight)

    def _split_exl2_weights_for_tp(self, layer: torch.nn.Module) -> None:
        """Split EXL2 weights at group boundaries for tensor parallel (only if needed)."""
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return  # No splitting needed for single GPU
            
        # For now, let the regular weight loader handle the sharding
        # The custom group-boundary splitting can be added later if needed
        # This method is kept for future enhancement
        pass

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Ensure all parameters are regular Parameters (they should already be)
        layer.q_weight = Parameter(layer.q_weight.data, requires_grad=False)
        layer.q_scale = Parameter(layer.q_scale.data, requires_grad=False)
        layer.q_groups = Parameter(layer.q_groups.data, requires_grad=False)
        layer.q_invperm = Parameter(layer.q_invperm.data, requires_grad=False)
        layer.q_scale_max = Parameter(layer.q_scale_max.data, requires_grad=False)
        
        # Split weights for tensor parallel if needed
        self._split_exl2_weights_for_tp(layer)
        
        # Perform EXL2-specific weight processing
        if layer.exllama_state == 0:
            layer.q_scale_max.data /= 256
            layer.q_invperm.data = layer.q_invperm.data.short()
            if not hasattr(layer, 'q_perm'):
                layer.q_perm = torch.argsort(layer.q_invperm).to(torch.short)
            if not hasattr(layer, 'q_group_map'):
                layer.q_group_map = make_group_map(layer.q_groups,
                                                   layer.q_weight.shape[0])
            layer.q_matrix = ops.make_q_matrix(
                layer.q_weight,
                layer.q_perm,
                layer.q_invperm,
                layer.q_scale,
                layer.q_scale_max,
                layer.q_groups,
                layer.q_group_map,
            )
            layer.exllama_state = 1

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Handle tensor parallel input gathering for EXL2
        # For row parallel layers with EXL2, we need to gather the input
        if (hasattr(layer, 'input_is_parallel') and layer.input_is_parallel and
            get_tensor_model_parallel_world_size() > 1):
            # This is a row parallel layer - gather input for EXL2
            x = tensor_model_parallel_all_gather(x)
        
        out_shape = x.shape[:-1] + (layer.q_weight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])

        # Weight processing is now handled in process_weights_after_loading
        # so we can directly use the pre-processed q_matrix
        output = ops.exl2_gemm(reshaped_x, layer.q_matrix)

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)

    def apply_moe_weights(self, w1: Dict[str,
                                         torch.Tensor], w2: Dict[str,
                                                                 torch.Tensor],
                          x: torch.Tensor, gating_output: torch.Tensor,
                          topk: int, renormalize: bool) -> torch.Tensor:
        raise NotImplementedError
