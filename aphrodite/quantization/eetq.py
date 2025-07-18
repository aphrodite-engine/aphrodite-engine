from contextlib import suppress
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from aphrodite.modeling.layers.linear import LinearBase, LinearMethodBase
from aphrodite.modeling.utils import set_weight_attrs
from aphrodite.quantization import QuantizationMethods
from aphrodite.quantization.base_config import QuantizationConfig

HAS_EETQ = False
with suppress(ImportError):
    from eetq import w8_a16_gemm
    HAS_EETQ = True


class EETQConfig(QuantizationConfig):
    """Config class for eetq.
    https://github.com/NetEase-FuXi/EETQ/tree/main
    """

    def __init__(
        self,
        weight_bits: int,
        zero_point: bool,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.zero_point = zero_point
        if self.weight_bits != 8:
            raise ValueError(
                "Currently, only 8-bit weight quantization is supported for "
                f"EETQ, but got {self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"EETQConfig(weight_bits={self.weight_bits}, "
                f"zero_point={self.zero_point})")

    def get_name(self) -> QuantizationMethods:
        return "eetq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # The EETQ kernel only supports Turing or newer GPUs.
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EETQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        return cls(weight_bits, zero_point)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["EETQLinearMethod"]:
        if isinstance(layer, LinearBase):
            return EETQLinearMethod(self)
        return None


class EETQLinearMethod(LinearMethodBase):
    """Linear method for EETQ.
    Args:
        quant_config: The EETQ quantization config.
    """

    def __init__(self, quant_config: EETQConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)
        qweight = Parameter(torch.empty(input_size_per_partition,
                                        output_size_per_partition,
                                        dtype=torch.int8),
                            requires_grad=False)
        weight_scales = Parameter(torch.empty(output_size_per_partition,
                                              dtype=torch.float16),
                                  requires_grad=False)
        set_weight_attrs(qweight, {
            "input_dim": 0,
            "output_dim": 1,
        })
        set_weight_attrs(weight_scales, {"input_dim": 0, "output_dim": 0})
        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("weight_scales", weight_scales)
        set_weight_attrs(weight_scales, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = layer.qweight.data
        weight_scales = layer.weight_scales.data

        if HAS_EETQ:
            output = w8_a16_gemm(x, qweight, weight_scales)
        else:
            raise ImportError("You have not installed EETQ. Please refer to "
                              "https://github.com/NetEase-FuXi/EETQ")

        return output

    def apply_moe_weights(self, w1: Dict[str,
                                         torch.Tensor], w2: Dict[str,
                                                                 torch.Tensor],
                          x: torch.Tensor, gating_output: torch.Tensor,
                          topk: int, renormalize: bool) -> torch.Tensor:
        raise NotImplementedError
