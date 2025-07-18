from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from aphrodite import _custom_ops as ops
from aphrodite.modeling.layers.linear import LinearBase, LinearMethodBase
from aphrodite.modeling.utils import set_weight_attrs
from aphrodite.quantization import QuantizationMethods
from aphrodite.quantization.base_config import QuantizationConfig
from aphrodite.quantization.quip_utils import (get_hadK, get_packed_abs_grid,
                                               matmul_hadU_cuda,
                                               matmul_hadUt_cuda)


class QuipConfig(QuantizationConfig):
    """Config class for Quip.

    Reference: https://cornell-relaxml.github.io/quip-sharp/
    """

    def __init__(self, codebook: int, use_rand: bool) -> None:
        super().__init__()
        self.codebook = codebook
        self.use_rand = use_rand

        if self.codebook != "E8P12":
            raise ValueError("Currently, only E8P12 is supported for "
                             f"Quip, but got {self.codebook}.")

    def __repr__(self) -> str:
        return (f"QuipConfig(codebook={self.codebook}, "
                f"rescale_WH={self.rescale_WH})")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "quip"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuipConfig":
        codebook = cls.get_from_keys(config, ["codebook"])
        use_rand = cls.get_from_keys(config, ["use_rand"])
        return cls(codebook, use_rand)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuipLinearMethod"]:
        if isinstance(layer, LinearBase):
            return QuipLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class QuipLinearMethod(LinearMethodBase):
    """Linear method for Quip.

    Args:
        quant_config: The Quip quantization config.
    """

    def __init__(self, quant_config: QuipConfig):
        self.quant_config = quant_config
        self.grid_packed_abs = get_packed_abs_grid().to(device="cuda")
        self.pack = 8
        self.idx_dtype = torch.int16

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        if (input_size != input_size_per_partition
                or output_size != output_size_per_partition):
            raise ValueError(
                "Currently Quip doesn't support tensor parallel yet")

        had_left, K_left, q_in_features = get_hadK(input_size,
                                                   self.quant_config.use_rand)
        had_right, K_right, q_out_features = get_hadK(
            output_size, self.quant_config.use_rand)

        if had_left is not None:
            layer.register_parameter(
                "had_left",
                Parameter(
                    had_left.to(dtype=params_dtype, device="cuda"),
                    requires_grad=False,
                ))
            set_weight_attrs(layer.had_left, extra_weight_attrs)
        if had_right is not None:
            layer.register_parameter(
                "had_right",
                Parameter(
                    had_right.to(dtype=params_dtype, device="cuda"),
                    requires_grad=False,
                ))
            set_weight_attrs(layer.had_right, extra_weight_attrs)
        layer.register_parameter(
            "Qidxs",
            Parameter(
                torch.empty(q_out_features,
                            q_in_features // self.pack,
                            device="cuda",
                            dtype=self.idx_dtype),
                requires_grad=False,
            ))
        set_weight_attrs(layer.Qidxs, extra_weight_attrs)
        layer.register_parameter(
            "Wscale",
            Parameter(
                torch.ones((), dtype=torch.float, device="cuda"),
                requires_grad=False,
            ))
        set_weight_attrs(layer.Wscale, extra_weight_attrs)
        layer.register_parameter(
            "SU",
            Parameter(
                torch.ones(
                    input_size,
                    device="cuda",
                    dtype=params_dtype,
                ),
                requires_grad=False,
            ))
        set_weight_attrs(layer.SU, extra_weight_attrs)
        layer.register_parameter(
            "SV",
            Parameter(
                torch.ones(
                    output_size,
                    device="cuda",
                    dtype=params_dtype,
                ),
                requires_grad=False,
            ))
        set_weight_attrs(layer.SV, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # First run
        if isinstance(layer.Wscale, torch.Tensor):
            layer.Wscale = layer.Wscale.item()
            if "SU" in layer and torch.all(layer.SU > 0):
                del layer.SU
            if "SV" in layer and torch.all(layer.SV > 0):
                del layer.SV

        reshaped_x = x.reshape(-1, x.shape[-1])
        out_dim = layer.Qidxs.shape[0]

        if "SU" in layer:
            reshaped_x = reshaped_x * layer.SU
        reshaped_x = matmul_hadUt_cuda(reshaped_x, layer.get("had_left", None),
                                       layer.K_left, layer.q_in_features,
                                       layer.Wscale)

        m, n = layer.Qidxs.shape
        if reshaped_x.size(0) < 32:
            out = ops.quip_gemv(reshaped_x, layer.Qidxs, self.grid_packed_abs)
        else:
            W_decompressed = torch.empty(m,
                                         n * 8,
                                         dtype=torch.float16,
                                         device=x.device)
            ops.quip_decompress(layer.Qidxs, self.grid_packed_abs,
                                W_decompressed)
            out = reshaped_x @ W_decompressed.T

        out = matmul_hadU_cuda(out, layer.get("had_right",
                                              None), layer.K_right,
                               layer.q_out_features)[..., :out_dim]
        if "SV" in layer:
            out = out * layer.SV
        out = out.view(*x.shape[:-1], out.shape[-1])
        out = out.add_(bias) if bias is not None else out
        return out
