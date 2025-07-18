from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger

from aphrodite import _custom_ops as ops
from aphrodite.distributed import get_tensor_model_parallel_rank
from aphrodite.modeling.layers.linear import LinearBase, LinearMethodBase
from aphrodite.modeling.utils import set_weight_attrs
from aphrodite.quantization import QuantizationMethods
from aphrodite.quantization.base_config import QuantizationConfig
from aphrodite.quantization.utils.fp6_utils import (_SPLIT_K_MAP,
                                                    from_scaled_tc_fpx,
                                                    to_scaled_tc_fpx)


class QuantLLMFPConfig(QuantizationConfig):
    """Config for QuantLLM FP quantizer. It supports fp2, fp3, fp4,
    fp5, fp6, fp7.

    Reference: https://arxiv.org/abs/2401.14112

    Args:
        weight_bits: the target quantization bits, should be one of
            2, 3, 4, 5, 6, 7.
    """

    def __init__(
        self,
        weight_bits: int = 6,
        exp_bits: int = 2,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.exponent_bits = exp_bits

        self.mantissa_bits = weight_bits - self.exponent_bits - 1

        self.valid_types = [torch.float16]

        if self.weight_bits not in [2, 3, 4, 5, 6, 7]:
            raise ValueError(
                "Currently, only 4-bit, 5-bit, 6-bit, and 7-bit "
                "quantization are "
                f"supported for QuantLLM FP quantizaiton, but got "
                f"{self.weight_bits} bits.")

        # Defer logging until model parallel groups are initialized
        self._logged = False

    def __repr__(self) -> str:
        return (f"QuantLLMFPConfig(weight_bits={self.weight_bits}), "
                f"exponent_bits={self.exponent_bits}")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "QuantLLMFP"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuantLLMFPConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        exp_bits = cls.get_from_keys(config, ["exp_bits"])
        return cls(weight_bits=weight_bits, exp_bits=exp_bits)

    def get_linear_method(self) -> "QuantLLMFPLinearMethod":
        return QuantLLMFPLinearMethod(self)


    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    def get_quant_method(
            self,
            layer: torch.nn.Module,
            prefix: str) -> Optional["QuantLLMFPLinearMethod"]:
        if isinstance(layer, LinearBase):
            # Log quantization info when the first linear layer is encountered
            self._log_if_needed()
            return QuantLLMFPLinearMethod(self)
        return None

    def _log_if_needed(self):
        """Log quantization info only once from rank 0 after model parallel groups are initialized."""
        if not self._logged:
            from aphrodite.distributed.parallel_state import model_parallel_is_initialized
            if model_parallel_is_initialized():
                try:
                    if get_tensor_model_parallel_rank() == 0:
                        logger.info(f"Loading model in FP{self.weight_bits}_E"
                                    f"{self.exponent_bits}M{self.mantissa_bits} format.")
                        self._logged = True
                except AssertionError:
                    # Still not initialized, skip logging
                    pass


class QuantLLMFPLinearMethod(LinearMethodBase):
    """Linear method for QuantLLMFP quantizer.
    Args:
        quant_config: the QuantLLMFP quantization config.
    """

    def __init__(self, quant_config: QuantLLMFPConfig):
        self.quant_config = quant_config
        self.weight = None

    def create_weights(self,
                       layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int],
                       input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype,
                       weight_loader=None,
                       **extra_weight_attrs):
        del output_size
        del input_size
        output_size_per_partition = sum(output_partition_sizes)
        weight = QuantLLMFPParameter(
            torch.Size((output_size_per_partition, input_size_per_partition)),
            params_dtype=params_dtype,
            quant_config=self.quant_config,
        )
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
        })
        layer.register_parameter("weight", weight)

        def quant_weight_loader(param, loaded_weight, *args, **kwargs):
            # Calls the original weight loader (if any), quantizes the result,
            # and then loads the quantized parameter.
            if weight_loader is not None:
                orig_param_data = param.data
                param.data = param.quant_llmdequantize()
                weight_loader(param, loaded_weight, *args, **kwargs)
                param.data, loaded_weight = orig_param_data, param.data
            param.quant_llmquantize_(loaded_weight.cuda())

        extra_weight_attrs["weight_loader"] = quant_weight_loader
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self,
              layer,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = layer.weight
        weights = weight.data
        scales = weight.scales
        out_dim, in_dim = weights.shape
        bsize = x.shape[0]
        splitK = _SPLIT_K_MAP[(bsize - 1) // 64].get(
            out_dim, 1) if bsize <= 768 else 1
        if bias is None:
            return ops.fp_eXmY_linear_forward_cuda(
                self.quant_config.exponent_bits,
                self.quant_config.mantissa_bits,
                x, weights, scales, splitK)
        else:
            return ops.fp_eXmY_linear_forward_cuda(
                self.quant_config.exponent_bits,
                self.quant_config.mantissa_bits,
                x, weights, scales, splitK) + bias

class QuantLLMFPParameter(nn.Parameter):
    """
    QuantLLMFP quantized parameter class that implements fp5/fp6/fp7
    quantization. Weights are stored in quantized form on
    GPUs, and can be directly applied to float16 activations.
    """

    def __new__(cls, orig_shape: torch.Size, params_dtype: torch.dtype,
                quant_config: QuantLLMFPConfig):

        data = torch.empty(torch.Size((orig_shape[0],
                            orig_shape[1] * quant_config.weight_bits // 8)),
                                   dtype=torch.uint8)


        self = torch.Tensor._make_subclass(cls, data, data.requires_grad)
        self.scales = torch.empty(orig_shape[0],
                                  dtype=torch.float16)
        self.quant_config = quant_config
        self.orig_shape = orig_shape
        return self

    def quant_llmquantize_(self, tensor: torch.Tensor):
        assert tensor.device.type == "cuda" and tensor.dtype != torch.int8
        data, scales = to_scaled_tc_fpx(
            tensor.data, self.quant_config.exponent_bits,
            self.quant_config.mantissa_bits)
        self.data.copy_(data)
        self.scales.copy_(scales)

    def quant_llmdequantize(self, output_dtype=None):
        output_dtype = output_dtype or torch.get_default_dtype()
        return from_scaled_tc_fpx(self.data, self.quant_config.exponent_bits,
                        self.quant_config.mantissa_bits, self.scales
                        ).to(output_dtype)
