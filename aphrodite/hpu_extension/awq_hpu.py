# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional

import torch

from aphrodite.modeling.layers.linear import LinearBase, LinearMethodBase
from aphrodite.modeling.parameter import (GroupQuantScaleParameter,
                                          PackedAphroditeParameter)
from aphrodite.quantization.base_config import QuantizationConfig


class AWQHPUConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits.")
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (f"AWQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point})")

    def get_name(self) -> str:
        return "awq_hpu"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 0

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQHPUConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        return cls(weight_bits, group_size, zero_point)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:

        is_valid_user_quant = user_quant == "awq_hpu"

        if is_valid_user_quant:
            return cls.get_name(cls)

        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["AWQHPULinearMethod"]:
        if isinstance(layer, LinearBase):
            return AWQHPULinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class AWQHPULinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQHPUConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedAphroditeParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        qzeros = PackedAphroditeParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        scales = GroupQuantScaleParameter(data=torch.empty(
            input_size_per_partition // self.quant_config.group_size,
            output_size_per_partition,
            dtype=params_dtype,
        ),
                                          input_dim=0,
                                          output_dim=1,
                                          weight_loader=weight_loader)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def pack_tensor(self, x):
        wf = torch.tensor(list(range(0, 32, self.quant_config.weight_bits)),
                          dtype=torch.int32).unsqueeze(0)
        xp = torch.sum(torch.bitwise_left_shift(
            x.reshape(x.shape[0], -1, (32 // self.quant_config.weight_bits)),
            wf.unsqueeze(0)),
                       dim=-1).to(torch.int32)
        return xp

    def unpack_tensor(self, xp):
        wf = torch.tensor(list(range(0, 32, self.quant_config.weight_bits)),
                          dtype=torch.int32).unsqueeze(0)
        x = torch.bitwise_right_shift(
            torch.unsqueeze(xp,
                            -1).expand(xp.shape[0], -1,
                                       32 // self.quant_config.weight_bits),
            wf.unsqueeze(0)).to(torch.int8)
        x = torch.bitwise_and(x, (2**self.quant_config.weight_bits) - 1)
        x = x.reshape((x.shape[0], -1))
        return x

    def awq_order(self, x):

        order = [0, 4, 1, 5, 2, 6, 3, 7]
        idx = torch.arange(
            x.shape[-1],
            dtype=torch.int32,
            device=x.device,
        )
        idx = idx.view(-1, 32 // self.quant_config.weight_bits)
        idx = idx[:, order]
        idx = idx.view(-1)

        x = x[:, idx]
        return x

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        # This unpacking-packing is required because HPU dequant kernel
        # is not compatible with AWQ format
        device = layer.qweight.device
        wq = layer.qweight.cpu()
        zq = layer.qzeros.cpu()
        wqu = self.awq_order(self.unpack_tensor(wq))
        zu = self.awq_order(self.unpack_tensor(zq))
        layer.qweight.data = self.pack_tensor(wqu).to(device)
        layer.qzeros.data = self.pack_tensor(zu).to(device)

        layer.qweight = torch.nn.Parameter(layer.qweight.data,
                                           requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data,
                                          requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data,
                                          requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
        reshaped_x = x.reshape(-1, x.shape[-1])

        weight = torch.ops.hpu.convert_from_uint4(qweight,
                                                scales,
                                                qzeros,
                                                x.dtype)
        out = torch.matmul(reshaped_x, weight)

        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
