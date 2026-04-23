# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from aphrodite import _custom_ops as ops
from aphrodite.distributed import get_tensor_model_parallel_world_size
from aphrodite.logger import init_logger
from aphrodite.model_executor.layers.linear import LinearBase, LinearMethodBase
from aphrodite.model_executor.layers.quantization import QuantizationMethods
from aphrodite.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from aphrodite.model_executor.parameter import BaseAphroditeParameter
from aphrodite.platforms import current_platform

logger = init_logger(__name__)


class Exl3Config(QuantizationConfig):
    """Config class for ExLlamaV3 EXL3 checkpoints.

    This implementation is inference-only and intentionally starts with dense
    linear layers. EXL3 MoE and quantization/conversion tools are not supported.
    """

    def __init__(
        self,
        bits: float | None = None,
        head_bits: float | None = None,
        tensor_storage: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.head_bits = head_bits
        self.tensor_storage = tensor_storage or {}

    def get_name(self) -> QuantizationMethods:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # EXL3 kernels operate on fp16 activations. bf16 models are accepted by
        # casting at the linear boundary and returning the original dtype.
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(
            bits=config.get("bits"),
            head_bits=config.get("head_bits"),
            tensor_storage=config.get("tensor_storage"),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return Exl3LinearMethod(self)
        return None


class Exl3Parameter(BaseAphroditeParameter):
    """Small placeholder parameter that stores EXL3 tensors loaded from disk.

    Packed layers such as QKV and gate-up load multiple HF tensors into one
    Aphrodite module. The actual tensors can have different shapes, so a single
    dense Parameter cannot represent them. This parameter keeps the tensors in a
    shard dictionary keyed by the loader shard id.
    """

    def __new__(cls, *, weight_loader):
        data = torch.empty(0, dtype=torch.uint8)
        return super().__new__(cls, data=data, weight_loader=weight_loader)

    def __init__(self, *, weight_loader):
        self.exl3_tensors: dict[str | int | None, torch.Tensor] = {}
        super().__init__(data=self.data, weight_loader=weight_loader)

    def load_exl3_weight(
        self,
        loaded_weight: torch.Tensor,
        shard_id: str | int | None = None,
    ) -> None:
        self.exl3_tensors[shard_id] = loaded_weight.contiguous()


def _exl3_weight_loader(
    param: Exl3Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: str | int | None = None,
) -> None:
    param.load_exl3_weight(loaded_weight, loaded_shard_id)


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Exl3Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("EXL3 currently supports tensor_parallel_size=1")
        if not current_platform.is_cuda():
            raise NotImplementedError("EXL3 is only supported on CUDA")

        layer.input_size_per_partition = input_size_per_partition
        layer.exl3_output_partition_sizes = output_partition_sizes
        layer.exl3_shard_ids = self._shard_ids_for_layer(layer, output_partition_sizes)

        for name in ("suh", "svh", "trellis", "mcg", "mul1"):
            layer.register_parameter(
                name,
                Exl3Parameter(weight_loader=_exl3_weight_loader),
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        missing: list[str] = []
        for attr in ("suh", "svh", "trellis"):
            param = getattr(layer, attr)
            for shard_id in layer.exl3_shard_ids:
                if shard_id not in param.exl3_tensors:
                    missing.append(f"{attr}[{shard_id!r}]")
        if missing:
            prefix = getattr(layer, "prefix", layer.__class__.__name__)
            raise ValueError(f"Missing EXL3 tensors for {prefix}: {', '.join(missing)}")

        for shard_id in layer.exl3_shard_ids:
            if shard_id in layer.mcg.exl3_tensors and shard_id in layer.mul1.exl3_tensors:
                prefix = getattr(layer, "prefix", layer.__class__.__name__)
                raise ValueError(
                    f"EXL3 tensor {prefix}[{shard_id!r}] specifies both mcg and mul1"
                )

        device = torch.device("cuda", torch.cuda.current_device())
        for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
            param = getattr(layer, attr)
            for shard_id, tensor in list(param.exl3_tensors.items()):
                param.exl3_tensors[shard_id] = tensor.to(device=device, non_blocking=True)

        self._setup_mgemm_if_supported(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = x.shape[:-1]
        original_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.dtype != torch.float16:
            x_2d = x_2d.to(torch.float16)
        else:
            x_2d = x_2d.contiguous()

        if getattr(layer, "exl3_can_mgemm", False) and x_2d.shape[0] <= 32:
            output = self._apply_fused_small_batch(layer, x_2d)
        else:
            outputs = [
                self._apply_one(layer, x_2d, shard_id)
                for shard_id in layer.exl3_shard_ids
            ]
            output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if bias is not None:
            output = output + bias
        output = output.reshape(*original_shape, output.shape[-1])
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output

    def _apply_one(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        shard_id: str | int | None,
    ) -> torch.Tensor:
        suh = layer.suh.exl3_tensors[shard_id]
        svh = layer.svh.exl3_tensors[shard_id]
        trellis = layer.trellis.exl3_tensors[shard_id]
        mcg = shard_id in layer.mcg.exl3_tensors
        mul1 = shard_id in layer.mul1.exl3_tensors

        out_features = trellis.shape[1] * 16
        output = torch.empty(
            (x.shape[0], out_features),
            device=x.device,
            dtype=torch.float16,
        )
        x_had = torch.empty_like(x)

        if x.shape[0] <= 32:
            ops.exl3_gemm(
                x,
                trellis,
                output,
                suh,
                x_had,
                svh,
                -1,
                mcg,
                mul1,
                0,
            )
            return output

        weight = torch.empty(
            (trellis.shape[0] * 16, trellis.shape[1] * 16),
            device=trellis.device,
            dtype=torch.float16,
        )
        ops.exl3_reconstruct(
            weight,
            trellis,
            trellis.shape[2] // 16,
            mcg,
            mul1,
        )
        ops.exl3_had_r_128(
            x,
            x_had,
            suh,
            None,
            1.0,
        )
        ops.exl3_hgemm(
            x_had,
            weight,
            output,
        )
        ops.exl3_had_r_128(
            output,
            output,
            None,
            svh,
            1.0,
        )
        return output

    def _apply_fused_small_batch(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self._apply_mgemm(layer, x)
        if getattr(layer, "exl3_mgemm_mode", None) == "qkv_kv":
            q = self._apply_one(layer, x, "q")
            return torch.cat([q, output[0], output[1]], dim=-1)
        return torch.cat([output[i] for i in range(output.shape[0])], dim=-1)

    def _apply_mgemm(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        x_3d = x.view(1, x.shape[0], x.shape[1])
        output = torch.empty(
            (
                layer.exl3_mgemm_num_shards,
                x.shape[0],
                layer.exl3_mgemm_out_features,
            ),
            device=x.device,
            dtype=torch.float16,
        )
        x_had = torch.empty(
            (layer.exl3_mgemm_num_shards, x.shape[0], x.shape[1]),
            device=x.device,
            dtype=torch.float16,
        )
        ops.exl3_mgemm(
            x_3d,
            layer.exl3_mgemm_ptrs_trellis,
            output,
            layer.exl3_mgemm_ptrs_suh,
            x_had,
            layer.exl3_mgemm_ptrs_svh,
            None,
            None,
            layer.exl3_mgemm_k,
            -1,
            layer.exl3_mgemm_mcg,
            layer.exl3_mgemm_mul1,
            -1,
            -1,
            0,
        )
        return output

    @staticmethod
    def _setup_mgemm_if_supported(layer: torch.nn.Module) -> None:
        layer.exl3_can_mgemm = False

        prefix = getattr(layer, "prefix", "")
        if prefix.endswith("gate_up_proj") and len(layer.exl3_shard_ids) == 2:
            mgemm_shard_ids = layer.exl3_shard_ids
            layer.exl3_mgemm_mode = "gate_up"
        elif prefix.endswith("qkv_proj") and layer.exl3_shard_ids == ["q", "k", "v"]:
            mgemm_shard_ids = ["k", "v"]
            layer.exl3_mgemm_mode = "qkv_kv"
        else:
            return

        trellises = [
            layer.trellis.exl3_tensors[shard_id] for shard_id in mgemm_shard_ids
        ]
        suhs = [layer.suh.exl3_tensors[shard_id] for shard_id in mgemm_shard_ids]
        svhs = [layer.svh.exl3_tensors[shard_id] for shard_id in mgemm_shard_ids]

        first_trellis = trellises[0]
        first_suh = suhs[0]
        first_svh = svhs[0]
        mcg = mgemm_shard_ids[0] in layer.mcg.exl3_tensors
        mul1 = mgemm_shard_ids[0] in layer.mul1.exl3_tensors
        if any(tensor.shape != first_trellis.shape for tensor in trellises[1:]):
            return
        if any(tensor.shape != first_suh.shape for tensor in suhs[1:]):
            return
        if any(tensor.shape != first_svh.shape for tensor in svhs[1:]):
            return
        if any(
            (shard_id in layer.mcg.exl3_tensors) != mcg
            for shard_id in mgemm_shard_ids[1:]
        ):
            return
        if any(
            (shard_id in layer.mul1.exl3_tensors) != mul1
            for shard_id in mgemm_shard_ids[1:]
        ):
            return

        device = first_trellis.device
        layer.exl3_mgemm_ptrs_trellis = torch.tensor(
            [tensor.data_ptr() for tensor in trellises],
            dtype=torch.long,
            device=device,
        )
        layer.exl3_mgemm_ptrs_suh = torch.tensor(
            [tensor.data_ptr() for tensor in suhs],
            dtype=torch.long,
            device=device,
        )
        layer.exl3_mgemm_ptrs_svh = torch.tensor(
            [tensor.data_ptr() for tensor in svhs],
            dtype=torch.long,
            device=device,
        )
        layer.exl3_mgemm_k = first_trellis.shape[2] // 16
        layer.exl3_mgemm_out_features = first_trellis.shape[1] * 16
        layer.exl3_mgemm_num_shards = len(mgemm_shard_ids)
        layer.exl3_mgemm_mcg = mcg
        layer.exl3_mgemm_mul1 = mul1
        layer.exl3_can_mgemm = True

    @staticmethod
    def _shard_ids_for_layer(
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
    ) -> list[str | int | None]:
        if len(output_partition_sizes) == 1:
            return [None]

        prefix = getattr(layer, "prefix", "")
        if prefix.endswith("qkv_proj"):
            return ["q", "k", "v"]
        if prefix.endswith("gate_up_proj"):
            return [0, 1]

        return list(range(len(output_partition_sizes)))
