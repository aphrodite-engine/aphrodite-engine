# SPDX-License-Identifier: Apache-2.0
"""Handle-based kt-kernel wrapper backed by Aphrodite custom ops."""

from __future__ import annotations

import os
from typing import ClassVar

import torch

import aphrodite._custom_ops as ops

from .loader import (
    BF16SafeTensorLoader,
    CompressedSafeTensorLoader,
    FP8SafeTensorLoader,
    GPTQSafeTensorLoader,
    SafeTensorLoader,
)


def _ptr_matrix(ptrs: list[list[int]]) -> torch.Tensor:
    if not ptrs:
        return torch.empty((0, 0), dtype=torch.int64)
    return torch.tensor(ptrs, dtype=torch.int64, device="cpu").contiguous()


def _empty_ptr_matrix() -> torch.Tensor:
    return torch.empty((0, 0), dtype=torch.int64, device="cpu")


def _cpu_supports_amx() -> bool:
    return bool(getattr(torch.cpu, "_is_amx_tile_supported", lambda: False)())


def _cpu_supports_avx512_bf16() -> bool:
    probe = getattr(torch.cpu, "_is_avx512_bf16_supported", None)
    if probe is not None:
        return bool(probe())
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            flags = f.read()
    except OSError:
        return False
    return "avx512_bf16" in flags or "avx512bf16" in flags


class _CPUInferPool:
    def __init__(self, threads: int, threadpool_count: int) -> None:
        threads = max(1, int(threads))
        threadpool_count = max(1, int(threadpool_count))
        thread_counts = [
            threads // threadpool_count + (1 if i < threads % threadpool_count else 0) for i in range(threadpool_count)
        ]
        self.numa_nodes = torch.arange(threadpool_count, dtype=torch.int64, device="cpu")
        self.thread_counts = torch.tensor(thread_counts, dtype=torch.int64, device="cpu")
        self.handle = ops.kt_create_cpu_infer(self.numa_nodes, self.thread_counts)

    def __del__(self) -> None:
        handle = getattr(self, "handle", None)
        if handle is not None:
            ops.kt_destroy_cpu_infer(handle)
            self.handle = None


class _CPUBuffer:
    capture_bs: ClassVar[list[int]] = []
    capture_buffers: ClassVar[dict[int, tuple[torch.Tensor, ...]]] = {}
    temp_bs: ClassVar[int] = 0
    temp_buffer: ClassVar[tuple[torch.Tensor, ...]] = ()
    buffer_depth: ClassVar[int] = 2

    @classmethod
    def get_buffer(cls, hidden_states: torch.Tensor, top_k: int) -> tuple[torch.Tensor, ...]:
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size = flat.shape[0]
        hidden_size = flat.shape[-1]
        if batch_size in cls.capture_buffers:
            return cls.capture_buffers[batch_size]
        if batch_size == cls.temp_bs:
            return cls.temp_buffer

        input_cpu = [
            torch.empty((batch_size, hidden_size), dtype=torch.bfloat16, device="cpu", pin_memory=True)
            for _ in range(cls.buffer_depth)
        ]
        ids_cpu = [
            torch.empty((batch_size, top_k), dtype=torch.int64, device="cpu", pin_memory=True)
            for _ in range(cls.buffer_depth)
        ]
        weights_cpu = [
            torch.empty((batch_size, top_k), dtype=torch.float32, device="cpu", pin_memory=True)
            for _ in range(cls.buffer_depth)
        ]
        output_cpu = [
            torch.empty((batch_size, hidden_size), dtype=torch.bfloat16, device="cpu", pin_memory=True)
            for _ in range(cls.buffer_depth)
        ]
        batch_cpu = [
            torch.tensor([batch_size], dtype=torch.int32, device="cpu", pin_memory=True)
            for _ in range(cls.buffer_depth)
        ]
        output_gpu = [
            torch.empty((batch_size, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
            for _ in range(cls.buffer_depth)
        ]
        buffer = (input_cpu, ids_cpu, weights_cpu, output_cpu, batch_cpu, output_gpu)
        if batch_size in cls.capture_bs:
            cls.capture_buffers[batch_size] = buffer
        cls.temp_bs = batch_size
        cls.temp_buffer = buffer
        return buffer


class KTMoEWrapper:
    _cpu_pool: ClassVar[_CPUInferPool | None] = None

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        gpu_experts_mask: torch.Tensor | None = None,
        num_gpu_experts: int | None = None,
        cpuinfer_threads: int | None = None,
        threadpool_count: int = 2,
        weight_path: str = "",
        chunked_prefill_size: int = 25600,
        method: str = "AMXINT4",
        activation: str = "silu",
        max_deferred_experts_per_token: int | None = None,
        **_: object,
    ) -> None:
        if gpu_experts_mask is None:
            gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool, device="cpu")
            if num_gpu_experts:
                gpu_experts_mask[:num_gpu_experts] = True
        self.gpu_experts_mask = gpu_experts_mask.to(device="cpu", dtype=torch.bool).contiguous()
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.weight_path = weight_path
        self.chunked_prefill_size = chunked_prefill_size
        self.method = method
        self.activation = activation.lower()
        self.max_deferred_experts_per_token = max_deferred_experts_per_token or 0
        self.handle: int | None = None
        self._tensors: list[torch.Tensor] = []

        if KTMoEWrapper._cpu_pool is None:
            KTMoEWrapper._cpu_pool = _CPUInferPool(
                cpuinfer_threads or os.cpu_count() or 1,
                threadpool_count,
            )

    @staticmethod
    def set_capture_batch_sizes(capture_bs: list[int]) -> None:
        _CPUBuffer.capture_bs = capture_bs

    def _load_pointer_matrices(self) -> tuple[torch.Tensor, ...]:
        if os.path.isfile(self.weight_path):
            has_safetensors = self.weight_path.endswith(".safetensors")
        else:
            has_safetensors = any(
                entry.name.endswith(".safetensors")
                for entry in os.scandir(self.weight_path)
            )
        if not has_safetensors:
            raise FileNotFoundError(f"No safetensors files found in kt_weight_path={self.weight_path!r}")

        method = self.method.upper()
        if method == "AUTO":
            method = SafeTensorLoader(self.weight_path).infer_moe_method(self.layer_idx)
        if method == "BF16" and not _cpu_supports_amx():
            method = "BF16_AVX512" if _cpu_supports_avx512_bf16() else "BF16_AVX2"
        if method == "FP8" and not _cpu_supports_amx():
            method = "FP8_AVX2"
        if method in ("AMXINT4", "AMXINT8", "RAWINT4", "FP8_PERCHANNEL") and not _cpu_supports_amx():
            raise NotImplementedError(f"{method} kt_kernel backend requires AMX CPU support")
        self.method = method

        if self.method in ("AMXINT4", "AMXINT8"):
            loader = SafeTensorLoader(self.weight_path)
            base_key = f"blk.{self.layer_idx}"
            weights = loader.load_experts(base_key)
            gate = weights["gate"]
            up = weights["up"]
            down = weights["down"]
            gate_scale = weights["gate_scale"]
            up_scale = weights["up_scale"]
            down_scale = weights["down_scale"]
            self._tensors.extend(
                [
                    *sum(gate, []),
                    *sum(up, []),
                    *sum(down, []),
                    *sum(gate_scale, []),
                    *sum(up_scale, []),
                    *sum(down_scale, []),
                ]
            )
            return (
                _ptr_matrix([[t.data_ptr() for t in row] for row in gate]),
                _ptr_matrix([[t.data_ptr() for t in row] for row in up]),
                _ptr_matrix([[t.data_ptr() for t in row] for row in down]),
                _ptr_matrix([[t.data_ptr() for t in row] for row in gate_scale]),
                _ptr_matrix([[t.data_ptr() for t in row] for row in up_scale]),
                _ptr_matrix([[t.data_ptr() for t in row] for row in down_scale]),
            )

        if self.method == "RAWINT4":
            loader = CompressedSafeTensorLoader(self.weight_path)
        elif self.method in ("FP8", "FP8_AVX2"):
            loader = FP8SafeTensorLoader(self.weight_path)
        elif self.method == "FP8_PERCHANNEL":
            loader = FP8SafeTensorLoader(self.weight_path, scale_suffix="weight_scale")
        elif self.method in ("BF16", "BF16_AVX2", "BF16_AVX512"):
            loader = BF16SafeTensorLoader(self.weight_path)
        elif self.method == "GPTQ_INT4":
            loader = GPTQSafeTensorLoader(self.weight_path)
        else:
            raise NotImplementedError(f"Unsupported kt_kernel method: {self.method}")

        base_key = f"model.layers.{self.layer_idx}"
        try:
            weights = loader.load_experts(base_key)
        except (ValueError, KeyError):
            weights = loader.load_experts(f"model.language_model.layers.{self.layer_idx}")
        gate = [t.contiguous() for t in weights["gate"]]
        up = [t.contiguous() for t in weights["up"]]
        down = [t.contiguous() for t in weights["down"]]
        self._tensors.extend([*gate, *up, *down])
        if self.method in ("BF16", "BF16_AVX2", "BF16_AVX512"):
            gate_scale = [torch.empty((), dtype=torch.uint8, device="cpu") for _ in gate]
            up_scale = [torch.empty((), dtype=torch.uint8, device="cpu") for _ in up]
            down_scale = [torch.empty((), dtype=torch.uint8, device="cpu") for _ in down]
        else:
            gate_scale = [t.contiguous() for t in weights["gate_scale"]]
            up_scale = [t.contiguous() for t in weights["up_scale"]]
            down_scale = [t.contiguous() for t in weights["down_scale"]]
            self._tensors.extend([*gate_scale, *up_scale, *down_scale])
        return (
            _ptr_matrix([[t.data_ptr() for t in gate]]),
            _ptr_matrix([[t.data_ptr() for t in up]]),
            _ptr_matrix([[t.data_ptr() for t in down]]),
            _ptr_matrix([[t.data_ptr() for t in gate_scale]]),
            _ptr_matrix([[t.data_ptr() for t in up_scale]]),
            _ptr_matrix([[t.data_ptr() for t in down_scale]]),
        )

    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor) -> None:
        assert KTMoEWrapper._cpu_pool is not None
        gate, up, down, gate_scale, up_scale, down_scale = self._load_pointer_matrices()
        quant_bits = 4 if self.method in ("AMXINT4", "RAWINT4", "GPTQ_INT4") else 8
        group_size = 0
        if self.method == "RAWINT4":
            group_size = self.hidden_size // max(1, gate_scale.shape[-1])
        elif self.method in ("FP8", "FP8_AVX2", "GPTQ_INT4"):
            group_size = 128
        self.handle = ops.kt_create_moe(
            KTMoEWrapper._cpu_pool.handle,
            self.method,
            self.layer_idx,
            self.num_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size,
            self.gpu_experts_mask,
            self.chunked_prefill_size,
            self.weight_path,
            True,
            False,
            gate,
            up,
            down,
            gate_scale,
            up_scale,
            down_scale,
            quant_bits,
            group_size,
            False,
            self.method == "FP8_PERCHANNEL",
            1 if self.activation == "gelu" else 0,
        )
        physical_map = physical_to_logical_map_cpu.to(device="cpu", dtype=torch.int64).contiguous()
        self._tensors.append(physical_map)
        ops.kt_moe_load_weights(self.handle, physical_map)

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream: int,
    ) -> None:
        del cuda_stream
        assert self.handle is not None
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        input_cpu, ids_cpu, weights_cpu, output_cpu, batch_cpu, _output_gpu = _CPUBuffer.get_buffer(
            flat,
            self.num_experts_per_tok,
        )
        slot = self.layer_idx % _CPUBuffer.buffer_depth
        input_cpu[slot].copy_(flat.to(dtype=torch.bfloat16), non_blocking=False)
        ids_cpu[slot].copy_(topk_ids.to(dtype=torch.int64), non_blocking=False)
        weights_cpu[slot].copy_(topk_weights.to(dtype=torch.float32), non_blocking=False)
        ops.kt_moe_submit_forward(
            self.handle,
            batch_cpu[slot],
            ids_cpu[slot],
            weights_cpu[slot],
            input_cpu[slot],
            output_cpu[slot],
            False,
        )

    def sync_forward(self, hidden_states: torch.Tensor, cuda_stream: int) -> torch.Tensor:
        del cuda_stream
        assert self.handle is not None
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        _input_cpu, _ids_cpu, _weights_cpu, output_cpu, _batch_cpu, output_gpu = _CPUBuffer.get_buffer(
            flat,
            self.num_experts_per_tok,
        )
        slot = self.layer_idx % _CPUBuffer.buffer_depth
        ops.kt_moe_sync_forward(self.handle)
        output_gpu[slot].copy_(output_cpu[slot], non_blocking=True)
        return output_gpu[slot]

    def __del__(self) -> None:
        handle = getattr(self, "handle", None)
        if handle is not None:
            ops.kt_destroy_moe(handle)
            self.handle = None
