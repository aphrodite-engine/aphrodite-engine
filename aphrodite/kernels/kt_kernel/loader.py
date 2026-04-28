# SPDX-License-Identifier: Apache-2.0
"""Safetensors loaders for vendored kt_kernel CPU MoE weights."""

from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open


class SafeTensorLoader:
    """Load expert tensors from one or more safetensors files."""

    def __init__(self, file_path: str):
        self.file_handle_map: dict[str, object] = {}
        self.tensor_file_map: dict[str, str] = {}
        self._load_tensor_file_map(file_path)

    def _load_tensor_file_map(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Path not found: {file_path}")

        folder_path = os.path.dirname(file_path) if os.path.isfile(file_path) else file_path
        for root, _, files in os.walk(folder_path):
            for filename in sorted(files):
                if not filename.endswith(".safetensors"):
                    continue
                path = os.path.join(root, filename)
                handle = safe_open(path, framework="pt")
                self.file_handle_map[path] = handle
                for key in handle.keys():
                    self.tensor_file_map[key] = path

        if not self.tensor_file_map:
            raise FileNotFoundError(f"No safetensors files found in {folder_path}")

    def has_tensor(self, name: str) -> bool:
        return name in self.tensor_file_map

    def load_tensor(self, key: str, device: str = "cpu") -> torch.Tensor:
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in safetensors files")
        tensor = self.file_handle_map[self.tensor_file_map[key]].get_tensor(key)
        return tensor if device == "cpu" else tensor.to(device)

    def close_all_handles(self) -> None:
        self.file_handle_map.clear()

    def infer_moe_method(self, layer_idx: int) -> str:
        """Infer the kt_kernel method from safetensors tensor names/dtypes."""
        kt_prefix = f"blk.{layer_idx}.ffn_up_exps.0.numa.0"
        if self.has_tensor(f"{kt_prefix}.weight"):
            # ktransformers prepacked checkpoints do not encode the kernel
            # variant in tensor names. Keep the legacy kt_kernel default.
            return "AMXINT4"

        sample_keys = list(self.tensor_file_map.keys())[:5000]
        if any(".experts.0." in key and ".qweight" in key for key in sample_keys):
            return "GPTQ_INT4"
        if any(".experts.0." in key and ".weight_packed" in key for key in sample_keys):
            return "RAWINT4"

        for key in sample_keys:
            if key.endswith(".experts.gate_up_proj"):
                tensor = self.load_tensor(key)
                if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    return "FP8"
                if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    return "BF16"
            if ".experts.0." not in key or not key.endswith(".weight"):
                continue
            tensor = self.load_tensor(key)
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                if self.has_tensor(key.replace(".weight", ".weight_scale")):
                    scale = self.load_tensor(key.replace(".weight", ".weight_scale"))
                    if scale.dim() == 1 or (scale.dim() == 2 and scale.shape[-1] == 1):
                        return "FP8_PERCHANNEL"
                return "FP8"
            if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
                return "BF16"

        raise ValueError(
            "Could not infer kt_kernel MoE weight format from safetensors. "
            "Pass --kt-method explicitly."
        )

    def load_experts(self, base_key: str, device: str = "cpu") -> dict[str, list[list[torch.Tensor]]]:
        """Load kt prepacked AMXINT expert tensors.

        Expected names:
        blk.{layer}.ffn_{up,gate,down}_exps.{expert}.numa.{numa}.{weight,scale}
        """
        up_base_key = f"{base_key}.ffn_up_exps"
        gate_base_key = f"{base_key}.ffn_gate_exps"
        down_base_key = f"{base_key}.ffn_down_exps"

        expert_count = 0
        while self.has_tensor(f"{up_base_key}.{expert_count}.numa.0.weight"):
            expert_count += 1
        numa_count = 0
        while self.has_tensor(f"{up_base_key}.0.numa.{numa_count}.weight"):
            numa_count += 1
        if expert_count == 0 or numa_count == 0:
            raise ValueError(f"No prepacked kt experts found for key {base_key}")

        def load_matrix(prefix: str, suffix: str) -> list[list[torch.Tensor]]:
            return [
                [
                    self.load_tensor(f"{prefix}.{expert_id}.numa.{numa_id}.{suffix}", device).contiguous()
                    for expert_id in range(expert_count)
                ]
                for numa_id in range(numa_count)
            ]

        return {
            "up": load_matrix(up_base_key, "weight"),
            "gate": load_matrix(gate_base_key, "weight"),
            "down": load_matrix(down_base_key, "weight"),
            "up_scale": load_matrix(up_base_key, "scale"),
            "gate_scale": load_matrix(gate_base_key, "scale"),
            "down_scale": load_matrix(down_base_key, "scale"),
        }


class _RegularMoELoader(SafeTensorLoader):
    MOE_FORMATS = {
        "deepseek": ("{base}.mlp.experts", "gate_proj", "up_proj", "down_proj"),
        "mixtral": ("{base}.block_sparse_moe.experts", "w1", "w3", "w2"),
        "mistral": ("{base}.experts", "w1", "w3", "w2"),
    }

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._detected_format = "deepseek"
        self._is_vl_model = False

    def _detect_format(self, weight_suffix: str = "weight") -> None:
        sample_keys = list(self.tensor_file_map.keys())[:2000]
        for fmt_name, (_, gate, _, _) in self.MOE_FORMATS.items():
            for key in sample_keys:
                if ".experts." not in key or f".{gate}.{weight_suffix}" not in key:
                    continue
                if fmt_name == "mixtral" and ".block_sparse_moe.experts." in key:
                    self._detected_format = fmt_name
                    return
                if fmt_name == "deepseek" and ".mlp.experts." in key:
                    self._detected_format = fmt_name
                    self._is_vl_model = key.startswith("model.language_model.")
                    return
                if fmt_name == "mistral" and ".mlp.experts." not in key and ".block_sparse_moe.experts." not in key:
                    self._detected_format = fmt_name
                    return

    def _get_experts_prefix_candidates(self, base_key: str) -> list[str]:
        path_tpl, _, _, _ = self.MOE_FORMATS[self._detected_format]
        candidates = []
        if self._is_vl_model:
            candidates.append(path_tpl.format(base=base_key.replace("model.layers", "model.language_model.layers")))
        candidates.append(path_tpl.format(base=base_key))
        if base_key.startswith("model."):
            candidates.append(path_tpl.format(base=base_key[len("model.") :]))
        return list(dict.fromkeys(candidates))

    def _get_proj_names(self) -> tuple[str, str, str]:
        _, gate, up, down = self.MOE_FORMATS[self._detected_format]
        return gate, up, down

    def _resolve_experts_prefix(self, base_key: str, weight_suffix: str = "weight") -> tuple[str, int]:
        gate_name, _, _ = self._get_proj_names()
        for prefix in self._get_experts_prefix_candidates(base_key):
            expert_count = 0
            while self.has_tensor(f"{prefix}.{expert_count}.{gate_name}.{weight_suffix}"):
                expert_count += 1
            if expert_count > 0:
                return prefix, expert_count
        raise ValueError(f"No experts found for keys: {self._get_experts_prefix_candidates(base_key)}")


class FP8SafeTensorLoader(_RegularMoELoader):
    """Load regular safetensors FP8 MoE expert weights."""

    def __init__(self, file_path: str, scale_suffix: str | None = None):
        super().__init__(file_path)
        self._scale_suffix = scale_suffix
        self._is_per_channel = scale_suffix == "weight_scale"
        self._detect_format("weight")
        self._detect_scale_suffix()

    def _detect_scale_suffix(self) -> None:
        if self._scale_suffix is not None:
            return
        _, gate, _, _ = self.MOE_FORMATS[self._detected_format]
        for key in list(self.tensor_file_map.keys())[:2000]:
            if f".{gate}.weight_scale_inv" in key:
                self._scale_suffix = "weight_scale_inv"
                self._is_per_channel = False
                return
            if f".{gate}.weight_scale" in key and "weight_scale_inv" not in key:
                self._scale_suffix = "weight_scale"
                scale = self.load_tensor(key)
                self._is_per_channel = scale.dim() == 1 or (scale.dim() == 2 and scale.shape[1] == 1)
                return
        self._scale_suffix = "weight_scale_inv"
        self._is_per_channel = False

    def load_experts(self, base_key: str, device: str = "cpu") -> dict[str, list[torch.Tensor]]:
        experts_prefix, expert_count = self._resolve_experts_prefix(base_key, "weight")
        gate_name, up_name, down_name = self._get_proj_names()

        def load_projection(proj_name: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            weights = []
            scales = []
            for expert_id in range(expert_count):
                weights.append(
                    self.load_tensor(f"{experts_prefix}.{expert_id}.{proj_name}.weight", device).contiguous()
                )
                scale = self.load_tensor(
                    f"{experts_prefix}.{expert_id}.{proj_name}.{self._scale_suffix}",
                    device,
                )
                if self._is_per_channel and scale.dim() == 2 and scale.shape[1] == 1:
                    scale = scale.squeeze(1)
                scales.append(scale.contiguous())
            return weights, scales

        gate, gate_scale = load_projection(gate_name)
        up, up_scale = load_projection(up_name)
        down, down_scale = load_projection(down_name)
        return {
            "gate": gate,
            "up": up,
            "down": down,
            "gate_scale": gate_scale,
            "up_scale": up_scale,
            "down_scale": down_scale,
        }


class BF16SafeTensorLoader(_RegularMoELoader):
    """Load regular safetensors BF16 MoE expert weights."""

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._detect_format("weight")

    def _resolve_packed_experts_prefix(self, base_key: str) -> str | None:
        for prefix in (
            f"{base_key}.experts",
            f"{base_key}.moe.experts",
            f"{base_key}.mlp.experts",
            f"{base_key}.feed_forward.experts",
            f"language_model.{base_key}.experts",
            f"language_model.{base_key}.moe.experts",
            f"language_model.{base_key}.mlp.experts",
        ):
            if self.has_tensor(f"{prefix}.gate_up_proj") and self.has_tensor(f"{prefix}.down_proj"):
                return prefix
        return None

    def load_experts(self, base_key: str, device: str = "cpu") -> dict[str, list[torch.Tensor] | None]:
        def load_bf16(key: str) -> torch.Tensor:
            tensor = self.load_tensor(key, device)
            if tensor.dtype != torch.bfloat16:
                tensor = tensor.to(dtype=torch.bfloat16)
            return tensor.contiguous()

        packed_prefix = self._resolve_packed_experts_prefix(base_key)
        if packed_prefix is not None:
            gate_up = load_bf16(f"{packed_prefix}.gate_up_proj")
            down_tensor = load_bf16(f"{packed_prefix}.down_proj")
            mid = gate_up.shape[1] // 2
            return {
                "gate": [gate_up[i, :mid, :].contiguous() for i in range(gate_up.shape[0])],
                "up": [gate_up[i, mid:, :].contiguous() for i in range(gate_up.shape[0])],
                "down": [down_tensor[i].contiguous() for i in range(down_tensor.shape[0])],
                "gate_scale": None,
                "up_scale": None,
                "down_scale": None,
            }

        experts_prefix, expert_count = self._resolve_experts_prefix(base_key, "weight")
        gate_name, up_name, down_name = self._get_proj_names()
        return {
            "gate": [
                load_bf16(f"{experts_prefix}.{i}.{gate_name}.weight")
                for i in range(expert_count)
            ],
            "up": [
                load_bf16(f"{experts_prefix}.{i}.{up_name}.weight")
                for i in range(expert_count)
            ],
            "down": [
                load_bf16(f"{experts_prefix}.{i}.{down_name}.weight")
                for i in range(expert_count)
            ],
            "gate_scale": None,
            "up_scale": None,
            "down_scale": None,
        }


class CompressedSafeTensorLoader(SafeTensorLoader):
    """Load RAWINT4 compressed safetensors expert weights."""

    def load_experts(self, base_key: str, device: str = "cpu") -> dict[str, list[torch.Tensor]]:
        prefixes = [
            f"{base_key}.mlp.experts",
            f"language_model.{base_key}.mlp.experts",
        ]
        experts_prefix = None
        expert_count = 0
        for prefix in prefixes:
            expert_count = 0
            while self.has_tensor(f"{prefix}.{expert_count}.up_proj.weight_packed"):
                expert_count += 1
            if expert_count:
                experts_prefix = prefix
                break
        if experts_prefix is None:
            raise ValueError(f"No RAWINT4 experts found for key {base_key}")

        def load_projection(proj_name: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            weights = []
            scales = []
            for expert_id in range(expert_count):
                weights.append(
                    self.load_tensor(
                        f"{experts_prefix}.{expert_id}.{proj_name}_proj.weight_packed",
                        device,
                    ).contiguous()
                )
                scales.append(
                    self.load_tensor(
                        f"{experts_prefix}.{expert_id}.{proj_name}_proj.weight_scale",
                        device,
                    ).contiguous()
                )
            return weights, scales

        gate, gate_scale = load_projection("gate")
        up, up_scale = load_projection("up")
        down, down_scale = load_projection("down")
        return {
            "gate": gate,
            "up": up,
            "down": down,
            "gate_scale": gate_scale,
            "up_scale": up_scale,
            "down_scale": down_scale,
        }


class GPTQSafeTensorLoader(FP8SafeTensorLoader):
    """Load symmetric GPTQ int4 safetensors expert weights."""

    def __init__(self, file_path: str):
        super().__init__(file_path, scale_suffix="scales")
        self._detect_format("qweight")
        self._verify_gptq_config(file_path)

    def _verify_gptq_config(self, file_path: str) -> None:
        config_path = os.path.join(file_path if os.path.isdir(file_path) else os.path.dirname(file_path), "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as f:
            quant_config = json.load(f).get("quantization_config", {})
        if quant_config.get("quant_method") != "gptq":
            return
        if quant_config.get("desc_act", False):
            raise NotImplementedError("GPTQ desc_act=true is not supported")
        if not quant_config.get("sym", True):
            raise NotImplementedError("GPTQ sym=false is not supported")

    def load_experts(self, base_key: str, device: str = "cpu") -> dict[str, list[torch.Tensor]]:
        experts_prefix, expert_count = self._resolve_experts_prefix(base_key, "qweight")
        gate_name, up_name, down_name = self._get_proj_names()

        def load_projection(proj_name: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            weights = []
            scales = []
            for expert_id in range(expert_count):
                weights.append(
                    self.load_tensor(f"{experts_prefix}.{expert_id}.{proj_name}.qweight", device).contiguous()
                )
                scales.append(
                    self.load_tensor(f"{experts_prefix}.{expert_id}.{proj_name}.scales", device).float().contiguous()
                )
            return weights, scales

        gate, gate_scale = load_projection(gate_name)
        up, up_scale = load_projection(up_name)
        down, down_scale = load_projection(down_name)
        return {
            "gate": gate,
            "up": up,
            "down": down,
            "gate_scale": gate_scale,
            "up_scale": up_scale,
            "down_scale": down_scale,
        }
