# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Any

import regex as re
import torch

from aphrodite.config import (
    AphroditeConfig,
    CompilationConfig,
    ModelConfig,
    set_current_aphrodite_config,
)
from aphrodite.model_executor.layers.quantization import get_quantization_config
from aphrodite.model_executor.model_loader.default_loader import DefaultModelLoader
from aphrodite.platforms import current_platform


def _limit_num_hidden_layers(model: torch.nn.Module, num_hidden_layers: int | None) -> None:
    if num_hidden_layers is None:
        return

    original_load_weights = model.load_weights

    def should_load_weight(name: str) -> bool:
        for prefix in ("model.layers.", "layers."):
            if name.startswith(prefix):
                layer_idx = int(name.removeprefix(prefix).split(".", 1)[0])
                return layer_idx < num_hidden_layers
        return True

    def load_weights(weights):
        weights = ((name, weight) for name, weight in weights if should_load_weight(name))
        return original_load_weights(weights)

    model.load_weights = load_weights


def load_model_without_aphrodite_runner(
    model_path: str,
    *,
    dtype: str | torch.dtype = "bfloat16",
    quantization: str | None = None,
    model_config_kwargs: dict[str, Any] | None = None,
    aphrodite_config_kwargs: dict[str, Any] | None = None,
    model_loader_cls: type = DefaultModelLoader,
) -> tuple[torch.nn.Module, AphroditeConfig]:
    """Instantiate a model, load weights, and process them for inference."""
    model_config = ModelConfig(
        model=model_path,
        dtype=dtype,
        quantization=quantization,
        **(model_config_kwargs or {}),
    )
    aphrodite_config_args = dict(aphrodite_config_kwargs or {})
    aphrodite_config_args.setdefault("compilation_config", CompilationConfig(mode=0))
    aphrodite_config = AphroditeConfig(model_config=model_config, **aphrodite_config_args)
    hf_overrides = (model_config_kwargs or {}).get("hf_overrides") or {}
    num_hidden_layers = hf_overrides.get("num_hidden_layers")

    with set_current_aphrodite_config(aphrodite_config):
        model_loader = model_loader_cls(aphrodite_config.load_config)
        if num_hidden_layers is not None:
            original_load_weights = model_loader.load_weights

            def load_weights(model, model_config):
                _limit_num_hidden_layers(model, num_hidden_layers)
                original_load_weights(model, model_config)

            model_loader.load_weights = load_weights
        model = model_loader.load_model(aphrodite_config, model_config)

    return model, aphrodite_config


def is_quant_method_supported(quant_method: str) -> bool:
    # Currently, quantization tests only run on supported GPU platforms.
    if not (current_platform.is_cuda() or current_platform.is_rocm() or current_platform.is_xpu()):
        return False

    try:
        current_platform.verify_quantization(quant_method)
    except ValueError:
        return False

    if current_platform.is_xpu():
        return True

    capability = current_platform.get_device_capability()
    assert capability is not None

    min_capability = get_quantization_config(quant_method).get_min_capability()

    return capability.to_int() >= min_capability


def _test_online_quant_peak_mem_impl(
    quantization_arg_value,
    aphrodite_runner,
    caplog_mp_spawn,
    monkeypatch,
) -> None:
    # Note: `allenai/OLMoE-1B-7B-0125-Instruct` was selected because:
    # 1. it covers both Linear and MoE paths
    # 2. it is already used by other tests in CI, so adding it here
    #    does not increase disk space for CI runners
    # I really wanted to use `ibm-granite/granite-3.0-1b-a400m-base`
    # which I think is the smallest MoE model in Aphrodite (2.5 GiB bf16,
    # 1.3 GiB fp8), but could not as adding one more model makes CI
    # run out of disk space.
    model_name = "allenai/OLMoE-1B-7B-0125-Instruct"

    # Force spawn to ensure caplog_mp_spawn works consistently
    # (it relies on APHRODITE_LOGGING_CONFIG_PATH which spawn reads but fork ignores)
    monkeypatch.setenv("APHRODITE_WORKER_MULTIPROC_METHOD", "spawn")

    with (
        caplog_mp_spawn(logging.DEBUG) as log_holder,
        aphrodite_runner(
            model_name,
            quantization=quantization_arg_value,
            enforce_eager=True,
        ) as llm,
    ):
        outputs = llm.generate_greedy(["The future of AI is"], max_tokens=4)
        print(outputs[0][1])

    log_text = log_holder.text

    # Parse memory usage from captured logs
    model_memory_gib = None
    peak_memory_gib = None
    for line in log_text.splitlines():
        if model_memory_gib is None:
            match = re.search(r"Model loading took ([\d.]+) GiB memory", line)
            if match:
                model_memory_gib = float(match.group(1))
        if peak_memory_gib is None:
            match = re.search(r"Peak GPU memory after loading weights: ([\d.]+) GiB", line)
            if match:
                peak_memory_gib = float(match.group(1))

    assert model_memory_gib is not None, "Could not find model loading memory log"
    assert peak_memory_gib is not None, "Could not find peak memory log"
    print(f"GPU memory used after loading weights: {model_memory_gib} GiB")
    print(f"Peak GPU memory usage while loading weights: {peak_memory_gib} GiB")

    expected_model_memory_gib = 6.7

    # for allenai/OLMoE-1B-7B-0125-Instruct the number we see today is 9.06
    # GiB on CUDA, which is 1.36x above model_memory_gib. A slightly higher
    # number is expected as when we load and quantize weights in a streaming
    # fashion we need to have individual weights in bf16 + fp8 alive at the
    # same time.
    expected_peak_memory_gib = expected_model_memory_gib * 1.4

    assert model_memory_gib < expected_model_memory_gib, f"{model_memory_gib=} higher than {expected_model_memory_gib}"
    assert peak_memory_gib < expected_peak_memory_gib, f"{peak_memory_gib=} higher than {expected_peak_memory_gib}"
