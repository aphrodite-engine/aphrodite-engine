# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up FA4 CuTeDSL kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aphrodite.v1.attention.backends.mla.prefill import get_mla_prefill_backend

if TYPE_CHECKING:
    from aphrodite.v1.worker.gpu_worker import Worker


def _warm_fa4_mla_prefill(worker: Worker) -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    aphrodite_config = runner.aphrodite_config
    if not aphrodite_config.model_config.use_mla:
        return

    try:
        backend_cls = get_mla_prefill_backend(aphrodite_config)
    except ValueError:
        # fall back to top-k MQA prefill path.
        return
    if backend_cls.get_name() != "FLASH_ATTN":
        return

    from aphrodite.v1.attention.backends.mla.prefill import flash_attn

    flash_attn.FA4_MLA_PREFILL_KERNEL.warmup(aphrodite_config)


def _warm_inkling_fa4_rel_attention(worker: Worker) -> None:
    from aphrodite.models.inkling.configs import InklingMMConfig, InklingModelConfig
    from aphrodite.models.inkling.nvidia.ops.fa4_rel_attention import (
        INKLING_FA4_REL_ATTENTION_KERNEL,
    )

    aphrodite_config = worker.aphrodite_config
    hf_config = aphrodite_config.model_config.hf_config
    if not isinstance(hf_config, (InklingMMConfig, InklingModelConfig)):
        return

    INKLING_FA4_REL_ATTENTION_KERNEL.warmup(aphrodite_config)


def fa4_cutedsl_warmup(worker: Worker) -> None:
    _warm_fa4_mla_prefill(worker)
    _warm_inkling_fa4_rel_attention(worker)
