# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from aphrodite.config import AphroditeConfig, ModelConfig, replace
from aphrodite.logger import init_logger
from aphrodite.model_executor.model_loader import get_model
from aphrodite.v1.attention.backends.registry import AttentionBackendEnum
from aphrodite.v1.worker.gpu.spec_decode.dflash.utils import (
    resolve_dflash_attention_backend,
    resolve_dflash_cache_dtype,
)
from aphrodite.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)

logger = init_logger(__name__)


def _resolve_dspark_attention_backend(
    draft_model_config: ModelConfig,
    draft_backend: AttentionBackendEnum | None,
    target_backend: AttentionBackendEnum | None,
) -> AttentionBackendEnum | None:
    if draft_backend is not None:
        return draft_backend
    # DeepSeek-V4 draft layers share the target's KV-cache layout. Other
    # DSpark architectures may use a different attention kind.
    if draft_model_config.hf_config.model_type == "deepseek_v4":
        if target_backend is not None:
            logger.info_once(
                "Using the target model's %s attention backend for the DeepSeek-V4 DSpark drafter.",
                target_backend.name,
            )
        return target_backend
    return None


def load_dspark_model(target_model: nn.Module, aphrodite_config: AphroditeConfig) -> nn.Module:
    speculative_config = aphrodite_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from aphrodite.compilation.backends import set_model_tag
    from aphrodite.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from aphrodite.model_executor.models.utils import get_draft_quant_config
    from aphrodite.v1.worker.gpu.spec_decode.eagle.utils import (
        maybe_share_target_embed,
    )

    draft_attention_backend = _resolve_dspark_attention_backend(
        draft_model_config,
        speculative_config.attention_backend,
        aphrodite_config.attention_config.backend,
    )

    draft_aphrodite_config = replace(
        aphrodite_config,
        parallel_config=speculative_config.draft_parallel_config,
        attention_config=replace(
            aphrodite_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=resolve_dflash_attention_backend(aphrodite_config) or draft_attention_backend,
        ),
        cache_config=replace(
            aphrodite_config.cache_config,
            cache_dtype=resolve_dflash_cache_dtype(aphrodite_config),
        ),
    )
    # AphroditeConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_aphrodite_config.quant_config = get_draft_quant_config(aphrodite_config)

    with set_model_tag("dspark_head"):
        draft_model = get_model(aphrodite_config=draft_aphrodite_config, model_config=draft_model_config)

    target_language_model = (
        target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model
    target_vocab_size = aphrodite_config.model_config.get_vocab_size()

    if draft_model_config.get_vocab_size() <= target_vocab_size:
        maybe_share_target_embed(draft_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    draft_output_vocab_size = (
        getattr(draft_model_config.hf_config, "draft_vocab_size", None) or draft_model_config.get_vocab_size()
    )
    if (
        target_lm_head is not None
        and draft_output_vocab_size == target_vocab_size
        and _should_share(draft_model, "has_own_lm_head", draft_lm_head, target_lm_head)
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
