# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from aphrodite.config import AphroditeConfig, replace
from aphrodite.config.speculative import resolve_draft_kv_cache_dtype
from aphrodite.distributed.parallel_state import get_pp_group
from aphrodite.logger import init_logger
from aphrodite.model_executor.model_loader import get_model
from aphrodite.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)

logger = init_logger(__name__)

_DCP_KV_REPLICATION_ARCHS = {"DFlashDraftModel", "Qwen3DSparkModel"}


def resolve_dflash_attention_backend(aphrodite_config: AphroditeConfig):
    """Select the DCP-capable backend for a KV-head-replicated draft."""
    speculative_config = aphrodite_config.speculative_config
    assert speculative_config is not None
    backend = speculative_config.attention_backend
    needs_head_replication = dflash_needs_dcp_kv_head_replication(aphrodite_config)
    if not needs_head_replication:
        return backend

    from aphrodite.v1.attention.backends.registry import AttentionBackendEnum

    if backend is None:
        return AttentionBackendEnum.FLASH_ATTN
    if backend != AttentionBackendEnum.FLASH_ATTN:
        raise ValueError(
            "DFlash/DSpark with DCP-replicated draft KV heads requires "
            f"attention_backend=FLASH_ATTN, got {backend.name}."
        )
    return backend


def dflash_needs_dcp_kv_head_replication(aphrodite_config: AphroditeConfig) -> bool:
    speculative_config = aphrodite_config.speculative_config
    assert speculative_config is not None
    parallel_config = aphrodite_config.parallel_config
    dcp_size = parallel_config.decode_context_parallel_size
    if dcp_size == 1:
        return False

    draft_model_config = speculative_config.draft_model_config
    total_kv_heads = draft_model_config.get_total_num_kv_heads()
    tp_size = parallel_config.tensor_parallel_size
    needs_replication = total_kv_heads >= tp_size
    if needs_replication and not (set(draft_model_config.architectures or []) & _DCP_KV_REPLICATION_ARCHS):
        raise NotImplementedError(
            "DFlash/DSpark with sharded draft KV heads under DCP requires a "
            "draft architecture that replicates its KV cache across the DCP "
            "group, but got "
            f"{draft_model_config.architectures}."
        )
    if not needs_replication:
        total_q_heads = draft_model_config.model_arch_config.total_num_attention_heads
        kv_replicated_across_dcp = (
            tp_size > total_kv_heads
            and dcp_size <= tp_size // total_kv_heads
            and (total_q_heads // total_kv_heads) % dcp_size == 0
        )
        if not kv_replicated_across_dcp:
            raise NotImplementedError(
                "DFlash/DSpark under DCP requires draft KV heads to be fully "
                "sharded or replicated across the DCP group; got "
                f"q_heads={total_q_heads}, kv_heads={total_kv_heads}, "
                f"tp={tp_size}, dcp={dcp_size}."
            )
    return needs_replication


def resolve_dflash_cache_dtype(aphrodite_config: AphroditeConfig):
    """Use a non-quantized cache when DCP replicates draft KV heads."""
    if dflash_needs_dcp_kv_head_replication(aphrodite_config):
        logger.warning_once(
            "DFlash/DSpark with DCP requires replicated draft KV heads; "
            "using the draft model dtype for its KV cache. The target KV "
            "cache dtype is unchanged."
        )
        return "auto"
    speculative_config = aphrodite_config.speculative_config
    assert speculative_config is not None
    return resolve_draft_kv_cache_dtype(
        speculative_config,
        aphrodite_config.cache_config.cache_dtype,
    )


def load_dflash_model(target_model: nn.Module, aphrodite_config: AphroditeConfig) -> nn.Module:
    from aphrodite.compilation.backends import set_model_tag
    from aphrodite.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

    speculative_config = aphrodite_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_aphrodite_config = replace(
        aphrodite_config,
        attention_config=replace(
            aphrodite_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=resolve_dflash_attention_backend(aphrodite_config),
        ),
        cache_config=replace(
            aphrodite_config.cache_config,
            cache_dtype=resolve_dflash_cache_dtype(aphrodite_config),
        ),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(aphrodite_config=draft_aphrodite_config, model_config=draft_model_config)

    target_language_model = (
        target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
    )
    target_inner = target_language_model.model
    draft_inner = dflash_model.model

    # Skip embedding sharing under PP — each rank owns its own embedding.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(target_inner, "embedding", None)
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(dflash_model, "has_own_embed_tokens", draft_embed, target_embed):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
