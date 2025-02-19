from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from aphrodite.platforms import CpuArchEnum, current_platform

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from aphrodite.common.config import ModelConfig
    from aphrodite.common.logits_processor import LogitsProcessor
    from aphrodite.common.sampling_params import GuidedDecodingParams


def maybe_backend_fallback(
        guided_params: GuidedDecodingParams) -> GuidedDecodingParams:
    # lm-format-enforce doesn't support grammar, fallback to xgrammar
    if (guided_params.backend == "lm-format-enforcer"
            and guided_params.grammar is not None):
        logger.warning(
            "lm-format-enforcer does not support grammar guided decoding. "
            "Falling back to use xgrammar instead.")
        guided_params.backend = "xgrammar"

    if guided_params.backend == "xgrammar":
        # xgrammar only has x86 wheels for linux, fallback to outlines
        if current_platform.get_cpu_architecture() is not CpuArchEnum.X86:
            logger.warning("xgrammar is only supported on x86 CPUs. "
                           "Falling back to use outlines instead.")
            guided_params.backend = "outlines"
        # xgrammar doesn't support regex, fallback to outlines
        if guided_params.regex is not None:
            logger.warning("xgrammar does not support regex guided decoding. "
                           "Falling back to use outlines instead.")
            guided_params.backend = "outlines"

    return guided_params


async def get_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig) -> LogitsProcessor | None:
    guided_params = maybe_backend_fallback(guided_params)
    # CFG grammar not supported by LMFE, so we use outlines instead
    if guided_params.backend == 'outlines':
        from aphrodite.modeling.guided_decoding.outlines_decoding import (  # noqa
            get_outlines_guided_decoding_logits_processor)
        return await get_outlines_guided_decoding_logits_processor(
            guided_params, tokenizer) # type: ignore
    if guided_params.backend == 'lm-format-enforcer':
        from aphrodite.modeling.guided_decoding.lm_format_enforcer_decoding import (  # noqa
            get_local_lm_format_enforcer_guided_decoding_logits_processor)
        return get_local_lm_format_enforcer_guided_decoding_logits_processor(
            guided_params, tokenizer) # type: ignore
    if guided_params.backend == 'xgrammar':
        from aphrodite.modeling.guided_decoding.xgrammar_decoding import (  # noqa
            get_local_xgrammar_guided_decoding_logits_processor)
        return get_local_xgrammar_guided_decoding_logits_processor(
            guided_params, tokenizer, model_config) # type: ignore

    raise ValueError(
        f"Unknown guided decoding backend '{guided_params.backend}'. "
        "Must be one of 'outlines, 'lm-format-enforcer', 'xgrammar'")


def get_local_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig) -> LogitsProcessor | None:
    guided_params = maybe_backend_fallback(guided_params)
    # CFG grammar not supported by LMFE, so we use outlines instead
    if guided_params.backend == 'outlines':
        from aphrodite.modeling.guided_decoding.outlines_decoding import (  # noqa
            get_local_outlines_guided_decoding_logits_processor)
        return get_local_outlines_guided_decoding_logits_processor(
            guided_params, tokenizer) # type: ignore
    if guided_params.backend == 'lm-format-enforcer':
        from aphrodite.modeling.guided_decoding.lm_format_enforcer_decoding import (  # noqa
            get_local_lm_format_enforcer_guided_decoding_logits_processor)
        return get_local_lm_format_enforcer_guided_decoding_logits_processor(
            guided_params, tokenizer) # type: ignore
    if guided_params.backend == 'xgrammar':
        from aphrodite.modeling.guided_decoding.xgrammar_decoding import (  # noqa
            get_local_xgrammar_guided_decoding_logits_processor)
        return get_local_xgrammar_guided_decoding_logits_processor(
            guided_params, tokenizer, model_config) # type: ignore

    raise ValueError(
        f"Unknown guided decoding backend '{guided_params.backend}'. "
        "Must be one of 'outlines, 'lm-format-enforcer', 'xgrammar'")
