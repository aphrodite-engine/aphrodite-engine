# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings


def __getattr__(name: str):
    # Keep until lm-eval is updated
    if name == "get_tokenizer":
        from aphrodite.tokenizers import get_tokenizer

        warnings.warn(
            "`aphrodite.transformers_utils.tokenizer.get_tokenizer` "
            "has been moved to `aphrodite.tokenizers.get_tokenizer`. "
            "The old name will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

        return get_tokenizer
    if name == "cached_tokenizer_from_config":
        from aphrodite.tokenizers import cached_tokenizer_from_config

        warnings.warn(
            "`aphrodite.transformers_utils.tokenizer.cached_tokenizer_from_config` "
            "has been moved to `aphrodite.tokenizers.cached_tokenizer_from_config`. "
            "The old name will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

        return cached_tokenizer_from_config

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
