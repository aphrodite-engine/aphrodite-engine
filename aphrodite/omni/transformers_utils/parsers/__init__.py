# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Custom Sonar config parsers for sonar-omni."""

from __future__ import annotations

import importlib

_CLASS_TO_MODULE: dict[str, str] = {
    "VoxtralTTSConfigParser": "aphrodite.omni.transformers_utils.parsers.voxtral_tts",
}

__all__ = ["VoxtralTTSConfigParser"]


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'aphrodite.omni.transformers_utils.parsers' has no attribute {name!r}")


def __dir__():
    return sorted(list(__all__))


# Eagerly import parser modules so their registry side-effects run as soon as
# `aphrodite.omni.transformers_utils.parsers` is imported.
from aphrodite.omni.transformers_utils.parsers import voxtral_tts as _voxtral_tts  # noqa: F401, E402
