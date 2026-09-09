# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .registry import OmniModelRegistry  # noqa: F401

# Model classes are lazily loaded via OmniModelRegistry.
# Do NOT eagerly import model classes here — it triggers heavy transitive
# imports (CUDA, pynvml, bitsandbytes, etc.) that crash in subprocess
# environments used by Sonar's model inspection.

__all__ = [
    "OmniModelRegistry",
]
