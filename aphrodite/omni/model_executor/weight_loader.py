# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-specific weight filtering on top of Sonar's core loader."""

from aphrodite.model_executor.models.utils import AutoWeightsLoader as CoreWeightsLoader


class AutoWeightsLoader(CoreWeightsLoader):
    def __init__(self, module, *, skip_prefixes=None, skip_substrs=None, **kwargs):
        super().__init__(module, **kwargs)
        self.skip_prefixes = tuple(skip_prefixes or ())
        self.skip_substrs = tuple(skip_substrs or ())

    def _can_skip(self, qualname: str) -> bool:
        return (
            qualname.startswith(self.skip_prefixes)
            or any(part in qualname for part in self.skip_substrs)
            or super()._can_skip(qualname)
        )
