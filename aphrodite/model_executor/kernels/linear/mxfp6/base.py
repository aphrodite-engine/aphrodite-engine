# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class Mxfp6LinearLayerConfig:
    weight_format: Literal["e2m3", "e3m2"] = "e2m3"
    activation_format: Literal["mxfp8", "mxfp6_e2m3", "mxfp6_e3m2"] = "mxfp8"


class Mxfp6LinearKernel(ABC):
    def __init__(self, config: Mxfp6LinearLayerConfig) -> None:
        supported, reason = self.is_supported()
        if not supported:
            raise ValueError(reason)
        self.config = config

    @classmethod
    @abstractmethod
    def is_supported(cls) -> tuple[bool, str | None]: ...

    @classmethod
    @abstractmethod
    def can_implement_shape(cls, n: int, k: int) -> tuple[bool, str | None]: ...

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
