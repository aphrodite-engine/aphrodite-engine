# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
import torch
from torch import Tensor

from aphrodite import ir
from aphrodite.platforms import current_platform

current_platform.import_kernels()

CUDA_ALIKE = current_platform.is_cuda_alike()
"""Most kernels in this file are supported on all CUDA-alike platforms."""

rms_no_var_size = (
    lambda x, weight, epsilon, variance_size=None: variance_size is None
    and (weight is None or weight.dtype == x.dtype)
)
"""Aphrodite kernel requires no variance_size override and matching input/weight dtype."""


@ir.ops.rms_norm.register_impl(
    "aphrodite_c", supports_args=rms_no_var_size, supported=CUDA_ALIKE
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    if weight is None:
        # Kernel requires weight tensor, pass ones
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    assert variance_size is None
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output
