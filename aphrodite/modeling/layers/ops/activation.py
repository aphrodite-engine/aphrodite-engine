# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The following code is loosely based on:
# https://github.com/unslothai/unsloth/blob/038e6d4c8d40207a87297ab3aaf787c19b1006d1/unsloth/kernels/swiglu.py
# and
# https://github.com/unslothai/unsloth/blob/038e6d4c8d40207a87297ab3aaf787c19b1006d1/unsloth/kernels/geglu.py

import torch
import triton
import triton.language as tl
from packaging.version import Version

if Version(triton.__version__) >= Version("3.0.0"):
    from triton.language.extra import libdevice
    triton_tanh = libdevice.tanh
    triton_erf = libdevice.erf
    triton_sqrt = libdevice.sqrt
else:
    triton_tanh = tl.math.tanh
    triton_erf = tl.math.erf
    triton_sqrt = tl.math.sqrt

@triton.jit
def _fg_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Compute SiLU activation and multiply with gate:
    h = silu(e) * g where silu(x) = x * sigmoid(x)
    
    Differences from unsloth:
    1. Support for 2D inputs
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)
    output = f_row * g_row

    tl.store(h + offsets, output, mask=mask)


def swiglu_fg_kernel(e, g):
    # If e is 2D (num_tokens x d), add a dummy batch dimension
    squeeze = False
    if e.dim() == 2:
        e = e.unsqueeze(0)
        g = g.unsqueeze(0)
        squeeze = True

    batch, num_tokens, d = e.shape
    n_elements = batch * num_tokens * d
    h = torch.empty((batch, num_tokens, d), dtype=e.dtype, device=e.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    with torch.cuda.device(e.device):
        _fg_kernel[grid](
            e.reshape(-1), g.reshape(-1), h.reshape(-1),
            n_elements, BLOCK_SIZE=1024
        )

    if squeeze:
        return h.squeeze(0)
    return h


@triton.jit
def _exact_gelu_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Compute exact GELU activation and multiply with gate:
    h = gelu(e) * g where gelu(x) = x * 0.5 * (1 + erf(x/sqrt(2)))
    
    Differences from unsloth:
    1. Support for 2D inputs
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    f_row = 0.5 * e_row * (triton_erf(triton_sqrt(0.5) * e_row) + 1.0)
    f_row = f_row.to(g_row.dtype)
    output = f_row * g_row

    tl.store(h + offsets, output, mask=mask)


@triton.jit
def _approx_gelu_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Compute approximate GELU activation and multiply with gate:
    h = gelu(e) * g where
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    
    Differences from unsloth:
    1. Support for 2D inputs
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    s = 0.7978845608028654  # sqrt(2/pi)
    f_row = 0.5 * e_row * (
        triton_tanh(s * e_row * (1.0 + 0.044715 * e_row * e_row)) + 1.0
    )
    f_row = f_row.to(g_row.dtype)
    output = f_row * g_row

    tl.store(h + offsets, output, mask=mask)


def geglu_exact_forward_kernel(e, g):
    # If e is 2D (num_tokens x d), add a dummy batch dimension
    squeeze = False
    if e.dim() == 2:
        e = e.unsqueeze(0)
        g = g.unsqueeze(0)
        squeeze = True

    batch, num_tokens, d = e.shape
    n_elements = batch * num_tokens * d
    h = torch.empty((batch, num_tokens, d), dtype=e.dtype, device=e.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    with torch.cuda.device(e.device):
        _exact_gelu_kernel[grid](
            e.reshape(-1), g.reshape(-1), h.reshape(-1),
            n_elements, BLOCK_SIZE=1024
        )

    if squeeze:
        return h.squeeze(0)
    return h


def geglu_approx_forward_kernel(e, g):
    # If e is 2D (num_tokens x d), add a dummy batch dimension
    squeeze = False
    if e.dim() == 2:
        e = e.unsqueeze(0)
        g = g.unsqueeze(0)
        squeeze = True

    batch, num_tokens, d = e.shape
    n_elements = batch * num_tokens * d
    h = torch.empty((batch, num_tokens, d), dtype=e.dtype, device=e.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    with torch.cuda.device(e.device):
        _approx_gelu_kernel[grid](
            e.reshape(-1), g.reshape(-1), h.reshape(-1),
            n_elements, BLOCK_SIZE=1024
        )

    if squeeze:
        return h.squeeze(0)
    return h


@triton.jit
def _gelu_new_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Compute new GELU activation (same as approximate GELU):
    gelu_new(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    x3 = x * x * x
    c = 0.79788456  # sqrt(2/pi)
    t = triton_tanh(c * (x + 0.044715 * x3))
    output = 0.5 * x * (1.0 + t)

    tl.store(output_ptr + offsets, output, mask=mask)


def gelu_new_kernel(x: torch.Tensor) -> torch.Tensor:
    """Triton kernel wrapper for new GELU activation."""
    # If x is 2D (num_tokens x d), add a dummy batch dimension
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze = True

    batch, num_tokens, d = x.shape
    n_elements = batch * num_tokens * d
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    with torch.cuda.device(x.device):
        _gelu_new_kernel[grid](
            x.reshape(-1), output.reshape(-1),
            n_elements, BLOCK_SIZE=1024
        )

    if squeeze:
        return output.squeeze(0)
    return output


@triton.jit
def _fast_gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Compute fast GELU activation:
    gelu_fast(x) = 0.5 * x * (1 + tanh(0.7978845608 * x * (1 + 0.044715 * x^2)))
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    c = 0.79788456  # sqrt(2/pi)
    inner = x * (1.0 + 0.044715 * x * x)
    t = triton_tanh(c * inner)
    output = 0.5 * x * (1.0 + t)

    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _quick_gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Compute quick GELU activation:
    quick_gelu(x) = x * sigmoid(1.702 * x)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    # Compute x * sigmoid(1.702 * x)
    output = x * (1.0 / (1.0 + tl.exp(-1.702 * x)))

    tl.store(output_ptr + offsets, output, mask=mask)


def fast_gelu_kernel(x: torch.Tensor) -> torch.Tensor:
    """Triton kernel wrapper for fast GELU activation."""
    # If x is 2D (num_tokens x d), add a dummy batch dimension
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze = True

    batch, num_tokens, d = x.shape
    n_elements = batch * num_tokens * d
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    with torch.cuda.device(x.device):
        _fast_gelu_kernel[grid](
            x.reshape(-1), output.reshape(-1),
            n_elements, BLOCK_SIZE=1024
        )

    if squeeze:
        return output.squeeze(0)
    return output


def quick_gelu_kernel(x: torch.Tensor) -> torch.Tensor:
    """Triton kernel wrapper for quick GELU activation."""
    # If x is 2D (num_tokens x d), add a dummy batch dimension
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze = True

    batch, num_tokens, d = x.shape
    n_elements = batch * num_tokens * d
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    with torch.cuda.device(x.device):
        _quick_gelu_kernel[grid](
            x.reshape(-1), output.reshape(-1),
            n_elements, BLOCK_SIZE=1024
        )

    if squeeze:
        return output.squeeze(0)
    return output
