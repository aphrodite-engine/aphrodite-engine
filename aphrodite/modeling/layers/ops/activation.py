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
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask)
    g_row = tl.load(g + offsets, mask=mask)

    e_f32 = tl.cast(e_row, tl.float32)

    # Compute SiLU: x * sigmoid(x)
    silu = e_f32 * (1 / (1 + tl.exp(-e_f32)))

    silu = tl.cast(silu, e_row.dtype)
    output = silu * g_row

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
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask)
    g_row = tl.load(g + offsets, mask=mask)

    e_f32 = tl.cast(e_row, tl.float32)
    
    # Compute GELU: x * 0.5 * (1 + erf(x/sqrt(2)))
    gelu = e_f32 * 0.5 * (1.0 + triton_erf(e_f32 / triton_sqrt(2.0)))

    gelu = tl.cast(gelu, e_row.dtype)
    output = gelu * g_row

    tl.store(h + offsets, output, mask=mask)


@triton.jit
def _approx_gelu_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Compute approximate GELU activation and multiply with gate:
    h = gelu(e) * g where
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask)
    g_row = tl.load(g + offsets, mask=mask)

    e_f32 = tl.cast(e_row, tl.float32)

    gelu = 0.5 * e_f32 * (1.0 + triton_tanh(
        triton_sqrt(2.0 / 3.141592653589793) * 
        (e_f32 + 0.044715 * e_f32 * e_f32 * e_f32)
    ))

    gelu = tl.cast(gelu, e_row.dtype)
    output = gelu * g_row

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
