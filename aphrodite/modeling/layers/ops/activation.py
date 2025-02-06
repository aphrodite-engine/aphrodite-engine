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

# The following code is adapted from:
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
else:
    triton_tanh = tl.math.tanh


@triton.jit
def _fg_kernel(e, g, h, n_elements,
               BLOCK_SIZE: tl.constexpr,
               COMPUTE_DTYPE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask, other=0)
    
    # Cast to the compile-time constant type passed in.
    e_math = tl.cast(e_row, COMPUTE_DTYPE)
    sig = tl.sigmoid(e_math)
    # Cast the sigmoid output back to the original type.
    sig = tl.cast(sig, e_row.dtype)
    f_row = e_row * sig

    tl.store(h + offsets, f_row, mask=mask)
pass


def swiglu_fg_kernel(e, g):
    # If e is 2D (num_tokens x d), add a dummy batch dimension.
    squeeze = False
    if e.dim() == 2:
        e = e.unsqueeze(0)  # Now shape becomes (1, num_tokens, d)
        g = g.unsqueeze(0)
        squeeze = True

    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype=e.dtype, device=e.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Decide on the compute dtype:
    # For FP16, we want to compute in FP32; for all other dtypes,
    # we use the same type.
    compute_dtype = tl.float32 if e.dtype == torch.float16 else e.dtype

    with torch.cuda.device(e.device):
         _fg_kernel[grid](e, g, h, n_elements,
                          BLOCK_SIZE=1024,
                          COMPUTE_DTYPE=compute_dtype)
    if squeeze:
         return h.squeeze(0)
    return h
pass


@triton.jit
def _DWf_DW_dfg_kernel(DW, e, g, n_elements, BLOCK_SIZE : tl.constexpr):
    """
    e = e.float()
    se = 1.0 / (1.0 + torch.exp(-e))
    f = (se * e).to(dtype)
    h = f * g
    df = DW * f
    dg = DW * g
    de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    """
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask = mask, other = 0)#.to(tl.float32)
    e_row  = tl.load(e  + offsets, mask = mask, other = 0).to(tl.float32)
    g_row  = tl.load(g  + offsets, mask = mask, other = 0)#.to(tl.float32)

    # e = e.float()
    # se = 1.0 / (1.0 + torch.exp(-e))
    e_math = tl.cast(e_row, tl.float32) if e.dtype == tl.float16 else e_row
    se_row = tl.sigmoid(e_math) # 1.0 / (1.0 + tl.exp(-e_row))
    # f = (se * e).to(dtype)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)
    # h = f * g
    h_row  =  f_row * g_row
    # df = DW * f
    df_row = DW_row * f_row
    # dg = DW * g
    dg_row = DW_row * g_row
    # de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    # Store derivatives in buffers
    tl.store(DW + offsets, h_row,  mask = mask) # h  = f * g
    tl.store(e  + offsets, df_row, mask = mask) # df = DW * f
    tl.store(g  + offsets, de_row, mask = mask) # de
pass


def swiglu_DWf_DW_dfg_kernel(DW, e, g):
    batch_seq_len, hd = e.shape
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _DWf_DW_dfg_kernel[grid](DW, e, g, n_elements, BLOCK_SIZE = 1024,)
    return DW, e, g
pass

@triton.jit
def _exact_forward_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # f = 1/2 * e * (1 + erf(1/sqrt(2) * e))
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    f_row = 0.5 * e_row * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)
pass


def geglu_exact_forward_kernel(gate, up):
    # If gate is 2D, add a dummy batch dimension.
    squeeze = False
    if gate.dim() == 2:
        gate = gate.unsqueeze(0)
        up = up.unsqueeze(0)
        squeeze = True

    batch, seq_len, hd = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hd), dtype=gate.dtype,
                      device=gate.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch.cuda.device(gate.device):
        _exact_forward_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE=1024)
    if squeeze:
        return out.squeeze(0)
    return out
pass


@triton.jit
def _approx_forward_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # f = 1/2 * e * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3 ) ))
    # f = 1/2 * e * (1 + tanh( sqrt(2/pi) * x * (1 + 0.044715 * x^2 ) ))
    # h = f * up
    s = 0.7978845608028654 # math.sqrt(2 / math.pi)
    
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    e_math = tl.cast(e_row, tl.float32) if e.dtype == tl.float16 else e_row
    f_row = 0.5 * e_math * (
        triton_tanh(s * e_math * (1.0 + 0.044715 * e_math * e_math)) \
        + 1.0
    )
    f_row = f_row.to(g_row.dtype) # Exact copy from HF
    h_row = f_row * g_row

    # Store h
    tl.store(h + offsets, h_row, mask = mask)
pass


def geglu_approx_forward_kernel(gate, up):
    # If gate is 2D, add a dummy batch dimension.
    squeeze = False
    if gate.dim() == 2:
        gate = gate.unsqueeze(0)
        up = up.unsqueeze(0)
        squeeze = True

    batch, seq_len, hd = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hd),
                      dtype=gate.dtype, device=gate.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch.cuda.device(gate.device):
        _approx_forward_kernel[grid](
            gate, up, out, n_elements, BLOCK_SIZE=1024)
    if squeeze:
        return out.squeeze(0)
    return out
pass
