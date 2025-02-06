# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved
# Copyright 2024-present Andrej Karpathy & the llm.c team. All rights reserved
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
# https://github.com/unslothai/unsloth/blob/038e6d4c8d40207a87297ab3aaf787c19b1006d1/unsloth/kernels/rms_layernorm.py

import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE : int = 65536
next_power_of_2 = triton.next_power_of_2


# Calculate the optimal block size and number of warps for the layernorm kernel
# borrowed from https://github.com/unslothai/unsloth/blob/038e6d4c8d40207a87297ab3aaf787c19b1006d1/unsloth/kernels/utils.py#L49-L59
def calculate_settings(n : int) -> tuple[int, int]:
    BLOCK_SIZE : int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >=  8192:
        num_warps = 16
    elif BLOCK_SIZE >=  2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps
pass


@triton.jit
def _rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr
):
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0)#.to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype) # Exact copy from HF
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask = mask)
pass


@triton.jit
def _gemma_rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr,
):
    # Copies https://github.com/google-deepmind/gemma/blob/main/gemma/layers.py#L31
    # and https://github.com/keras-team/keras-nlp/blob/v0.8.2/keras_nlp/models/gemma/rms_normalization.py#L33
    # exactly. Essentially all in float32!
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)

    tl.store(Y + col_offsets, output, mask = mask)
pass


class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X : torch.Tensor,
        W : torch.Tensor,
        eps : float,
        gemma : bool = False,
    ):
        shape = X.shape
        dim : int = shape[-1]
        X = X.view(-1, dim)
        n_rows : int
        n_cols : int
        n_rows, n_cols = X.shape
        BLOCK_SIZE : int
        num_warps  : int
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = X.device)
        r = torch.empty(n_rows, dtype = torch.float32, device = X.device)

        fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
        with torch.cuda.device(X.device):
            fx[(n_rows,)](
                Y, Y.stride(0),
                X, X.stride(0),
                W, W.stride(0),
                r, r.stride(0),
                n_cols, eps,
                BLOCK_SIZE = BLOCK_SIZE,
                num_warps  = num_warps,
            )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.GEMMA = gemma
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)
    pass
pass


# [TODO] Unsure why RMS Layernorm is not torch.compiling properly
@torch.compiler.disable
def fast_rms_layernorm(layernorm, X : torch.Tensor, gemma : bool = False):
    W : torch.Tensor = layernorm.weight
    eps : float = layernorm.variance_epsilon if \
        hasattr(layernorm, "variance_epsilon") \
        else layernorm.eps
    out = Fast_RMS_Layernorm.apply(X, W, eps, gemma)
    return out
pass

