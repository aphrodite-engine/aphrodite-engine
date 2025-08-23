# Adapted from https://github.com/stackav-oss/conch/blob/e4ac60c47ebcaf76a5fb38982d6a8bbc169f65f6/conch/kernels/activation  # noqa: E501

from typing import Final

import torch

from aphrodite.triton_utils import libdevice, tl, triton

_M_2_SQRTPI: Final = 1.12837916709551257390
_M_SQRT2: Final = 1.41421356237309504880


@triton.jit  # type: ignore[misc]
def _gelu_tanh_and_mul_kernel(
    out_ptr: tl.tensor,  # [..., d]
    out_stride: int,
    in_ptr: tl.tensor,  # [..., 2 * d]
    in_stride: int,
    d: int,
    cxpr_block_size: tl.constexpr,
    cxpr_beta: tl.constexpr,
    cxpr_kappa: tl.constexpr,
) -> None:
    """Apply tanh-approximated GeLU and multiply.

    This step is part of the GeGLU activation used in the GEMMA model.
    https://arxiv.org/abs/2002.05202
    https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/image2_l7UnOuC.original.png

    Ref implementation summary:
    - One block per token
    - Threads work in parallel to perform point-wise GeLU and multiply
    - Threads will loop until all d elements are processed

    Args:
        out_ptr: Storage for results; expected dimensions
            (num_tokens, projection_dim).
        out_stride: step size between output rows.
        in_ptr: Adjoined projections to be scaled
            [ ..., [gate_proj | up_proj]]; ([opt-batch],
             num_tokens, projection_dim * 2).
        in_stride: step size between input rows.
        d: dimensionality of projection space.
        cxpr_block_size: working block size.
        cxpr_beta: BETA constant.
        cxpr_kappa: KAPPA constant.
    """
    token_idx = tl.program_id(axis=0)
    in_block_start = in_ptr + (token_idx * in_stride)
    local_offset = tl.arange(0, cxpr_block_size)
    out_block_start = out_ptr + (token_idx * out_stride)
    for _ in tl.range(0, d, cxpr_block_size):
        mask = local_offset < d
        x = tl.load(in_block_start + local_offset, mask=mask)
        y = tl.load(in_block_start + d + local_offset, mask=mask)
        x_cubed = x * x * x
        inner = cxpr_beta * (x + cxpr_kappa * x_cubed)
        gelu_act = 0.5 * x * (1 + libdevice.tanh(inner))
        scaled = gelu_act * y
        tl.store(out_block_start + local_offset, scaled, mask=mask)
        local_offset += cxpr_block_size


def gelu_tanh_and_mul_launcher(
    output: torch.Tensor,
    projections: torch.Tensor,
) -> None:
    """gelu_tanh_and_mul launcher.

    Args:
        output: Storage for results; expected dimensions
            (num_tokens, projection_dim)
        projections: Adjoined projections to be scaled;
            (num_tokens, projection_dim * 2)
    """
    d = projections.shape[-1] // 2
    num_tokens = projections.numel() // (2 * d)
    beta = _M_SQRT2 * _M_2_SQRTPI * 0.5
    kappa = 0.044715
    block_size = min(256, triton.next_power_of_2(d))

    # Note about using negative indices in shape and stride
    # The commonly (?) accepted organization of data appears to be:
    # - batch, token, other higher level dimensions in the leading positions
    # - data-specific dimensions in the lower positions
    # For example: [batch_size, num_token, embedding_size]
    # Some higher level dimensions may or may not exist.  For example, for
    # unbatched data, there wouldn't be a batch_size dimension.  On the other
    # hand, the lower-level dimensions will always exist.  This is why it makes
    # some sense to count backwards when trying to access dimension
    # information.

    _gelu_tanh_and_mul_kernel[(num_tokens,)](
        output,
        output.stride(-2),
        projections,
        projections.stride(-2),
        d,
        cxpr_block_size=block_size,
        cxpr_beta=beta,
        cxpr_kappa=kappa,
    )

@triton.jit  # type: ignore[misc]
def _silu_and_mul_kernel(  # noqa: PLR0913
    # Pointers to tensors
    output_ptr: tl.tensor,  # [..., d]
    x_ptr: tl.tensor,  # [..., 2 * d]
    # Strides of relevant tensors
    output_stride: int,
    input_stride: int,
    # Scalars
    d: int,
    # Constexprs
    cxpr_block_size: tl.constexpr,
) -> None:
    """Implementation of SiLU and multiply kernel.

    Args:
        output_ptr: Pointer to output tensor, shape: (num_tokens, d) or
                    (batch_size, sequence_length, d).
        x_ptr: Pointer to input tensor, shape: (num_tokens, 2 * d) or
               (batch_size, sequence_length, 2 * d).
        output_stride: Stride of output tensor between elements in the last
                       dimension.
        input_stride: Stride of input tensor between elements in the last
                      dimension.
        d: Size of last dimension of output.
        cxpr_block_size: Number of elements to process at once.
    """
    token_index = tl.program_id(0)

    x_token_offset = token_index * input_stride
    y_token_offset = x_token_offset + d
    output_token_offset = token_index * output_stride

    block_offsets = tl.arange(0, cxpr_block_size)

    for _ in tl.range(0, d, cxpr_block_size):
        mask = block_offsets < d

        # For parity with vLLM, compute `x * sigmoid(x)` in fp32
        x = tl.load(x_ptr + x_token_offset + block_offsets,
                    mask=mask, other=0.0).to(tl.float32)
        y = tl.load(x_ptr + y_token_offset + block_offsets,
                    mask=mask, other=0.0)

        silu = (x * tl.sigmoid(x)).to(x_ptr.dtype.element_ty)
        silu *= y

        tl.store(output_ptr + output_token_offset + block_offsets,
                 silu, mask=mask)

        block_offsets += cxpr_block_size


def silu_and_mul_launcher(
    output: torch.Tensor,
    x: torch.Tensor,
) -> None:
    """Launch silu_and_mul kernel.

    Args:
        x: Input tensor, of shape (num_tokens, 2 * d) or
           (batch_size, seq_len, 2 * d).
        output: Output tensor, of shape (num_tokens, d) or
                (batch_size, seq_len, d).
    """
    d = x.size(-1) // 2
    cxpr_block_size = min(1024, triton.next_power_of_2(d))

    num_tokens = x.numel() // x.size(-1)

    assert output.shape == x.shape[:-1] + (d,), (
        "Output shape must match input shape with last dimension halved!")
    assert x.is_contiguous()  # noqa: S101
    assert output.is_contiguous()  # noqa: S101

    # Parallelize over the number of tokens
    grid = (num_tokens,)

    # Launch kernel
    _silu_and_mul_kernel[grid](
        # Tensors
        output_ptr=output,
        x_ptr=x,
        # Strides of relevant tensors
        output_stride=output.stride(-2),
        input_stride=x.stride(-2),
        # Scalars
        d=d,
        # Constexprs
        cxpr_block_size=cxpr_block_size,
    )


def gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    gelu_tanh_and_mul_launcher(output, x)

    return output


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    silu_and_mul_launcher(output, x)

    return output
