# Adapted from https://github.com/stackav-oss/conch/blob/e4ac60c47ebcaf76a5fb38982d6a8bbc169f65f6/conch/kernels/normalization  # noqa: E501

import torch

from aphrodite.triton_utils import tl, triton


@triton.jit  # type: ignore[misc]
def _rms_norm_kernel(  # noqa: PLR0913
    # Tensors
    output_ptr: tl.tensor,  # [..., hidden_size]
    x_ptr: tl.tensor,  # [..., hidden_size]
    residual_ptr: tl.tensor,  # [..., hidden_size]
    weight_ptr: tl.tensor,  # [hidden_size]
    # Scalars
    hidden_size: int,
    epsilon: float,
    # Constexprs
    cxpr_block_size: tl.constexpr,
    cxpr_use_residual: tl.constexpr,
) -> None:
    """Implementation of RMS norm kernel.

    Args:
        output_ptr: Pointer to output tensor, shape: (num_tokens, hidden_size).
        x_ptr: Pointer to input tensor, shape: (num_tokens, hidden_size).
        residual_ptr: Pointer to residual tensor, shape:
                      (num_tokens, hidden_size).
        weight_ptr: Pointer to weight tensor, shape: (hidden_size,).
        hidden_size: Hidden size.
        epsilon: Epsilon value.
        cxpr_block_size: Number of elements to process at once.
        cxpr_use_residual: Whether to use residual tensor.
    """
    token_index = tl.program_id(0)
    token_offset = token_index * hidden_size

    block_offsets = tl.arange(0, cxpr_block_size)
    mask = block_offsets < hidden_size

    x = tl.load(x_ptr + token_offset + block_offsets, mask=mask)
    w = tl.load(weight_ptr + block_offsets, mask=mask)

    if cxpr_use_residual:
        # Load residual, add it to x, and store it
        residual = tl.load(residual_ptr + token_offset + block_offsets,
                           mask=mask)
        x += residual
        tl.store(residual_ptr + token_offset + block_offsets, x, mask=mask)

        # If we are using the residual, we will write the result to the input
        # tensor
        output_ptr = x_ptr

    # For parity with vLLM, we will use fp32 here
    x = x.to(tl.float32)
    mean_of_squares = tl.sum(x * x) / hidden_size
    rms_inv = tl.rsqrt(mean_of_squares + epsilon)

    result = (x * rms_inv).to(x_ptr.dtype.element_ty) * w

    tl.store(output_ptr + token_offset + block_offsets, result, mask=mask)


def rms_norm_launcher(
    output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """Launch rms_norm kernel.

    Args:
        output: Output tensor, of shape (..., hidden_size).
        x: Input tensor, of shape (..., hidden_size).
        weight: Weight tensor, of shape (hidden_size,).
        epsilon: Epsilon value.
    """
    hidden_size = x.size(-1)
    num_tokens = x.numel() // hidden_size

    assert output.shape == x.shape, (
        "Output shape must match input shape")
    assert output.stride(-2) == x.stride(-2), (
        "Output and input strides must match")
    assert output.stride(-2) == hidden_size, (
        "Hidden size must match second-to-last stride of input/output")
    assert weight.size(0) == hidden_size, (
        "Weight size must match hidden size")

    assert output.is_contiguous()
    assert x.is_contiguous()
    assert weight.is_contiguous()

    # Parallelize over the number of tokens
    grid = (num_tokens,)

    # Launch kernel
    _rms_norm_kernel[grid](
        # Tensors
        output_ptr=output,
        x_ptr=x,
        residual_ptr=None,
        weight_ptr=weight,
        # Scalars
        hidden_size=hidden_size,
        epsilon=epsilon,
        # Constexprs
        # Note: we _could_ run out of shared memory here if hidden_size is too
        # large.
        # If this is a concern, we could allocate some extra memory for the
        # sum of squares and then reduce it in a second kernel.
        cxpr_block_size=triton.next_power_of_2(hidden_size),
        cxpr_use_residual=False,
    )


def fused_add_rms_norm_launcher(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """Launch rms_norm kernel.

    Args:
        x: Input tensor, of shape (..., hidden_size).
        residual: Residual tensor, of shape (..., hidden_size).
        weight: Weight tensor, of shape (hidden_size,).
        epsilon: Epsilon value.
    """
    hidden_size = x.size(-1)
    num_tokens = x.numel() // hidden_size

    assert x.shape == residual.shape, (
        "Input shape must match residual shape")
    assert x.stride(-2) == residual.stride(-2), (
        "Input and residual strides must match")
    assert x.stride(-2) == hidden_size, (
        "Hidden size must match second-to-last stride of input")
    assert weight.size(0) == hidden_size, (
        "Weight size must match hidden size")

    assert x.is_contiguous()
    assert residual.is_contiguous()
    assert weight.is_contiguous()

    # Parallelize over the number of tokens
    grid = (num_tokens,)

    # Launch kernel
    _rms_norm_kernel[grid](
        # Tensors
        output_ptr=None,
        x_ptr=x,
        residual_ptr=residual,
        weight_ptr=weight,
        # Scalars
        hidden_size=hidden_size,
        epsilon=epsilon,
        # Constexprs
        # Note: we _could_ run out of shared memory here if hidden_size is too
        # large.
        # If this is a concern, we could allocate some extra memory for the
        # sum of squares and then reduce it in a second kernel.
        cxpr_block_size=triton.next_power_of_2(hidden_size),
        cxpr_use_residual=True,
    )


@triton.jit  # type: ignore[misc]
def _gemma_rms_norm_inplace_kernel(
    x_ptr: tl.tensor,
    weights_ptr: tl.const,
    hidden_size: int,
    eps: float,
    cxpr_block_size: tl.constexpr,
) -> None:
    """Perform Gemma's version of RMS layer norm.

    Args:
        x_ptr: Tensor to be normalized, shape: (num_tokens, hidden_size).
        weights_ptr: learned weights of norm layer, shape: (hidden_size,).
        hidden_size: hidden size, often corresponds to head size.
        eps: value to pad during inverse-rms calculation.
        cxpr_block_size: must be next-power-of-two >= hidden_size.
    """
    # One block per row/token
    token_id = tl.program_id(0)
    token_row_ptr = x_ptr + token_id * hidden_size

    # Gemma RMS details
    # - Compute in float32, then convert back to original type
    # - mean-of-squares = mean(x ** 2)
    # - root-of-mean-of-squares (RMS) = sqrt(mean-of-squares + eps)
    # - weigh scaled inputs: x = (x / RMS) * (1 + weight)
    # fp32 as working format is discussed here:
    # https://github.com/huggingface/transformers/pull/29402
    # It would be reasonable do something a little bit more precise,
    # i.e. basing the precision on the input precision.  On the
    # other hand, fp32 is probably the most efficient type on most
    # GPUs' compute cores.

    offsets = tl.arange(0, cxpr_block_size)
    mask = offsets < hidden_size
    x = tl.load(token_row_ptr + offsets, mask=mask).to(tl.float32)
    x_sq = x * x
    mean_squares = tl.sum(x_sq) / hidden_size
    recip_rms = tl.rsqrt(mean_squares + eps)
    w = tl.load(weights_ptr + offsets, mask=mask).to(tl.float32)
    x = x * recip_rms * (1.0 + w)
    tl.store(token_row_ptr + offsets, x, mask=mask)  # Implicit cast back to original dtype  # noqa: E501


def gemma_rms_norm_inplace_launcher(
    x: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float = 1e-6,
) -> None:
    """Perform Gemma's version of RMS layer norm.

    Args:
        x: Tensor to be normalized, shape: (num_tokens, hidden_size).
        weights: learned weights of norm layer, shape: (hidden_size,).
        epsilon: value to pad during inverse-rms calculation.
    """
    # Sanity check that the hidden dimensions match
    hidden_size = x.shape[-1]
    if hidden_size != weights.shape[-1]:
        msg = (
            f"Input hidden dimenson ({hidden_size}) does not match length of "
            f"weights ({weights.shape[-1]})")
        raise ValueError(msg)

    # Only support two-dimensional x
    if len(x.shape) != 2:  # noqa: PLR2004
        msg = (
            f"x is {len(x.shape)}-dimensional.  Only supporting two "
            "dimensions.")
        raise ValueError(msg)

    block_size = triton.next_power_of_2(hidden_size)
    grid = (x.shape[0],)
    _gemma_rms_norm_inplace_kernel[grid](
        x_ptr=x,
        weights_ptr=weights,
        hidden_size=hidden_size,
        eps=epsilon,
        cxpr_block_size=block_size,
    )

def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Root-mean-square normalization.

    Args:
        x: Input tensor, of shape (..., hidden_size).
        weight: Weight tensor, of shape (hidden_size,).
        epsilon: Epsilon value.
    """
    output = torch.empty_like(x, dtype=x.dtype, device=x.device)

    # Call kernel launch wrapper
    rms_norm_launcher(output, x, weight, epsilon)

    return output


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Root-mean-square normalization with fused add.

    Args:
        x: Input tensor, of shape (..., hidden_size).
        residual: Residual tensor, of shape (..., hidden_size).
        weight: Weight tensor, of shape (hidden_size,).
        epsilon: Epsilon value.
    """
    fused_add_rms_norm_launcher(x, residual, weight, epsilon)
    return x, residual


def gemma_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Gemma RMS norm operation."""
    if residual is not None:
        x = x + residual
        residual = x

    gemma_rms_norm_inplace_launcher(x, weight, variance_epsilon)

    return x if residual is None else (x, residual)
