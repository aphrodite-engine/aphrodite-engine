import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_kernel(
    Q,  # [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]  # noqa: E501
    K,
    Cos,
    Sin,
    Positions,  # [batch_size, seq_len] or [num_tokens]
    Offsets,  # [num_tokens]
    stride_qb,  # Batch stride for Q
    stride_qs,  # Sequence stride for Q
    stride_qh,  # Head stride for Q
    stride_kb,  # Batch stride for K
    stride_ks,  # Sequence stride for K
    stride_kh,  # Head stride for K
    stride_cosbs,
    stride_cosd,
    stride_sinbs,
    stride_sind,
    batch_size,  # Can be 1 for unbatched case
    seq_len,
    HEAD_Q,
    HEAD_K,
    rotary_dim: tl.constexpr,  # Mark as constexpr
    is_neox_style: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Program ID 0 is for heads, ID 1 is for sequence, ID 2 is for batch
    cur_head = tl.program_id(0)
    cur_seq = tl.program_id(1)
    cur_batch = tl.program_id(2)

    cur_head_range = cur_head * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    cur_seq_range = cur_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

    # Load positions and apply offsets
    pos = tl.load(
        Positions + cur_batch * seq_len + cur_seq_range,
        mask=cur_seq_range < seq_len,
        other=0,
    )
    if Offsets is not None:
        offsets = tl.load(
            Offsets + cur_batch * seq_len + cur_seq_range,
            mask=cur_seq_range < seq_len,
            other=0,
        )
        pos = pos + offsets

    if is_neox_style:
        # NeoX style: split in half
        dim_range0 = tl.arange(0, rotary_dim // 2)
        dim_range1 = tl.arange(rotary_dim // 2, rotary_dim)
    else:
        # GPT-J style: alternate indices
        dim_range0 = tl.arange(0, rotary_dim, 2)
        dim_range1 = tl.arange(1, rotary_dim, 2)

    # Update offsets to include batch dimension
    off_q = (
        cur_batch * stride_qb
        + cur_seq_range[:, None, None] * stride_qs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range0[None, None, :] * 1  # Assuming contiguous in last dim
    )

    off_q1 = (
        cur_seq_range[:, None, None] * stride_qs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range1[None, None, :] * stride_qs
    )

    off_dimcos_sin = (
        cur_seq_range[:, None, None] * stride_cosbs
        + dim_range0[None, None, :] * stride_cosd
    )

    q0 = tl.load(
        Q + off_q,
        mask=(cur_seq_range[:, None, None] < seq_len)
        & (cur_head_range[None, :, None] < HEAD_Q)
        & (cur_batch < batch_size),
        other=0.0,
    )
    q1 = tl.load(
        Q + off_q1,
        mask=(cur_seq_range[:, None, None] < seq_len)
        & (cur_head_range[None, :, None] < HEAD_Q)
        & (cur_batch < batch_size),
        other=0.0,
    )

    cos = tl.load(
        Cos + off_dimcos_sin,
        mask=cur_seq_range[:, None, None] < seq_len,
        other=0.0,
    )
    sin = tl.load(
        Sin + off_dimcos_sin,
        mask=cur_seq_range[:, None, None] < seq_len,
        other=0.0,
    )

    out0 = q0 * cos - q1 * sin
    out1 = q0 * sin + q1 * cos

    tl.store(
        Q + off_q,
        out0,
        mask=(cur_seq_range[:, None, None] < seq_len)
        & (cur_head_range[None, :, None] < HEAD_Q)
        & (cur_batch < batch_size),
    )
    tl.store(
        Q + off_q1,
        out1,
        mask=(cur_seq_range[:, None, None] < seq_len)
        & (cur_head_range[None, :, None] < HEAD_Q)
        & (cur_batch < batch_size),
    )

    off_k = (
        cur_batch * stride_kb
        + cur_seq_range[:, None, None] * stride_ks
        + cur_head_range[None, :, None] * stride_kh
        + dim_range0[None, None, :] * stride_ks
    )
    off_k1 = (
        cur_seq_range[:, None, None] * stride_ks
        + cur_head_range[None, :, None] * stride_kh
        + dim_range1[None, None, :] * stride_ks
    )

    off_dimcos_sin = (
        cur_seq_range[:, None, None] * stride_cosbs
        + dim_range0[None, None, :] * stride_cosd
    )

    k0 = tl.load(
        K + off_k,
        mask=(cur_seq_range[:, None, None] < seq_len)
        & (cur_head_range[None, :, None] < HEAD_K)
        & (cur_batch < batch_size),
        other=0.0,
    )
    k1 = tl.load(
        K + off_k1,
        mask=(cur_seq_range[:, None, None] < seq_len)
        & (cur_head_range[None, :, None] < HEAD_K)
        & (cur_batch < batch_size),
        other=0.0,
    )
    cos = tl.load(
        Cos + off_dimcos_sin,
        mask=cur_seq_range[:, None, None] < seq_len,
        other=0.0,
    )
    sin = tl.load(
        Sin + off_dimcos_sin,
        mask=cur_seq_range[:, None, None] < seq_len,
        other=0.0,
    )

    out_k0 = k0 * cos - k1 * sin
    out_k1 = k0 * sin + k1 * cos

    tl.store(
        K + off_k,
        out_k0,
        mask=(cur_seq_range[:, None, None] < seq_len)
        & (cur_head_range[None, :, None] < HEAD_K)
        & (cur_batch < batch_size),
    )
    tl.store(
        K + off_k1,
        out_k1,
        mask=(cur_seq_range[:, None, None] < seq_len)
        & (cur_head_range[None, :, None] < HEAD_K)
        & (cur_batch < batch_size),
    )

    # For partial rotary, we need to handle the pass-through portion
    if rotary_dim < BLOCK_DMODEL:
        pass_dim_range = tl.arange(rotary_dim, BLOCK_DMODEL)

        # Load and store pass-through values for Q
        off_q_pass = (
            cur_seq_range[:, None, None] * stride_qs
            + cur_head_range[None, :, None] * stride_qh
            + pass_dim_range[None, None, :] * stride_qs
        )
        q_pass = tl.load(
            Q + off_q_pass,
            mask=(cur_seq_range[:, None, None] < seq_len)
            & (cur_head_range[None, :, None] < HEAD_Q)
            & (cur_batch < batch_size),
            other=0.0,
        )
        tl.store(
            Q + off_q_pass,
            q_pass,
            mask=(cur_seq_range[:, None, None] < seq_len)
            & (cur_head_range[None, :, None] < HEAD_Q)
            & (cur_batch < batch_size),
        )

        # Same for K
        off_k_pass = (
            cur_seq_range[:, None, None] * stride_ks
            + cur_head_range[None, :, None] * stride_kh
            + pass_dim_range[None, None, :] * stride_ks
        )
        k_pass = tl.load(
            K + off_k_pass,
            mask=(cur_seq_range[:, None, None] < seq_len)
            & (cur_head_range[None, :, None] < HEAD_K)
            & (cur_batch < batch_size),
            other=0.0,
        )
        tl.store(
            K + off_k_pass,
            k_pass,
            mask=(cur_seq_range[:, None, None] < seq_len)
            & (cur_head_range[None, :, None] < HEAD_K)
            & (cur_batch < batch_size),
        )
    return


@torch.no_grad()
def rotary_emb_fwd(
    q, k, cos, sin, rotary_dim=None, is_neox_style=True, offsets=None
):
    # Handle different input shapes
    if q.ndim == 3:  # Batched input
        batch_size, seq_len = q.shape[:2]
        num_heads = q.shape[-1] // q.shape[-1]
    else:  # Unbatched input
        batch_size = 1
        seq_len = q.shape[0]
        num_heads = 1

    head_dim = q.shape[-1] // num_heads
    if rotary_dim is None:
        rotary_dim = head_dim

    # Create positions tensor if needed
    if q.ndim == 3:
        positions = torch.arange(seq_len, device=q.device).expand(
            batch_size, seq_len
        )
    else:
        positions = torch.arange(seq_len, device=q.device)

    # Handle offsets
    if offsets is None:
        offsets = torch.zeros_like(positions)

    # Skip if q is token indices
    if q.ndim == 2 and q.shape[-1] == 1:
        return q, k

    BLOCK_SEQ = 16
    BLOCK_HEAD = 4
    num_warps = 8 if head_dim >= 128 else 4

    grid = (
        triton.cdiv(num_heads, BLOCK_HEAD),
        triton.cdiv(seq_len, BLOCK_SEQ),
        batch_size,
    )

    _rotary_kernel[grid](
        q,
        k,
        cos,
        sin,
        positions,
        offsets,
        q.stride(0) if q.ndim == 3 else 0,  # Batch stride
        q.stride(1) if q.ndim == 3 else q.stride(0),  # Sequence stride
        q.stride(-1),  # Head stride
        k.stride(0) if k.ndim == 3 else 0,
        k.stride(1) if k.ndim == 3 else k.stride(0),
        k.stride(-1),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        batch_size,
        seq_len,
        num_heads,
        num_heads,  # Assuming same number of heads for K
        rotary_dim,
        is_neox_style,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return q, k
