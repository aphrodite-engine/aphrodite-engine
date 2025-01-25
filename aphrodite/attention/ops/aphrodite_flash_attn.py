from typing import Optional, Union

import torch

import aphrodite._custom_ops as ops


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _flash_attn_forward(
    q, k, v, dropout_p, softmax_scale, causal,
    window_size, softcap, alibi_slopes,
    return_softmax, *, out=None
):
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    (out, q, k, v, out_padded, softmax_lse,
     S_dmask, rng_state) = ops.fwd(
        q=q,
        k=k,
        v=v,
        out=out,
        alibi_slopes=alibi_slopes,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        return_softmax=return_softmax,
        gen=None,
    )  # type: ignore
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


def _flash_attn_varlen_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    return_softmax,
    block_table,
    *,
    out=None
):
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    (out, q, k, v, out_padded, softmax_lse,
     S_dmask, rng_state) = ops.varlen_fwd(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        block_table=block_table,
        return_softmax=return_softmax,
        gen=None,
        out=out,
        seqused_k=None,
        zero_tensors=False,
    )  # type: ignore
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        out=None,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        (out, q, k, v, out_padded, softmax_lse,
         S_dmask, rng_state) = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            out=out,
        )
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        block_table,
        out=None,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        (out, q, k, v, out_padded, softmax_lse,
         S_dmask, rng_state) = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=block_table,
            out=out,
        )
        ctx.save_for_backward(
            q, k, v, out_padded, softmax_lse, cu_seqlens_q,
            cu_seqlens_k, rng_state
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)
    

def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    *,
    out=None,
):
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        out,
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
    *,
    out=None,
):
    assert k_cache.stride(-1) == 1, (
        "k_cache must have contiguous last dimension"
    )
    assert v_cache.stride(-1) == 1, (
        "v_cache must have contiguous last dimension"
    )
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens,
            dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    out, softmax_lse = ops.fwd_kvcache(
        q=q,
        kcache=k_cache,
        vcache=v_cache,
        k=k,
        v=v,
        seqlens_k=cache_seqlens,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_batch_idx=cache_batch_idx,
        block_table=block_table,
        alibi_slopes=alibi_slopes,
        out=out,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        num_splits=num_splits,
    )  # type: ignore
    return (out, softmax_lse) if return_softmax_lse else out
