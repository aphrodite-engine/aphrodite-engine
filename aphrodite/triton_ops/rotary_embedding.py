# Adapted from https://github.com/stackav-oss/conch/blob/e4ac60c47ebcaf76a5fb38982d6a8bbc169f65f6/conch/kernels/embedding/rotary_embedding.py  # noqa: E501

import torch

from aphrodite.triton_utils import tl, triton


@triton.jit  # type: ignore[misc]
def _rotary_embedding_kernel(
    positions_ptr: tl.const,
    query_ptr: tl.tensor,
    key_ptr: tl.tensor,
    cos_sin_cache_ptr: tl.const,
    rot_dim: int,
    query_stride: int,
    key_stride: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    cxpr_block_size: tl.constexpr,
) -> None:
    """Apply rotary positional embedding to query and key.

    Args:
        positions_ptr: token positions [num_tokens].
        query_ptr: queries [num_tokens, num_heads * head_size].
        key_ptr: keys [num_tokens, num_heads * head_size].
        cos_sin_cache_ptr: precacluated cos-sin, typically [rot_dim],
                           split into [cos | sin] for NeoX.
        rot_dim: cos_sin_cache_ptr len, typically head_size.
        query_stride: step between token entries.
        key_stride: step between token entries.
        num_heads: number of query heads.
        num_kv_heads: number of kv heads.
        head_size: head size.
        cxpr_block_size: block size (needs to be pow-2).
    """
    # Each block is responsible for one token
    token_idx = tl.program_id(0)
    pos = tl.load(positions_ptr + token_idx)
    # Rotation dep. on token pos; offset to entries for this position
    rot_cache_ptr = cos_sin_cache_ptr + pos * rot_dim

    # "apply_rotary_embedding" device function
    embed_dim = rot_dim // 2
    # For NeoX models, the cos-sin cache is expected to be [cos | sin]
    # Rotation is applied to pairs of entires, so there will be d/2 cos and
    # d/2 sin entries, e.g. one pair of cos-sin values for every two entries
    cos_ptr = rot_cache_ptr
    sin_ptr = rot_cache_ptr + embed_dim

    # Applying rotation to each vector
    # Each "thread" will:
    # - Determine which head its operating on and within this head which pair
    # of entires it is operating on
    # - Knowing which pair of entries, find the corresponding cos and sin
    # values
    # - Load the entries pair, apply rotation, write back
    #
    # Each token's entry has dimension [num_heads * head_size]
    # The loop ensures that the block of threads will be repeatedly used to
    # cover the entire token entry

    nq = num_heads * embed_dim
    # Offset to query entries for this token
    query_token_offset = token_idx.to(tl.int64) * query_stride
    i = tl.arange(0, cxpr_block_size)
    for _ in tl.range(0, nq, cxpr_block_size):
        head_idx = i // embed_dim
        token_head = query_token_offset + head_idx * head_size
        rot_offset = i % embed_dim

        # "apply_token_rotary_embedding" device function + IS_NEOX=true
        x_index = rot_offset
        y_index = rot_offset + embed_dim
        query_offset = query_ptr + token_head
        mask = i < nq
        x = tl.load(query_offset + x_index, mask=mask)
        y = tl.load(query_offset + y_index, mask=mask)
        cos = tl.load(cos_ptr + x_index, mask=mask)
        sin = tl.load(sin_ptr + x_index, mask=mask)
        x_rot = x * cos - y * sin
        y_rot = y * cos + x * sin
        tl.store(query_offset + x_index, x_rot, mask=mask)
        tl.store(query_offset + y_index, y_rot, mask=mask)
        # Next set of "block IDs"
        i += cxpr_block_size

    nk = num_kv_heads * embed_dim
    # Offset to key entires for this token
    key_token_offset = token_idx.to(tl.int64) * key_stride
    j = tl.arange(0, cxpr_block_size)
    for _ in tl.range(0, nk, cxpr_block_size):
        head_idx = j // embed_dim
        token_head = key_token_offset + head_idx * head_size
        rot_offset = j % embed_dim

        # "apply_token_rotary_embedding" device function + IS_NEOX=true
        x_index = rot_offset
        y_index = rot_offset + embed_dim
        key_offset = key_ptr + token_head
        mask = j < nk
        x = tl.load(key_offset + x_index, mask=mask)
        y = tl.load(key_offset + y_index, mask=mask)
        cos = tl.load(cos_ptr + x_index, mask=mask)
        sin = tl.load(sin_ptr + x_index, mask=mask)
        x_rot = x * cos - y * sin
        y_rot = y * cos + x * sin
        tl.store(key_offset + x_index, x_rot, mask=mask)
        tl.store(key_offset + y_index, y_rot, mask=mask)
        # Next set of "block IDs"
        j += cxpr_block_size


def rotary_embedding_launcher(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    *,
    is_neox: bool = True,
) -> None:
    """Apply rotary positional embedding to query and key.

    NOTE: Only supporting NeoX model.

    Args:
        positions: token positions [num_tokens].
        query: queries [num_tokens, num_heads * head_size].
        key: keys [num_tokens, num_heads * head_size].
        head_size: head size.
        cos_sin_cache: precacluated cos-sin, typically [rot_dim], split into
                       [cos | sin] for NeoX; use tensor provided by
                       aphrodite::RotaryEmbedding.
        is_neox: whether using NeoX, must be True; exists mostly for
                 visibility.
    """
    assert is_neox  # noqa: S101
    # Some checks to indirectly ensure that we're not batching
    assert len(positions.shape) == 1 and len(query.shape) == 2 and len(key.shape) == 2  # noqa: S101, PT018, PLR2004, E501

    # Note about using negative indices in shape and stride
    # The commonly (?) accepted organization of data appears to be:
    # - batch, token, other higher level dimensions in the leading positions
    # - data-specific dimensions in the lower positions
    # For example: [batch_size, num_token, embedding_size]
    # Some higher level dimensions may or may not exist.  For example, for
    # unbatched data, there wouldn't be a
    # batch_size dimension.  On the other hand, the lower-level dimensions
    # will always exist.  This is why it makes some sense to count backwards
    # when trying to access dimension information.

    # The use of negative indices may be excessive here given we're enforcing
    # batch==1 (and thus shape), but it's worth doing so to more easily
    # support batched processing in the future.

    num_tokens = query.numel() // query.shape[-1]
    rot_dim = cos_sin_cache.shape[-1]
    num_heads = query.shape[-1] // head_size
    num_kv_heads = key.shape[-1] // head_size
    query_stride = query.stride(-2)
    key_stride = key.stride(-2)

    grid = (num_tokens,)
    block = triton.next_power_of_2(head_size)
    _rotary_embedding_kernel[grid](
        positions,
        query,
        key,
        cos_sin_cache,
        rot_dim,
        query_stride,
        key_stride,
        num_heads,
        num_kv_heads,
        head_size,
        block,
    )


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    *,
    is_neox: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key.

    NOTE: Only supporting NeoX model.

    Args:
        positions: token positions [num_tokens].
        query: queries [num_tokens, num_heads * head_size].
        key: keys [num_tokens, num_heads * head_size].
        head_size: head size.
        cos_sin_cache: precacluated cos-sin, typically [rot_dim], split into [cos | sin] for NeoX; use tensor provided by vllm::RotaryEmbedding.
        is_neox: whether using NeoX, must be True; exists mostly for visibility.
    """
    rotary_embedding_launcher(positions, query, key, head_size, cos_sin_cache, is_neox=is_neox)
    return query, key
