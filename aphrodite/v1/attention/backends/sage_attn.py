"""Attention layer with SageAttention."""
from dataclasses import dataclass
from typing import Optional

import torch
from sageattention import sageattn

from aphrodite.attention.backends.abstract import (AttentionBackend,
                                                   AttentionImpl,
                                                   AttentionType)
from aphrodite.config import AphroditeConfig
from aphrodite.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                                   CommonAttentionMetadata)
from aphrodite.v1.kv_cache_interface import AttentionSpec


@dataclass
class SageAttentionMetadata:
    """Metadata for SageAttentionBackend."""
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


class SageAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = False

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # SageAttention doesn't support head sizes larger than 128
        return [64, 96, 128]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        supported_head_sizes = cls.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by SageAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN_APHRODITE_V1"

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["SageAttentionMetadata"]:
        return SageAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["SageAttentionMetadataBuilder"]:
        return SageAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        # Standard cache shape for paged attention
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        # Standard stride order
        return (0, 1, 2, 3, 4)


class SageAttentionMetadataBuilder(
        AttentionMetadataBuilder[SageAttentionMetadata]):

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 aphrodite_config: AphroditeConfig, device: torch.device):
        self.device = device

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> SageAttentionMetadata:
        """Build attention metadata."""
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        return SageAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=causal)

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        # SageAttention doesn't support cascade attention
        return False


class SageAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
    ) -> None:
        if logits_soft_cap is not None:
            raise ValueError("SageAttention does not support logits soft cap.")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "KV sharing is not supported in SageAttention backend.")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        SageAttentionBackend.validate_head_size(head_size)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if kv_cache_dtype != "auto":
            raise NotImplementedError(
                "SageAttention backend does not support quantized KV cache.")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with SageAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, block_size, num_kv_heads,
                      head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not supported for "
                "SageAttention")

        if attn_metadata is None:
            # Profiling run.
            return query  # Just return query for profiling

        # Create output tensor since accept_output_buffer = False
        num_tokens = query.shape[0]
        output = torch.empty(num_tokens, self.num_heads * self.head_size,
                             dtype=query.dtype, device=query.device)

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY,
                              AttentionType.ENCODER):
            result = self._forward_encoder_attention(
                query[:attn_metadata.num_actual_tokens],
                key[:attn_metadata.num_actual_tokens],
                value[:attn_metadata.num_actual_tokens],
                attn_metadata)
            output[:attn_metadata.num_actual_tokens] = result
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens

        # For decoder and cross-attention, use KV cache
        if kv_cache.numel() > 0:
            key_cache, value_cache = kv_cache.unbind(0)

            # Update KV cache if we have new key/value pairs
            if key is not None and value is not None:
                self._reshape_and_cache(
                    key, value, key_cache, value_cache,
                    attn_metadata.slot_mapping)

            # Use cached keys and values for attention computation
            result = self._forward_with_kv_cache(
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                attn_metadata)
            output[:num_actual_tokens] = result
            return output
        else:
            # Direct attention computation (prefill without cache)
            result = self._forward_direct_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                attn_metadata)
            output[:num_actual_tokens] = result
            return output

    def _reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Reshape and cache key/value tensors."""
        # Flatten slot mapping and use it to update cache
        flat_slot_mapping = slot_mapping.flatten()

        # Get the number of tokens to cache
        num_tokens = key.shape[0]

        # Reshape key and value to match cache format
        key_to_cache = key.view(num_tokens, self.num_kv_heads,
                                self.head_size)
        value_to_cache = value.view(num_tokens, self.num_kv_heads,
                                    self.head_size)

        # Update the cache using slot mapping
        for i, slot_idx in enumerate(flat_slot_mapping[:num_tokens]):
            if slot_idx >= 0:  # Valid slot
                block_idx = slot_idx // key_cache.shape[1]
                block_offset = slot_idx % key_cache.shape[1]
                key_cache[block_idx, block_offset, :, :] = key_to_cache[i]
                value_cache[block_idx, block_offset, :, :] = value_to_cache[i]

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache."""
        # For encoder attention, we use direct Q, K, V tensors
        return self._compute_sage_attention(
            query, key, value, attn_metadata, is_causal=False)

    def _forward_direct_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
    ) -> torch.Tensor:
        """Forward pass with direct Q, K, V tensors (prefill)."""
        return self._compute_sage_attention(
            query, key, value, attn_metadata,
            is_causal=attn_metadata.causal)

    def _forward_with_kv_cache(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
    ) -> torch.Tensor:
        """Forward pass using KV cache (decode or prefill with cache)."""
        batch_size = attn_metadata.query_start_loc.shape[0] - 1

        max_seq_len = attn_metadata.max_seq_len
        key_list = []
        value_list = []

        for i in range(batch_size):
            seq_len = attn_metadata.seq_lens[i].item()
            block_table = attn_metadata.block_table[i]

            seq_key = self._extract_from_cache(key_cache, block_table,
                                               seq_len)
            seq_value = self._extract_from_cache(value_cache, block_table,
                                                 seq_len)

            key_list.append(seq_key)
            value_list.append(seq_value)

        key = self._pad_and_concat(key_list, max_seq_len)
        value = self._pad_and_concat(value_list, max_seq_len)

        return self._compute_sage_attention(
            query, key, value, attn_metadata,
            is_causal=attn_metadata.causal)

    def _extract_from_cache(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Extract a sequence from the paged cache."""
        block_size = cache.shape[1]
        num_blocks = (seq_len + block_size - 1) // block_size

        extracted = []
        for block_idx in range(num_blocks):
            block_id = block_table[block_idx].item()
            if block_idx == num_blocks - 1:
                # Last block might be partial
                end_offset = seq_len % block_size
                if end_offset == 0:
                    end_offset = block_size
                extracted.append(cache[block_id, :end_offset])
            else:
                extracted.append(cache[block_id])

        return torch.cat(extracted, dim=0)

    def _pad_and_concat(
        self,
        tensor_list: list[torch.Tensor],
        max_len: int,
    ) -> torch.Tensor:
        """Pad tensors to max_len and concatenate."""
        padded = []
        for tensor in tensor_list:
            if tensor.shape[0] < max_len:
                padding = torch.zeros(
                    max_len - tensor.shape[0],
                    *tensor.shape[1:],
                    dtype=tensor.dtype,
                    device=tensor.device)
                padded.append(torch.cat([tensor, padding], dim=0))
            else:
                padded.append(tensor[:max_len])

        return torch.cat(padded, dim=0)

    def _compute_sage_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SageAttentionMetadata,
        is_causal: bool,
    ) -> torch.Tensor:
        """Compute attention using SageAttention."""
        # NOTE: We'll handle GQA expansion after reshaping to avoid dimension
        # issues

        # NOTE: SageAttention doesn't support ALiBi or sliding window in basic
        # API
        if self.alibi_slopes is not None:
            raise NotImplementedError(
                "ALiBi slopes are not supported in SageAttention backend")
        if self.sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is not supported in SageAttention backend")

        num_tokens = query.shape[0]
        output = torch.empty(num_tokens, self.num_heads * self.head_size,
                             dtype=query.dtype, device=query.device)

        # Process each sequence separately
        batch_size = attn_metadata.query_start_loc.shape[0] - 1

        for i in range(batch_size):
            query_start = attn_metadata.query_start_loc[i].item()
            query_end = attn_metadata.query_start_loc[i + 1].item()
            query_len = query_end - query_start
            seq_len = attn_metadata.seq_lens[i].item()

            seq_query = query[
                query_start:query_end]  # [query_len, num_heads * head_size]
            seq_key = key[
                i * attn_metadata.max_seq_len:i *
                attn_metadata.max_seq_len + seq_len
            ]  # [seq_len, num_kv_heads * head_size]
            seq_value = value[
                i * attn_metadata.max_seq_len:i *
                attn_metadata.max_seq_len +
                seq_len]  # [seq_len, num_kv_heads * head_size]

            # First reshape to separate heads and head_size
            query_len = query_end - query_start
            seq_query = seq_query.view(
                query_len, self.num_heads,
                self.head_size)  # [query_len, num_heads, head_size]
            seq_key = seq_key.view(
                seq_len, self.num_kv_heads,
                self.head_size)  # [seq_len, num_kv_heads, head_size]
            seq_value = seq_value.view(
                seq_len, self.num_kv_heads,
                self.head_size)  # [seq_len, num_kv_heads, head_size]

            # Expand KV heads if needed (for GQA)
            if self.num_kv_heads != self.num_heads:
                seq_key = seq_key.repeat_interleave(
                    self.num_queries_per_kv,
                    dim=1)  # [seq_len, num_heads, head_size]
                seq_value = seq_value.repeat_interleave(
                    self.num_queries_per_kv,
                    dim=1)  # [seq_len, num_heads, head_size]

            # Reshape for SageAttention
            # SageAttention expects
            # [batch_size, num_heads, seq_len, head_size] (HND layout)
            seq_query = seq_query.unsqueeze(
                0).transpose(1, 2)  # [1, num_heads, query_len, head_size]
            seq_key = seq_key.unsqueeze(
                0).transpose(1, 2)  # [1, num_heads, seq_len, head_size]
            seq_value = seq_value.unsqueeze(
                0).transpose(1, 2)  # [1, num_heads, seq_len, head_size]

            # Compute attention using SageAttention
            # For decode phase (query_len=1), we can't use causal since
            # qo_len != kv_len
            use_causal = is_causal and (query_len == seq_len)

            seq_output = sageattn(
                q=seq_query,
                k=seq_key,
                v=seq_value,
                tensor_layout="HND",
                is_causal=use_causal,
                sm_scale=self.scale)

            # Convert back to [query_len, num_heads, head_size] then flatten
            # to [query_len, num_heads * head_size]
            seq_output = seq_output.squeeze(
                0).transpose(0, 1)  # [query_len, num_heads, head_size]
            seq_output_flat = seq_output.contiguous().view(
                seq_output.shape[0], -1)  # [query_len, num_heads * head_size]
            output[query_start:query_end] = seq_output_flat

        return output
