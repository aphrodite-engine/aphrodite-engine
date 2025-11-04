from abc import abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from aphrodite.config import AphroditeConfig
from aphrodite.modeling.layers.attention_layer_base import AttentionLayerBase
from aphrodite.v1.kv_cache_interface import KVCacheSpec, MambaSpec

if TYPE_CHECKING:
    from aphrodite.attention.backends.abstract import AttentionBackend


class MambaBase(AttentionLayerBase):
    """
    Base class for Mamba-like layers which support the v1 engine.
    Inherit from this class if you implement a custom layer.
    """

    # Contains the KV cache (mamba state) for the layer
    # in the shape specified by `self.get_state_shape`.
    kv_cache: tuple[torch.Tensor, ...]

    @abstractmethod
    def get_state_shape(self) -> Iterable[tuple[int, ...]]:
        """
        Defines the shape of the state.
        For mamba layers this is usually a (conv_state, ssm_state) tuple.
        In this case, returns (conv_state_shape, ssm_state_shape).
        """
        pass

    @property
    @abstractmethod
    def mamba_type(self) -> str:
        pass

    @abstractmethod
    def get_attn_backend(self) -> type["AttentionBackend"]:
        """Get the attention backend class for this Mamba layer."""
        pass

    @abstractmethod
    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        pass

    def get_kv_cache_spec(self, aphrodite_config: AphroditeConfig) -> KVCacheSpec | None:
        if (
            aphrodite_config.speculative_config is not None
            and aphrodite_config.model_config.hf_config.model_type not in ["qwen3_next"]
        ):
            raise NotImplementedError("Mamba with speculative decoding is not supported yet.")
        mamba_block_size = aphrodite_config.cache_config.mamba_block_size
        page_size_padded = aphrodite_config.cache_config.mamba_page_size_padded
        return MambaSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.mamba_type,
            num_speculative_blocks=(
                aphrodite_config.speculative_config.num_speculative_tokens if aphrodite_config.speculative_config else 0
            ),
        )
