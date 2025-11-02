"""Base class for attention-like layers."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aphrodite.config import AphroditeConfig
from aphrodite.v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    from aphrodite.attention.backends.abstract import AttentionBackend


class AttentionLayerBase(ABC):
    """
    Base class for attention-like layers (Attention, Mamba, etc.)
    that support the v1 engine.

    This provides a common interface for getting attention backends
    from different layer types.
    """

    @abstractmethod
    def get_attn_backend(self) -> type["AttentionBackend"]:
        """Get the attention backend class for this layer."""
        pass

    @abstractmethod
    def get_kv_cache_spec(self, aphrodite_config: AphroditeConfig) -> KVCacheSpec | None:
        """
        Get the KV cache spec for this layer.
        May be None if the layer does not need KV cache.
        """
        pass
