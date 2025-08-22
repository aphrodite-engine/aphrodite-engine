"""
Smart CPU offload policy for selectively moving model parameters to CPU.

Provides a policy-based approach to CPU offloading that targets
parameters with low access frequency and computational intensity, rather than
blindly offloading the first N bytes of parameters.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch.nn as nn


@dataclass
class OffloadPolicy:
    """Policy for determining which parameters should be offloaded to CPU."""

    max_bytes: int
    """Maximum number of bytes to offload to CPU."""

    offload_predicates: list[Callable[[str, nn.Parameter], bool]]
    """List of predicate functions that determine if a parameter should be
    offloaded. Each predicate takes (parameter_name, parameter) and returns
    True if it should be offloaded."""

    def should_offload(self, name: str, param: nn.Parameter) -> bool:
        """Check if a parameter should be offloaded based on predicates."""
        return any(predicate(name, param)
                   for predicate in self.offload_predicates)


def is_embedding_table(name: str, param: nn.Parameter) -> bool:
    """Identify embedding tables - used once per prompt, high memory."""
    return any(pattern in name.lower() for pattern in [
        ".embed_tokens.",
        ".embeddings.",
        ".word_embeddings.",
        ".token_embeddings.",
        "embed_tokens.weight",
        "embeddings.weight",
    ])


def is_norm_vector(name: str, param: nn.Parameter) -> bool:
    """Identify normalization vectors - small compute, infrequent access."""
    return (
        param.ndim == 1 and  # Must be a vector
        any(pattern in name.lower() for pattern in [
            ".norm.",
            ".layer_norm.",
            ".rms_norm.",
            ".ln_",
            "_norm",
            "norm.weight",
            "norm.bias",
        ])
    )


def is_positional_embedding(name: str, param: nn.Parameter) -> bool:
    """Identify positional embeddings - used once per sequence."""
    return any(pattern in name.lower() for pattern in [
        ".pos_emb",
        ".position_emb",
        ".rotary_emb.",
        ".rope.",
        "positional_encoding",
    ])


def is_lm_head(name: str, param: nn.Parameter) -> bool:
    """Identify language model head - only used for final logits."""
    return any(pattern in name.lower() for pattern in [
        ".lm_head.",
        ".output_projection.",
        ".classifier.",
        "output.weight",
    ])


def is_bias_vector(name: str, param: nn.Parameter) -> bool:
    """Identify bias vectors - small compute overhead."""
    return (
        param.ndim == 1 and
        name.lower().endswith(".bias")
    )


def create_conservative_policy(max_bytes: int) -> OffloadPolicy:
    """Create a conservative offload policy targeting very safe parameters."""
    return OffloadPolicy(
        max_bytes=max_bytes,
        offload_predicates=[
            is_norm_vector,
            is_positional_embedding,
            is_bias_vector,
        ]
    )


def create_aggressive_policy(max_bytes: int) -> OffloadPolicy:
    """Create an aggressive offload policy with embeddings and LM head."""
    return OffloadPolicy(
        max_bytes=max_bytes,
        offload_predicates=[
            is_embedding_table,
            is_norm_vector,
            is_positional_embedding,
            is_lm_head,
            is_bias_vector,
        ]
    )


def create_embeddings_only_policy(max_bytes: int) -> OffloadPolicy:
    """Create a policy that only offloads embedding tables."""
    return OffloadPolicy(
        max_bytes=max_bytes,
        offload_predicates=[
            is_embedding_table,
        ]
    )


_global_offload_policy: Optional[OffloadPolicy] = None


def set_global_offload_policy(policy: Optional[OffloadPolicy]) -> None:
    """Set the global offload policy."""
    global _global_offload_policy
    _global_offload_policy = policy


def get_global_offload_policy() -> Optional[OffloadPolicy]:
    """Get the current global offload policy."""
    return _global_offload_policy


def has_global_offload_policy() -> bool:
    """Check if a global offload policy is set."""
    return _global_offload_policy is not None
