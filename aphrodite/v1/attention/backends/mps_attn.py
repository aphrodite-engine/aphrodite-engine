import torch

from aphrodite.attention.backends.abstract import AttentionBackend
from aphrodite.attention.backends.utils import CommonAttentionState

from .cpu_attn import TorchSDPABackendImpl as MpsSDPABackendImpl
from .cpu_attn import TorchSDPAMetadata as MpsSDPAMetadata
from .cpu_attn import TorchSDPAMetadataBuilderV1 as MpsSDPAMetadataBuilderV1
from .cpu_attn import _PagedAttention


class MpsSDPABackend(AttentionBackend):
    accept_output_buffer: bool = False

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        is_valid, supported_head_sizes = _PagedAttention.validate_head_size(head_size)
        if not is_valid:
            raise ValueError(
                f"Head size {head_size} is not supported on MPS. "
                f"Supported head sizes are: {supported_head_sizes}."
            )

    @staticmethod
    def get_name() -> str:
        return "MPS_SDPA_APHRODITE_V1"

    @staticmethod
    def get_impl_cls() -> type["MpsSDPABackendImpl"]:
        return MpsSDPABackendImpl

    @staticmethod
    def get_metadata_cls() -> type["MpsSDPAMetadata"]:
        return MpsSDPAMetadata

    @staticmethod
    def get_state_cls() -> type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> type["MpsSDPAMetadataBuilderV1"]:
        return MpsSDPAMetadataBuilderV1

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return _PagedAttention.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size
        )

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


# Aliases for consistency with other backends
MpsSDPAMetadata = MpsSDPAMetadata  # type: ignore
MpsSDPABackendImpl = MpsSDPABackendImpl  # type: ignore
MpsSDPAMetadataBuilderV1 = MpsSDPAMetadataBuilderV1  # type: ignore


