"""Python wrapper for VMM (Virtual Memory Management) operations."""

import torch

import aphrodite._custom_ops as ops


def init_kvcached(dev_str: str, page_size: int, contiguous_layout: bool) -> None:
    """Initialize the kvcached VMM system."""
    ops.init_kvcached(dev_str, page_size, contiguous_layout)


def shutdown_kvcached() -> None:
    """Shutdown the kvcached VMM system."""
    ops.shutdown_kvcached()


def create_kv_tensors(size: int, dtype_size: int, dev_str: str, num_layers: int) -> list[torch.Tensor]:
    """Create KV cache tensors with virtual memory backing."""
    return ops.create_kv_tensors(size, dtype_size, dev_str, num_layers)


def kv_tensors_created() -> bool:
    """Check if KV tensors have been created."""
    return ops.kv_tensors_created()


def map_to_kv_tensors(offsets: list[int]) -> None:
    """Map physical memory pages to KV tensors at given offsets."""
    ops.map_to_kv_tensors(offsets)


def unmap_from_kv_tensors(offsets: list[int]) -> None:
    """Unmap physical memory pages from KV tensors at given offsets."""
    ops.unmap_from_kv_tensors(offsets)
