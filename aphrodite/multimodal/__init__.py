from .hasher import MultiModalHasher
from .inputs import (
    BatchedTensorInputs,
    ModalityData,
    MultiModalDataBuiltins,
    MultiModalDataDict,
    MultiModalKwargs,
    MultiModalKwargsItems,
    MultiModalPlaceholderDict,
    MultiModalUUIDDict,
    NestedTensors,
)
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()
"""
The global [`MultiModalRegistry`][aphrodite.multimodal.registry.MultiModalRegistry]
is used by model runners to dispatch data processing according to the target
model.

Info:
    [mm_processing](../../../design/mm_processing.md)
"""

__all__ = [
    "BatchedTensorInputs",
    "ModalityData",
    "MultiModalDataBuiltins",
    "MultiModalDataDict",
    "MultiModalHasher",
    "MultiModalKwargs",
    "MultiModalKwargsItems",
    "MultiModalPlaceholderDict",
    "MultiModalUUIDDict",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]
