from typing import (TYPE_CHECKING, List, Optional, Protocol, Type, Union,
                    overload, runtime_checkable)

import torch
import torch.nn as nn
from loguru import logger
from transformers import PretrainedConfig
from typing_extensions import TypeIs, TypeVar

from aphrodite.common.utils import supports_kw

if TYPE_CHECKING:
    from aphrodite.attention import AttentionMetadata
    from aphrodite.common.config import CacheConfig
    from aphrodite.modeling.layers.pooler import PoolerOutput
    from aphrodite.modeling.layers.sampler import SamplerOutput
    from aphrodite.modeling.pooling_metadata import PoolingMetadata
    from aphrodite.modeling.sampling_metadata import SamplingMetadata
    from aphrodite.quantization import QuantizationConfig

# The type of HF config
C_co = TypeVar("C_co", bound=PretrainedConfig, covariant=True)

# The type of hidden states
# Currently, T = torch.Tensor for all models except for Medusa
# which has T = List[torch.Tensor]
T = TypeVar("T", default=torch.Tensor)
T_co = TypeVar("T_co", default=torch.Tensor, covariant=True)

# NOTE: Unlike those in `interfaces.py`, we don't define `ClassVar` tags
# for the base interfaces to avoid breaking OOT registration for existing models
# that don't inherit from the base interface classes


@runtime_checkable
class AphroditeModel(Protocol[C_co, T_co]):

    def __init__(
        self,
        config: C_co,
        *,
        cache_config: Optional["CacheConfig"],
        quant_config: Optional["QuantizationConfig"],
    ) -> None:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: "AttentionMetadata",
    ) -> T_co:
        ...


def _check_aphrodite_model_init(model: Union[Type[object], object]) -> bool:
    model_init = model.__init__
    aphrodite_kws = ("cache_config", "quant_config")
    missing_kws = tuple(kw for kw in aphrodite_kws
                        if not supports_kw(model_init, kw))

    if missing_kws and (isinstance(model, type)
                        and issubclass(model, nn.Module)):
        logger.warning(
            "The model (%s) is missing "
            "Aphrodite-specific keywords from its initializer: %s",
            model,
            missing_kws,
        )

    return len(missing_kws) == 0


def _check_aphrodite_model_forward(model: Union[Type[object], object]) -> bool:
    model_forward = getattr(model, "forward", None)
    if not callable(model_forward):
        return False

    aphrodite_kws = ("input_ids", "positions", "kv_caches", "attn_metadata")
    missing_kws = tuple(kw for kw in aphrodite_kws
                        if not supports_kw(model_forward, kw))

    if missing_kws and (isinstance(model, type)
                        and issubclass(model, nn.Module)):
        logger.warning(
            "The model (%s) is missing "
            "Aphrodite-specific keywords from its initializer: %s",
            model,
            missing_kws,
        )

    return len(missing_kws) == 0


@overload
def is_aphrodite_model(model: Type[object]) -> TypeIs[Type[AphroditeModel]]:
    ...


@overload
def is_aphrodite_model(model: object) -> TypeIs[AphroditeModel]:
    ...


def is_aphrodite_model(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[AphroditeModel]], TypeIs[AphroditeModel]]:
    return (_check_aphrodite_model_init(model)
            and _check_aphrodite_model_forward(model))


@runtime_checkable
class AphroditeModelForTextGeneration(AphroditeModel[C_co, T],
                                      Protocol[C_co, T]):

    def compute_logits(
        self,
        hidden_states: T,
        sampling_metadata: "SamplingMetadata",
    ) -> Optional[T]:
        """Return `None` if TP rank > 0."""
        ...

    def sample(
        self,
        logits: T,
        sampling_metadata: "SamplingMetadata",
    ) -> "SamplerOutput":
        """Only called on TP rank 0."""
        ...


@overload
def is_text_generation_model(
        model: Type[object]) -> TypeIs[Type[AphroditeModelForTextGeneration]]:
    ...


@overload
def is_text_generation_model(
        model: object) -> TypeIs[AphroditeModelForTextGeneration]:
    ...


def is_text_generation_model(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[AphroditeModelForTextGeneration]],
           TypeIs[AphroditeModelForTextGeneration]]:
    if not is_aphrodite_model(model):
        return False

    if isinstance(model, type):
        return isinstance(model, AphroditeModelForTextGeneration)

    return isinstance(model, AphroditeModelForTextGeneration)


@runtime_checkable
class AphroditeModelForEmbedding(AphroditeModel[C_co, T], Protocol[C_co, T]):

    def pooler(
        self,
        hidden_states: T,
        pooling_metadata: "PoolingMetadata",
    ) -> "PoolerOutput":
        """Only called on TP rank 0."""
        ...


@overload
def is_embedding_model(
        model: Type[object]) -> TypeIs[Type[AphroditeModelForEmbedding]]:
    ...


@overload
def is_embedding_model(model: object) -> TypeIs[AphroditeModelForEmbedding]:
    ...


def is_embedding_model(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[AphroditeModelForEmbedding]],
           TypeIs[AphroditeModelForEmbedding]]:
    if not is_aphrodite_model(model):
        return False

    if isinstance(model, type):
        return isinstance(model, AphroditeModelForEmbedding)

    return isinstance(model, AphroditeModelForEmbedding)
