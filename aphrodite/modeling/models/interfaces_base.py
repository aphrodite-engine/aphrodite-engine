from typing import (TYPE_CHECKING, Optional, Protocol, Type, Union, overload,
                    runtime_checkable)

import torch
import torch.nn as nn
from loguru import logger
from typing_extensions import TypeIs, TypeVar

from aphrodite.common.utils import supports_kw

if TYPE_CHECKING:
    from aphrodite.common.config import AphroditeConfig
    from aphrodite.modeling.layers.pooler import PoolerOutput
    from aphrodite.modeling.pooling_metadata import PoolingMetadata
    from aphrodite.modeling.sampling_metadata import SamplingMetadata

# The type of hidden states
# Currently, T = torch.Tensor for all models except for Medusa
# which has T = List[torch.Tensor]
T = TypeVar("T", default=torch.Tensor)
T_co = TypeVar("T_co", default=torch.Tensor, covariant=True)

# NOTE: Unlike those in `interfaces.py`, we don't define `ClassVar` tags
# for the base interfaces to avoid breaking OOT registration for existing models
# that don't inherit from the base interface classes


@runtime_checkable
class AphroditeModel(Protocol[T_co]):
    """The interface required for all models in vLLM."""

    def __init__(
        self,
        aphrodite_config: "AphroditeConfig",
        prefix: str = "",
    ) -> None:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> T_co:
        ...


def _check_aphrodite_model_init(model: Union[Type[object], object]) -> bool:
    model_init = model.__init__
    return supports_kw(model_init, "aphrodite_config")


def _check_aphrodite_model_forward(model: Union[Type[object], object]) -> bool:
    model_forward = getattr(model, "forward", None)
    if not callable(model_forward):
        return False

    aphrodite_kws = ("input_ids", "positions")
    missing_kws = tuple(kw for kw in aphrodite_kws
                        if not supports_kw(model_forward, kw))

    if missing_kws and (isinstance(model, type)
                        and issubclass(model, nn.Module)):
        logger.warning(
            "The model ({}) is missing "
            "vLLM-specific keywords from its `forward` method: {}",
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
    return _check_aphrodite_model_init(model) and _check_aphrodite_model_forward(model)


@runtime_checkable
class AphroditeModelForTextGeneration(AphroditeModel[T], Protocol[T]):
    """The interface required for all generative models in vLLM."""

    def compute_logits(
        self,
        hidden_states: T,
        sampling_metadata: "SamplingMetadata",
    ) -> Optional[T]:
        """Return `None` if TP rank > 0."""
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
class AphroditeModelForPooling(AphroditeModel[T], Protocol[T]):
    """The interface required for all pooling models in vLLM."""

    def pooler(
        self,
        hidden_states: T,
        pooling_metadata: "PoolingMetadata",
    ) -> "PoolerOutput":
        """Only called on TP rank 0."""
        ...


@overload
def is_pooling_model(model: Type[object]) -> TypeIs[Type[AphroditeModelForPooling]]:
    ...


@overload
def is_pooling_model(model: object) -> TypeIs[AphroditeModelForPooling]:
    ...


def is_pooling_model(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[AphroditeModelForPooling]], TypeIs[AphroditeModelForPooling]]:
    if not is_aphrodite_model(model):
        return False

    if isinstance(model, type):
        return isinstance(model, AphroditeModelForPooling)

    return isinstance(model, AphroditeModelForPooling)
