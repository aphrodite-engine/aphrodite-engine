import sys
from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from typing import (Callable, Dict, List, Mapping, Optional, Tuple, Type,
                    TypedDict, TypeVar, Union, cast, final)

import numpy as np
import torch
import torch.types
from loguru import logger
from PIL import Image
from torch import nn
from typing_extensions import TypeAlias

from aphrodite.common.config import ModelConfig
from aphrodite.common.utils import (get_allowed_kwarg_only_overrides,
                                    is_list_of, json_map_leaves)
from aphrodite.inputs import InputContext

NestedTensors = Union[List["NestedTensors"], List[torch.Tensor], torch.Tensor]
"""
Uses a list instead of a tensor if the dimensions of each element do not match.
"""

BatchedTensorInputs: TypeAlias = Dict[str, NestedTensors]
"""
A dictionary containing nested tensors which have been batched via
:meth:`MultiModalInputs.batch`.
"""

if sys.version_info < (3, 9):
    # UserDict cannot be subscripted
    class _MultiModalInputsBase(UserDict):
        pass
else:

    class _MultiModalInputsBase(UserDict[str, NestedTensors]):
        pass


class MultiModalInputs(_MultiModalInputsBase):
    """
    A dictionary that represents the keyword arguments to
    :meth:`~torch.nn.Module.forward`.
    """

    @staticmethod
    def _try_stack(nested_tensors: NestedTensors) -> NestedTensors:
        """
        Recursively stacks lists of tensors when they all have the same shape.
        """
        if isinstance(nested_tensors, torch.Tensor):
            return nested_tensors

        if isinstance(nested_tensors, np.ndarray):
            return torch.from_numpy(nested_tensors)
        if isinstance(nested_tensors, (int, float)):
            return torch.tensor(nested_tensors)
        stacked = [MultiModalInputs._try_stack(t) for t in nested_tensors]
        if not is_list_of(stacked, torch.Tensor, check="all"):
            # Only tensors (not lists) can be stacked.
            return stacked

        tensors_ = cast(List[torch.Tensor], stacked)
        if any(t.shape != tensors_[0].shape for t in tensors_):
            # The tensors have incompatible shapes and can't be stacked.
            return tensors_

        return torch.stack(tensors_)

    @staticmethod
    def batch(inputs_list: List["MultiModalInputs"]) -> BatchedTensorInputs:
        """
        Batch multiple inputs together into a dictionary.
        The resulting dictionary has the same keys as the inputs.
        If the corresponding value from each input is a tensor and they all
        share the same shape, the output value is a single batched tensor;
        otherwise, the output value is a list containing the original value
        from each input.
        """
        if len(inputs_list) == 0:
            return {}

        item_lists: Dict[str, List[NestedTensors]] = defaultdict(list)

        for inputs in inputs_list:
            # For models that supports multiple modalities (e.g. Qwen2-VL),
            # different modalities will return different data keys,
            # so batch() should skip the same key check.

            for k, v in inputs.items():
                item_lists[k].append(v)

        return {
            k: MultiModalInputs._try_stack(item_list)
            for k, item_list in item_lists.items()
        }  # type: ignore

    @staticmethod
    def as_kwargs(
        batched_inputs: BatchedTensorInputs,
        *,
        device: torch.types.Device,
    ) -> BatchedTensorInputs:
        return json_map_leaves(lambda x: x.to(device, non_blocking=True),
                               batched_inputs)


_T = TypeVar("_T")

MultiModalData: TypeAlias = Union[_T, List[_T]]
"""
Either a single data instance, or a list of data instances.
The number of data instances allowed per modality is restricted by
`--limit-mm-per-prompt`.
"""


@final
class MultiModalDataBuiltins(TypedDict, total=False):
    """Modality types that are predefined by vLLM."""

    image: MultiModalData[Image.Image]
    """The input image(s)."""

    audio: MultiModalData[Tuple[np.ndarray, Union[int, float]]]
    """The input audio item(s) and corresponding sampling rate(s)."""


MultiModalDataDict = Union[MultiModalDataBuiltins,
                           Mapping[str, MultiModalData[object]]]
"""
A dictionary containing an item for each modality type to input.

The data belonging to each modality is converted into keyword arguments
to the model by the corresponding mapper. By default, the mapper of
the corresponding plugin with the same modality key is applied.
"""

MultiModalInputMapper = Callable[[InputContext, MultiModalData[object]],
                                 MultiModalInputs]
"""
Return a dictionary to be passed as keyword arguments to
:meth:`~torch.nn.Module.forward`. This is similar in concept to tokenizers
and processors in HuggingFace Transformers.
If the data is not supported, throw :exc:`TypeError`.
"""

MultiModalTokensCalc = Union[int, Callable[[InputContext], int]]
"""
Calculate the maximum number of multimodal tokens input to the language
model. This does not include tokens that correspond to the input text.
"""

N = TypeVar("N", bound=Type[nn.Module])


class MultiModalPlugin(ABC):
    """
    Base class that defines data processing logic for a specific modality.

    In particular, we adopt a registry pattern to dispatch data processing
    according to the model being used (considering that different models may
    process the same data differently). This registry is in turn used by
    :class:`~MultiModalRegistry` which acts at a higher level
    (i.e., the modality of the data).
    """

    def __init__(self) -> None:
        self._input_mappers: Dict[Type[nn.Module], MultiModalInputMapper] = {}
        self._max_mm_tokens: Dict[Type[nn.Module], MultiModalTokensCalc] = {}

    @abstractmethod
    def get_data_key(self) -> str:
        """
        Get the data key corresponding to the modality.
        """
        raise NotImplementedError

    @abstractmethod
    def _default_input_mapper(
        self,
        ctx: InputContext,
        data: MultiModalData[object],
    ) -> MultiModalInputs:
        """
        Return a dictionary to be passed as keyword arguments to
        :meth:`~torch.nn.Module.forward`. This is similar in concept to
        tokenizers and processors in HuggingFace Transformers.
        If the data is not supported, throw :exc:`TypeError`.
        """
        raise NotImplementedError

    def register_input_mapper(
        self,
        mapper: Optional[MultiModalInputMapper] = None,
    ):
        """
        Register an input mapper to a model class.
        When the model receives input data that matches the modality served by
        this plugin (see :meth:`get_data_type`), the provided function is
        invoked to transform the data into a dictionary of model inputs.
        If `None` is provided, then the default input mapper is used instead.

        See also:
            :ref:`input_processing_pipeline`
            :ref:`adding_a_new_multimodal_model`
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._input_mappers:
                logger.warning(
                    f"Model class {model_cls} already has an input mapper "
                    f"registered to {self}. It is overwritten by the new one.")

            self._input_mappers[model_cls] = mapper \
                or self._default_input_mapper

            return model_cls

        return wrapper

    def map_input(self, model_config: ModelConfig,
                  data: MultiModalData[object]) -> MultiModalInputs:
        """
        Apply an input mapper to a data passed
        to the model, transforming the data into a dictionary of model inputs.

        If the data is not something that the mapper expects, throws TypeError.

        The model is identified by ``model_config``.

        See also:
            :ref:`adding_a_new_multimodal_model`
        """
        # Avoid circular import
        from aphrodite.modeling.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)

        mapper = self._input_mappers.get(model_cls)
        # Only get processor kwargs at mapping time if we are not using the
        # input mapper; no overrides are used on the default here because they
        # should be passed to the huggingface resource at initialization time.
        if mapper is not None and mapper != self._default_input_mapper:
            mm_processor_kwargs = get_allowed_kwarg_only_overrides(
                mapper, overrides=model_config.mm_processor_kwargs)
        else:
            mm_processor_kwargs = {}
        if mapper is None:
            raise KeyError(f"No input mapper in {self} is registered for "
                           f"model class {model_cls.__name__}.")

        return mapper(InputContext(model_config), data, **mm_processor_kwargs)

    @abstractmethod
    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        """
        Calculate the maximum number of tokens, corresponding to a single
        instance of multimodal data, that are passed to the language model.
        """
        raise NotImplementedError

    def _validate_max_multimodal_tokens(self, max_mm_tokens: int):
        if max_mm_tokens < 1:
            raise ValueError("You should set the number of tokens to a "
                             f"positive integer. Found: {max_mm_tokens}")

    def register_max_multimodal_tokens(
        self,
        max_mm_tokens: Optional[MultiModalTokensCalc] = None,
    ):
        """
        Register the maximum number of tokens, corresponding to a single
        instance of multimodal data, that are passed to the language model
        for a model class.
        If `None` is provided, then the default calculation is used instead.
        See also:
            :ref:`adding_a_new_multimodal_model`
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._max_mm_tokens:
                logger.warning(
                    f"Model class {model_cls} already calculates maximum "
                    f"number of tokens in {self}. It is overwritten by the "
                    "new one.")

            if isinstance(max_mm_tokens, int):
                self._validate_max_multimodal_tokens(max_mm_tokens)

            self._max_mm_tokens[model_cls] = max_mm_tokens \
                or self._default_max_multimodal_tokens

            return model_cls

        return wrapper

    def get_max_multimodal_tokens(self, model_config: ModelConfig) -> int:
        """
        Get the maximum number of multi-modal tokens
        for profiling the memory usage of a model.
        If this registry is not applicable to the model, `0` is returned.
        The model is identified by ``model_config``.
        See also:
            :ref:`adding_a_new_multimodal_model`
        """
        # Avoid circular import
        from aphrodite.modeling.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)

        if model_cls not in self._input_mappers:
            return 0

        max_mm_tokens = self._max_mm_tokens.get(model_cls)
        if max_mm_tokens is None:
            raise KeyError(f"No maximum number of multi-modal tokens is given "
                           f"for model class {model_cls.__name__} in {self}.")

        if callable(max_mm_tokens):
            mm_processor_kwargs = get_allowed_kwarg_only_overrides(
                max_mm_tokens, overrides=model_config.mm_processor_kwargs)
            max_mm_tokens = max_mm_tokens(InputContext(model_config),
                                          **mm_processor_kwargs)

        self._validate_max_multimodal_tokens(max_mm_tokens)

        return max_mm_tokens
