from typing import Literal, Optional

from torch import nn

from aphrodite.config import LoadConfig, ModelConfig, AphroditeConfig
from loguru import logger
from aphrodite.modeling.model_loader.base_loader import BaseModelLoader
from aphrodite.modeling.model_loader.bitsandbytes_loader import (
    BitsAndBytesModelLoader)
from aphrodite.modeling.model_loader.default_loader import DefaultModelLoader
from aphrodite.modeling.model_loader.dummy_loader import DummyModelLoader
from aphrodite.modeling.model_loader.gguf_loader import GGUFModelLoader
from aphrodite.modeling.model_loader.runai_streamer_loader import (
    RunaiModelStreamerLoader)
from aphrodite.modeling.model_loader.sharded_state_loader import (
    ShardedStateLoader)
from aphrodite.modeling.model_loader.tensorizer_loader import TensorizerLoader
from aphrodite.modeling.model_loader.utils import (
    get_architecture_class_name, get_model_architecture, get_model_cls)

# Reminder: Please update docstring in `LoadConfig`
# if a new load format is added here
LoadFormats = Literal[
    "auto",
    "bitsandbytes",
    "dummy",
    "fastsafetensors",
    "gguf",
    "mistral",
    "npcache",
    "pt",
    "runai_streamer",
    "runai_streamer_sharded",
    "safetensors",
    "sharded_state",
    "tensorizer",
]
_LOAD_FORMAT_TO_MODEL_LOADER: dict[str, type[BaseModelLoader]] = {
    "auto": DefaultModelLoader,
    "bitsandbytes": BitsAndBytesModelLoader,
    "dummy": DummyModelLoader,
    "fastsafetensors": DefaultModelLoader,
    "gguf": GGUFModelLoader,
    "mistral": DefaultModelLoader,
    "npcache": DefaultModelLoader,
    "pt": DefaultModelLoader,
    "runai_streamer": RunaiModelStreamerLoader,
    "runai_streamer_sharded": ShardedStateLoader,
    "safetensors": DefaultModelLoader,
    "sharded_state": ShardedStateLoader,
    "tensorizer": TensorizerLoader,
}


def register_model_loader(load_format: str):
    """Register a customized aphrodite model loader.

    When a load format is not supported by aphrodite, you can register a customized
    model loader to support it.

    Args:
        load_format (str): The model loader format name.

    Examples:
        >>> from aphrodite.config import LoadConfig
        >>> from aphrodite.modeling.model_loader import get_model_loader, register_model_loader
        >>> from aphrodite.modeling.model_loader.base_loader import BaseModelLoader
        >>>
        >>> @register_model_loader("my_loader")
        ... class MyModelLoader(BaseModelLoader):
        ...     def download_model(self):
        ...         pass
        ...
        ...     def load_weights(self):
        ...         pass
        >>>
        >>> load_config = LoadConfig(load_format="my_loader")
        >>> type(get_model_loader(load_config))
        <class 'MyModelLoader'>
    """  # noqa: E501

    def _wrapper(model_loader_cls):
        if load_format in _LOAD_FORMAT_TO_MODEL_LOADER:
            logger.warning(
                "Load format `{}` is already registered, and will be "
                "overwritten by the new loader class `{}`.", load_format,
                model_loader_cls)
        if not issubclass(model_loader_cls, BaseModelLoader):
            raise ValueError("The model loader must be a subclass of "
                             "`BaseModelLoader`.")
        _LOAD_FORMAT_TO_MODEL_LOADER[load_format] = model_loader_cls
        logger.info("Registered model loader `{}` with load format `{}`",
                    model_loader_cls, load_format)
        return model_loader_cls

    return _wrapper


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""
    load_format = load_config.load_format
    if load_format not in _LOAD_FORMAT_TO_MODEL_LOADER:
        raise ValueError(f"Load format `{load_format}` is not supported")
    return _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)


def get_model(*,
              aphrodite_config: AphroditeConfig,
              model_config: Optional[ModelConfig] = None) -> nn.Module:
    loader = get_model_loader(aphrodite_config.load_config)
    if model_config is None:
        model_config = aphrodite_config.model_config
    return loader.load_model(aphrodite_config=aphrodite_config,
                             model_config=model_config)


__all__ = [
    "get_model",
    "get_model_loader",
    "get_architecture_class_name",
    "get_model_architecture",
    "get_model_cls",
    "register_model_loader",
    "BaseModelLoader",
    "BitsAndBytesModelLoader",
    "GGUFModelLoader",
    "DefaultModelLoader",
    "DummyModelLoader",
    "RunaiModelStreamerLoader",
    "ShardedStateLoader",
    "TensorizerLoader",
]
