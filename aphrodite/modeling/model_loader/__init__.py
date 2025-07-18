from torch import nn

from aphrodite.common.config import AphroditeConfig
from aphrodite.modeling.model_loader.loader import (BaseModelLoader,
                                                    get_model_loader)
from aphrodite.modeling.model_loader.utils import (get_architecture_class_name,
                                                   get_model_architecture)


def get_model(*, aphrodite_config: AphroditeConfig) -> nn.Module:
    loader = get_model_loader(aphrodite_config.load_config)
    return loader.load_model(aphrodite_config=aphrodite_config)


__all__ = [
    "get_model", "get_model_loader", "BaseModelLoader",
    "get_architecture_class_name", "get_model_architecture"
]
