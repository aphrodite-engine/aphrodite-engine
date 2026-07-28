# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
"""Fuser detection for the Transformers modeling backend.

`get_fuser` traces a module class once (see `fx_utils`) and matches it against
each concrete fuser in `fusers`; `Fusers` caches the result per class for a
whole model. `base.recursive_replace` then applies the matched fuser per
instance. RMSNorm-shaped modules the tracer cannot match are warned about.
"""

from collections import UserDict
from typing import TYPE_CHECKING

from cachetools import cached
from torch import nn

from aphrodite.logger import init_logger
from aphrodite.model_executor.models.transformers.fusers import (
    BaseFuser,
    GLUFuser,
    PackedQKVFuser,
    QKVFuser,
    RewriteFuser,
    RMSNormFuser,
)
from aphrodite.model_executor.models.transformers.fx_utils import trace

if TYPE_CHECKING:
    from aphrodite.config import AphroditeConfig

logger = init_logger(__name__)


def key(module: nn.Module) -> tuple:
    """Cache key for `get_fuser`. Considers module type and its immediate children."""
    return (type(module), tuple(name for name, _ in module.named_children()))


@cached(cache={}, key=key)
def get_fuser(module: nn.Module) -> BaseFuser | None:
    """The fuser for `module`'s class and shape (cached), or `None` if no match."""
    # Projection fusions need >=2 sibling linears; the RMSNorm fusion needs a
    # leaf module (raw tensor math, no submodules). Nothing else can match, and
    # tracing is skipped for it.
    n_linear = sum(isinstance(c, nn.Linear) for c in module.children())
    is_leaf = next(module.children(), None) is None
    if n_linear < 2 and not is_leaf:
        return None
    if (graph := trace(module)) is None:
        return None
    for fuser_cls in (GLUFuser, QKVFuser, PackedQKVFuser, RMSNormFuser):
        if (fuser := fuser_cls.match(graph, module)) is not None:
            if isinstance(fuser, RewriteFuser):
                try:
                    fuser.update_forward(module)
                except Exception as exc:
                    logger.debug(
                        "Attempted to fuse %s using %s but failed to update its forward method: %s",
                        type(module),
                        fuser_cls.__name__,
                        exc,
                    )
                    continue
            return fuser
    # A norm we could not match structurally is left unfused; flag likely misses.
    if module.__class__.__name__.endswith("RMSNorm"):
        logger.warning_once(
            "%s looks like an RMSNorm but its computation did not match the expected pattern, so it was left unfused.",
            module.__class__.__name__,
        )
    return None


class Fusers(UserDict):
    """Mapping from module class to fuser, for all fusable classes in a model."""

    def __init__(self, model: nn.Module, aphrodite_config: "AphroditeConfig"):
        self.aphrodite_config = aphrodite_config
        super().__init__({type(m): get_fuser(m) for m in model.modules()})

    def __getitem__(self, m: nn.Module) -> BaseFuser | None:
        fuser = self.data.get(type(m))
        if fuser is not None and fuser.validate(m, self.aphrodite_config):
            return fuser
        return None
