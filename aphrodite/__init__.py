"""Aphrodite Engine: Large-scale LLM inference engine"""

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

import typing

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import aphrodite.env_override  # noqa: F401

MODULE_ATTRS = {
    "bc_linter_skip": "._bc_linter:bc_linter_skip",
    "bc_linter_include": "._bc_linter:bc_linter_include",
    "AsyncEngineArgs": ".engine.args_tools:AsyncEngineArgs",
    "EngineArgs": ".engine.args_tools:EngineArgs",
    "AsyncAphrodite": ".engine.async_aphrodite:AsyncAphrodite",
    "AphroditeEngine": ".engine.aphrodite_engine:AphroditeEngine",
    "LLM": ".endpoints.llm:LLM",
    "initialize_ray_cluster": ".v1.executor.ray_utils:initialize_ray_cluster",
    "PromptType": ".inputs:PromptType",
    "TextPrompt": ".inputs:TextPrompt",
    "TokensPrompt": ".inputs:TokensPrompt",
    "ModelRegistry": ".modeling.models:ModelRegistry",
    "SamplingParams": ".common.sampling_params:SamplingParams",
    "PoolingParams": ".common.pooling_params:PoolingParams",
    "ClassificationOutput": ".outputs:ClassificationOutput",
    "ClassificationRequestOutput": ".outputs:ClassificationRequestOutput",
    "CompletionOutput": ".outputs:CompletionOutput",
    "EmbeddingOutput": ".outputs:EmbeddingOutput",
    "EmbeddingRequestOutput": ".outputs:EmbeddingRequestOutput",
    "PoolingOutput": ".outputs:PoolingOutput",
    "PoolingRequestOutput": ".outputs:PoolingRequestOutput",
    "RequestOutput": ".outputs:RequestOutput",
    "ScoringOutput": ".outputs:ScoringOutput",
    "ScoringRequestOutput": ".outputs:ScoringRequestOutput",
}

if typing.TYPE_CHECKING:
    from aphrodite.common.pooling_params import PoolingParams
    from aphrodite.common.sampling_params import SamplingParams
    from aphrodite.endpoints.llm import LLM
    from aphrodite.engine.aphrodite_engine import AphroditeEngine
    from aphrodite.engine.args_tools import AsyncEngineArgs, EngineArgs
    from aphrodite.engine.async_aphrodite import AsyncAphrodite
    from aphrodite.inputs import PromptType, TextPrompt, TokensPrompt
    from aphrodite.modeling.models import ModelRegistry
    from aphrodite.outputs import (ClassificationOutput,
                                   ClassificationRequestOutput,
                                   CompletionOutput, EmbeddingOutput,
                                   EmbeddingRequestOutput, PoolingOutput,
                                   PoolingRequestOutput, RequestOutput,
                                   ScoringOutput, ScoringRequestOutput)
    from aphrodite.v1.executor.ray_utils import initialize_ray_cluster

    from ._bc_linter import bc_linter_include, bc_linter_skip
else:

    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        else:
            raise AttributeError(
                f'module {__package__} has no attribute {name}')


__all__ = [
    "__version__",
    "__version_tuple__",
    "bc_linter_skip",
    "bc_linter_include",
    "LLM",
    "ModelRegistry",
    "PromptType",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "PoolingOutput",
    "PoolingRequestOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "ClassificationOutput",
    "ClassificationRequestOutput",
    "ScoringOutput",
    "ScoringRequestOutput",
    "AphroditeEngine",
    "EngineArgs",
    "AsyncAphrodite",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]
