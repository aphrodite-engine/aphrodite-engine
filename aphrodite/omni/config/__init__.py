# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Configuration module for Sonar Omni.
"""

from aphrodite.omni.config.lora import LoRAConfig
from aphrodite.omni.config.model import OmniModelConfig
from aphrodite.omni.config.omni_config import (
    AphroditeOmniARStageConfig,
    AphroditeOmniConfig,
    AphroditeOmniDiffusionStageConfig,
    AphroditeOmniGenerationStageConfig,
    AphroditeOmniOrchestratorConfig,
    BaseAphroditeOmniStageConfig,
    OmniStageCacheConfig,
    OmniStageConnectorConfig,
    OmniStageDiffusionParallelConfig,
    OmniStageLoadConfig,
    OmniStageModelConfig,
    OmniStageParallelConfig,
    OmniStageRuntimeConfig,
    OmniStageSchedulerConfig,
    StageConfigType,
)
from aphrodite.omni.config.stage_config import (
    PIPELINE_WIDE_ENGINE_FIELDS,
    DeployConfig,
    PipelineConfig,
    StageConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    StageType,
    load_deploy_config,
    merge_pipeline_deploy,
)
from aphrodite.omni.config.yaml_util import (
    create_config,
    load_yaml_config,
    merge_configs,
    to_dict,
)

# StageConfigFactory / register_pipeline pull pipeline_registry, which eagerly
# imports PI0_PIPELINE → diffusion.data. Keep those lazy so
# `from aphrodite.omni.config.lora import LoRAConfig` (used while data.py is still
# loading) cannot close a circular import through DiffusionOutput.
_LAZY_ATTRS = {
    "OmniConfigResolution": ("aphrodite.omni.config.resolver", "OmniConfigResolution"),
    "StageConfigFactory": ("aphrodite.omni.config.config_factory", "StageConfigFactory"),
    "register_pipeline": ("aphrodite.omni.config.pipeline_registry", "register_pipeline"),
    "resolve_omni_config": ("aphrodite.omni.config.resolver", "resolve_omni_config"),
}

__all__ = [
    # Legacy model-level configs.
    "LoRAConfig",
    "OmniModelConfig",
    # Structured Omni config entry points.
    "AphroditeOmniConfig",
    "BaseAphroditeOmniStageConfig",
    "AphroditeOmniARStageConfig",
    "AphroditeOmniGenerationStageConfig",
    "AphroditeOmniDiffusionStageConfig",
    "StageConfigType",
    "OmniConfigResolution",
    "resolve_omni_config",
    # Structured Omni sub-configs.
    "OmniStageCacheConfig",
    "OmniStageConnectorConfig",
    "OmniStageDiffusionParallelConfig",
    "OmniStageLoadConfig",
    "OmniStageModelConfig",
    "AphroditeOmniOrchestratorConfig",
    "OmniStageParallelConfig",
    "OmniStageRuntimeConfig",
    "OmniStageSchedulerConfig",
    # Legacy pipeline/stage deploy config surface.
    "PIPELINE_WIDE_ENGINE_FIELDS",
    "DeployConfig",
    "PipelineConfig",
    "StageConfig",
    "StageConfigFactory",
    "StageDeployConfig",
    "StageType",
    "StageExecutionType",
    "StagePipelineConfig",
    "load_deploy_config",
    "merge_pipeline_deploy",
    "register_pipeline",
    # YAML utility helpers.
    "create_config",
    "load_yaml_config",
    "merge_configs",
    "to_dict",
]


def __getattr__(name: str):
    spec = _LAZY_ATTRS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = spec
    import importlib

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
