"""
Model configs may be defined in this directory for the following reasons:

- There is no configuration file defined by HF Hub or Transformers library.
- There is a need to override the existing config to support vLLM.
"""

from aphrodite.transformers_utils.configs.chatglm import ChatGLMConfig
from aphrodite.transformers_utils.configs.deepseek_vl2 import (
    DeepseekVLV2Config)
from aphrodite.transformers_utils.configs.eagle import EAGLEConfig
# RWConfig is for the original tiiuae/falcon-40b(-instruct) and
# tiiuae/falcon-7b(-instruct) models. Newer Falcon models will use the
# `FalconConfig` class from the official HuggingFace transformers library.
from aphrodite.transformers_utils.configs.falcon import RWConfig
from aphrodite.transformers_utils.configs.jais import JAISConfig
from aphrodite.transformers_utils.configs.kimi_vl import KimiVLConfig
from aphrodite.transformers_utils.configs.medusa import MedusaConfig
from aphrodite.transformers_utils.configs.mlp_speculator import (
    MLPSpeculatorConfig)
from aphrodite.transformers_utils.configs.moonvit import MoonViTConfig
from aphrodite.transformers_utils.configs.nemotron import NemotronConfig
from aphrodite.transformers_utils.configs.nemotron_h import NemotronHConfig
from aphrodite.transformers_utils.configs.nemotron_vl import (
    Nemotron_Nano_VL_Config)
from aphrodite.transformers_utils.configs.ovis import OvisConfig
from aphrodite.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from aphrodite.transformers_utils.configs.speculators.base import (
    SpeculatorsConfig)
from aphrodite.transformers_utils.configs.step3_vl import (
    Step3TextConfig, Step3VisionEncoderConfig, Step3VLConfig)
from aphrodite.transformers_utils.configs.ultravox import UltravoxConfig

__all__ = [
    "ChatGLMConfig",
    "DeepseekVLV2Config",
    "EAGLEConfig",
    "RWConfig",
    "JAISConfig",
    "MedusaConfig",
    "MLPSpeculatorConfig",
    "MoonViTConfig",
    "KimiVLConfig",
    "NemotronConfig",
    "NemotronHConfig",
    "Nemotron_Nano_VL_Config",
    "OvisConfig",
    "SpeculatorsConfig",
    "UltravoxConfig",
    "Step3VLConfig",
    "Step3VisionEncoderConfig",
    "Step3TextConfig",
    "Qwen3NextConfig",
]
