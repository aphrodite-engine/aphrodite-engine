# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from collections import namedtuple
from typing import Dict, Type

import torch.nn as nn
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      LlamaDecoderLayer,
                                                      LlamaForCausalLM,
                                                      LlamaModel, LlamaRMSNorm)

from aphrodite.qaic.custom_ops import CustomRMSNormAIC

from .models.llama.modeling_llama import (QEffLlamaAttention,
                                          QEffLlamaDecoderLayer,
                                          QEffLlamaForCausalLM, QEffLlamaModel)

# Define a named tuple for ModelArchitectures
# Required for the Automation tool
ModelArchitectures = namedtuple("ModelArchitectures", ["architectures"])

get_lists_of_cb_qeff_models = ModelArchitectures(
    [
        LlamaForCausalLM.__name__,
    ]
)
# Create an instance of the named tuple
qeff_supported_architectures = ModelArchitectures(
    [
        LlamaForCausalLM.__name__,
    ]
)

# Define a transformers layers to QEff layers dictionary
# While onboarding new models make sure to add the new layer maps to this
# dictionary.
TransformersToQEffModulesDict: Dict[Type[nn.Module], Type[nn.Module]] = {
    # Llama model layers
    LlamaModel: QEffLlamaModel,
    LlamaAttention: QEffLlamaAttention,
    LlamaForCausalLM: QEffLlamaForCausalLM,
    LlamaDecoderLayer: QEffLlamaDecoderLayer,
    LlamaRMSNorm: CustomRMSNormAIC,
}
