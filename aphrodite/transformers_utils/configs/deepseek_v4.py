# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
