# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# for now, it suffices to use Sonar's implementation directly
# as this is a user-facing variable, defined here to so that user can directly import LoRAConfig from aphrodite.omni
from aphrodite.config.lora import LoRAConfig

__all__ = ["LoRAConfig"]
