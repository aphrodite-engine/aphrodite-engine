# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

import torch

from aphrodite.omni.data_entry_keys import OmniPayload
from aphrodite.sequence import IntermediateTensors


class OmniOutput(NamedTuple):
    """Output from the merged Omni model containing both text and audio."""

    text_hidden_states: torch.Tensor
    multimodal_outputs: OmniPayload | None = None
    intermediate_tensors: IntermediateTensors | None = None
    next_token_id: torch.Tensor | None = None
