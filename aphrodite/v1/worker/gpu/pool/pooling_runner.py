# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from aphrodite.model_executor.models import AphroditeModelForPooling, is_pooling_model
from aphrodite.tasks import PoolingTask
from aphrodite.v1.worker.gpu.input_batch import InputBatch
from aphrodite.v1.worker.gpu.states import RequestState


# NOTE(woosuk): Currently, this class only supports the "LAST" pooling task
# on decoder-only models. How to support other pooling tasks and models
# is to be determined.
class PoolingRunner:
    def __init__(self, model: nn.Module):
        self.model = cast(AphroditeModelForPooling, model)

    @staticmethod
    def get_supported_tasks(model: nn.Module) -> list[PoolingTask]:
        if not is_pooling_model(model):
            return []
        assert "embed" in model.pooler.get_supported_tasks()
        return ["embed"]

    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO(woosuk): Support different types of pooling tasks.
        last_hidden_states = hidden_states[input_batch.logits_indices]
        # TODO(woosuk): Make normalization optional.
        last_hidden_states = F.normalize(last_hidden_states, p=2, dim=-1)

        prompt_len = req_states.prompt_len.gpu[input_batch.idx_mapping]
        is_valid = input_batch.seq_lens == prompt_len
        return last_hidden_states, is_valid

    def dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        F.normalize(hidden_states, p=2, dim=-1)
        return
