# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from aphrodite.config import AphroditeConfig
from aphrodite.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from aphrodite.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


class MTPSpeculator(AutoRegressiveSpeculator):
    def __init__(self, aphrodite_config: AphroditeConfig, device: torch.device):
        super().__init__(aphrodite_config, device)

        spec_config = aphrodite_config.speculative_config
        draft_hf_config = spec_config.draft_model_config.hf_config if spec_config is not None else None
        self.share_mtp_topk_indices = getattr(draft_hf_config, "index_share_for_mtp_iteration", False)

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.aphrodite_config)
        self.share_mtp_topk_indices = self.share_mtp_topk_indices and hasattr(draft_model.model, "set_skip_topk")
        return draft_model

    def on_prefill_end(self, num_reqs: int, num_tokens: int) -> None:
        if self.share_mtp_topk_indices and self.num_speculative_steps > 1:
            self.model.model.compact_topk_indices(self.last_token_indices[:num_reqs])

    def on_multi_step_decode_begin(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(True)

    def on_multi_step_decode_end(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)
