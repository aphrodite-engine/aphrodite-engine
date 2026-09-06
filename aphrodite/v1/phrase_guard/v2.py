# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from aphrodite.triton_utils import tl, triton
from aphrodite.v1.phrase_guard.matcher import MAX_RETRIES
from aphrodite.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


class RetryMask:
    def __init__(self, max_num_reqs, device):
        self.positions = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.counts = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.tokens = StagedWriteTensor((max_num_reqs, MAX_RETRIES), dtype=torch.int32, device=device)

    def add_request(self, index, prompt_len, retry):
        self.counts.np[index] = 0
        if retry is not None:
            position, tokens = retry
            self.positions.np[index] = prompt_len + position - 1
            self.counts.np[index] = len(tokens)
            self.tokens.stage_write(index, 0, tokens)

    def apply_staged_writes(self):
        self.positions.copy_to_uva()
        self.counts.copy_to_uva()
        self.tokens.apply_write()

    def apply(self, logits, expanded_idx_mapping, idx_mapping_np, positions):
        if not np.any(self.counts.np[idx_mapping_np]):
            return
        _mask[(logits.shape[0],)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            positions,
            self.positions.gpu,
            self.counts.gpu,
            self.tokens.gpu,
            WIDTH=MAX_RETRIES,
        )


@triton.jit
def _mask(logits, stride, mapping, positions, checkpoints, counts, tokens, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    request = tl.load(mapping + row)
    if tl.load(positions + row) == tl.load(checkpoints + request):
        offsets = tl.arange(0, WIDTH)
        valid = offsets < tl.load(counts + request)
        ids = tl.load(tokens + request * WIDTH + offsets, valid, other=0)
        tl.store(logits + row * stride + ids, float("-inf"), valid)
