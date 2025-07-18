###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from functools import wraps
import os
from functools import lru_cache

import habana_frameworks.torch as htorch
import torch

from .cache_ops import insert_or_update_cache


@lru_cache(maxsize=None)
def is_fake_hpu() -> bool:
    return os.environ.get('APHRODITE_USE_FAKE_HPU', '0') != '0'


def with_mark_steps(fn):

    @wraps(fn)
    def wrapped(*args, **kwargs):
        htorch.core.mark_step()
        result = fn(*args, **kwargs)
        del args
        del kwargs
        htorch.core.mark_step()
        return result

    return wrapped


class Matmul(torch.nn.Module):

    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class Softmax(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dim=None, inv_head=None):
        return torch.softmax(x, dim)


class VLLMKVCache(torch.nn.Module):

    def __init__(self):
        super(VLLMKVCache, self).__init__()
        self.use_contiguous_pa = os.environ.get('APHRODITE_CONTIGUOUS_PA',
                                                'true').lower() == 'true'

    def forward(self, input, cache, block_indices, block_offset):
        # In cross-attention kv cache forward inputs are None in decode
        # We don't want to store them in the cache in such case
        if input is not None:
            insert_or_update_cache(input, cache, block_indices, block_offset)
        return cache

    def fetch_from_cache(self, cache, blocks):
        if self.use_contiguous_pa:
            return cache[:blocks.size(0)]
        else:
            return cache.index_select(0, blocks)


class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        assert fusedSDPA is not None, 'fusedSDPA kernel is None'
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
    ):
        return self._hpu_kernel_fsdpa.apply(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            softmax_mode,
            recompute_mode,
            valid_sequence_lengths,
            padding_side,
        )
