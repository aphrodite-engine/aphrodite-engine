# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from aphrodite.v1.worker.gpu.spec_decode.rejection_sampler import (
    _iter_request_chunks,
)


def test_iter_request_chunks_preserves_request_boundaries():
    cu_num_logits = np.array([0, 3, 4, 11, 13], dtype=np.int32)

    assert list(_iter_request_chunks(cu_num_logits, max_chunk_logits=5)) == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]
