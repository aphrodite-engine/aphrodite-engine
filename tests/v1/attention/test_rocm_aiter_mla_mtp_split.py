# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from aphrodite.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm AITER MLA tests", allow_module_level=True)

from aphrodite.v1.attention.backends.mla import rocm_aiter_mla  # noqa: E402
from aphrodite.v1.attention.backends.mla.rocm_aiter_mla import (  # noqa: E402
    AiterMLAMetadataBuilder,
)


class _NoOpTritonKernel:
    def __getitem__(self, grid):
        self.grid = grid
        return self

    def __call__(self, *args, **kwargs):
        pass


def _builder(
    *,
    mtp_decode_qlen: int,
    has_full_cudagraphs: bool = False,
    num_heads: int = 16,
):
    max_decode_rows = 32
    return SimpleNamespace(
        device=torch.device("cpu"),
        num_heads=num_heads,
        paged_kv_last_page_len=torch.ones(max_decode_rows, dtype=torch.int32),
        paged_kv_indices=torch.empty(1024, dtype=torch.int32),
        paged_kv_indptr=torch.empty(max_decode_rows + 1, dtype=torch.int32),
        qo_indptr=torch.empty(max_decode_rows + 1, dtype=torch.int32),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: has_full_cudagraphs)
        ),
        _mtp_decode_qlen=mtp_decode_qlen,
        _uniform_padded_mtp_qo_len=AiterMLAMetadataBuilder._uniform_padded_mtp_qo_len,
        kernel_block_size=1,
        _num_attention_heads=16,
        _mla_work_meta_data=torch.empty(1, dtype=torch.int32),
        _mla_work_info_set=torch.empty(1, dtype=torch.int32),
        _mla_work_indptr=torch.empty(1, dtype=torch.int32),
        _mla_reduce_indptr=torch.empty(1, dtype=torch.int32),
        _mla_reduce_final_map=torch.empty(1, dtype=torch.int32),
        _mla_reduce_partial_map=torch.empty(1, dtype=torch.int32),
        _mla_q_dtype=torch.bfloat16,
        _mla_kv_dtype=torch.bfloat16,
        decode_attn_out_dtype=torch.bfloat16,
    )


def test_mtp_decode_qlen4_keeps_uniform_rows_with_metadata(monkeypatch):
    get_mla_metadata_v1 = mock.MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(get_mla_metadata_v1=get_mla_metadata_v1),
    )
    monkeypatch.setattr(rocm_aiter_mla, "_expand_page_indices_kernel", _NoOpTritonKernel())

    metadata = AiterMLAMetadataBuilder._build_decode(
        _builder(mtp_decode_qlen=4),
        block_table_tensor=torch.arange(16, dtype=torch.int32).view(2, 8),
        seq_lens_device=torch.tensor([7, 5], dtype=torch.int32),
        max_seq_len=7,
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        query_start_loc_device=torch.tensor([0, 4, 8], dtype=torch.int32),
        num_decode_tokens=8,
        dcp_tot_seq_lens_device=None,
    )

    assert metadata.max_qo_len == 4
    assert torch.equal(metadata.qo_indptr, torch.tensor([0, 4, 8], dtype=torch.int32))
    assert metadata.has_persistent_metadata
    assert get_mla_metadata_v1.call_args.kwargs["max_seqlen_qo"] == 4
    assert get_mla_metadata_v1.call_args.kwargs["uni_seqlen_qo"] == 4


def test_full_cudagraph_padded_uniform_mtp_synthesizes_decode_indptr(
    monkeypatch,
):
    get_mla_metadata_v1 = mock.MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(get_mla_metadata_v1=get_mla_metadata_v1),
    )
    monkeypatch.setattr(rocm_aiter_mla, "_expand_page_indices_kernel", _NoOpTritonKernel())

    mtp_qlen = 4
    seq_lens = torch.tensor([7, 0], dtype=torch.int32)
    builder = _builder(mtp_decode_qlen=mtp_qlen, has_full_cudagraphs=True)
    metadata = AiterMLAMetadataBuilder._build_decode(
        builder,
        block_table_tensor=torch.arange(16, dtype=torch.int32).view(2, 8),
        seq_lens_device=seq_lens,
        max_seq_len=7,
        query_start_loc_cpu=torch.tensor([0, mtp_qlen, mtp_qlen], dtype=torch.int32),
        query_start_loc_device=torch.tensor([0, mtp_qlen, mtp_qlen], dtype=torch.int32),
        num_decode_tokens=seq_lens.numel() * mtp_qlen,
        dcp_tot_seq_lens_device=None,
    )

    assert torch.equal(metadata.seq_lens, torch.tensor([7, 4], dtype=torch.int32))
    assert torch.equal(metadata.paged_kv_indptr, torch.tensor([0, 7, 11], dtype=torch.int32))
    assert torch.equal(metadata.qo_indptr, torch.tensor([0, 4, 8], dtype=torch.int32))
    assert metadata.has_persistent_metadata


@pytest.mark.parametrize(
    "mtp_decode_qlen, qo_len, num_heads, expect_persistent",
    [
        (1, 1, 16, True),
        (4, 2, 16, True),
        (4, 4, 16, True),
        (2, 4, 16, False),
        (4, 4, 8, False),
    ],
)
def test_persistent_metadata_gate(monkeypatch, mtp_decode_qlen, qo_len, num_heads, expect_persistent):
    get_mla_metadata_v1 = mock.MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(get_mla_metadata_v1=get_mla_metadata_v1),
    )
    monkeypatch.setattr(rocm_aiter_mla, "_expand_page_indices_kernel", _NoOpTritonKernel())
    query_start_loc = torch.arange(0, 3 * qo_len, step=qo_len, dtype=torch.int32)

    metadata = AiterMLAMetadataBuilder._build_decode(
        _builder(mtp_decode_qlen=mtp_decode_qlen, num_heads=num_heads),
        block_table_tensor=torch.arange(16, dtype=torch.int32).view(2, 8),
        seq_lens_device=torch.tensor([8, 8], dtype=torch.int32),
        max_seq_len=8,
        query_start_loc_cpu=query_start_loc,
        query_start_loc_device=query_start_loc,
        num_decode_tokens=2 * qo_len,
        dcp_tot_seq_lens_device=None,
    )

    assert metadata.has_persistent_metadata is expect_persistent
    assert get_mla_metadata_v1.called is expect_persistent
