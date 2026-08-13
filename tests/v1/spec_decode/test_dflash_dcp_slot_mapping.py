# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP-aware slot mapping for DFlash and DSpark draft inputs."""

from types import SimpleNamespace

import pytest
import torch

from aphrodite.platforms import current_platform
from aphrodite.v1.attention.backends.utils import PAD_SLOT_ID
from aphrodite.v1.worker.gpu.input_batch import InputBuffers
from aphrodite.v1.worker.gpu.spec_decode.dflash.speculator import (
    prepare_dflash_inputs,
)

pytest.importorskip("triton")
if not current_platform.is_cuda_alike():
    pytest.skip("CUDA required for DFlash kernel tests", allow_module_level=True)

DEVICE = "cuda"
BLOCK_SIZE = 16
NUM_SPECULATIVE_STEPS = 3
NUM_QUERY_PER_REQ = 1 + NUM_SPECULATIVE_STEPS
MAX_NUM_REQS = 8
MAX_NUM_TOKENS = 512
MAX_NUM_BLOCKS = 64
MAX_MODEL_LEN = 4096


def _ref_slots(
    positions: torch.Tensor,
    block_row: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    cp_interleave: int,
) -> torch.Tensor:
    virtual_block = BLOCK_SIZE * cp_size
    block_indices = (positions // virtual_block).clamp(max=block_row.shape[0] - 1)
    block_offsets = positions % virtual_block
    block_ids = block_row[block_indices].long()
    if cp_size == 1:
        return block_ids * BLOCK_SIZE + block_offsets
    is_local = (block_offsets // cp_interleave) % cp_size == cp_rank
    local_offsets = block_offsets // (cp_interleave * cp_size) * cp_interleave + block_offsets % cp_interleave
    slots = block_ids * BLOCK_SIZE + local_offsets
    return torch.where(is_local, slots, torch.full_like(slots, PAD_SLOT_ID))


def _run_prepare(cp_size: int, cp_rank: int, cp_interleave: int):
    context_positions = [
        torch.arange(100, 120, device=DEVICE),
        torch.arange(0, 1, device=DEVICE),
        torch.arange(37, 100, device=DEVICE),
    ]
    num_rejected = torch.tensor([2, 0, 0], dtype=torch.int32, device=DEVICE)
    num_sampled = torch.tensor([1, 1, 0], dtype=torch.int32, device=DEVICE)
    num_reqs = len(context_positions)

    positions = torch.cat(context_positions)
    num_context = torch.tensor([len(p) for p in context_positions], device=DEVICE)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    query_start_loc[1:] = num_context.cumsum(0)
    num_tokens = int(query_start_loc[-1])

    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_scheduled_tokens=num_context.cpu().numpy(),
        positions=positions,
        query_start_loc=query_start_loc,
        idx_mapping=torch.arange(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE),
    )
    input_buffers = InputBuffers(MAX_NUM_REQS, MAX_NUM_TOKENS, torch.device(DEVICE))
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    block_table = (
        torch.randperm(
            MAX_NUM_REQS * MAX_NUM_BLOCKS,
            generator=generator,
            device=DEVICE,
        )
        .view(MAX_NUM_REQS, MAX_NUM_BLOCKS)
        .to(torch.int32)
    )

    query_slot_mapping = torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64, device=DEVICE)
    output_context_positions = torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64, device=DEVICE)
    context_slot_mapping = torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64, device=DEVICE)
    max_num_sampled = MAX_NUM_REQS * NUM_SPECULATIVE_STEPS
    sample_indices = torch.zeros(max_num_sampled, dtype=torch.int64, device=DEVICE)
    sample_pos = torch.zeros(max_num_sampled, dtype=torch.int64, device=DEVICE)
    sample_idx_mapping = torch.zeros(max_num_sampled, dtype=torch.int32, device=DEVICE)
    temperature = torch.zeros(MAX_NUM_REQS, dtype=torch.float32, device=DEVICE)
    seeds = torch.zeros(MAX_NUM_REQS, dtype=torch.int64, device=DEVICE)
    last_sampled = torch.full((MAX_NUM_REQS,), 7, dtype=torch.int64, device=DEVICE)
    next_prefill_tokens = torch.full((MAX_NUM_REQS,), 11, dtype=torch.int64, device=DEVICE)

    prepare_dflash_inputs(
        input_buffers,
        query_slot_mapping,
        output_context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        block_table,
        BLOCK_SIZE,
        torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE),
        parallel_drafting_token_id=1,
        num_query_per_req=NUM_QUERY_PER_REQ,
        num_speculative_steps=NUM_SPECULATIVE_STEPS,
        max_num_reqs=MAX_NUM_REQS,
        max_num_tokens=MAX_NUM_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_interleave=cp_interleave,
    )
    return SimpleNamespace(
        context_positions=context_positions,
        num_rejected=num_rejected,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        query_start_loc=query_start_loc,
        block_table=block_table,
        input_buffers=input_buffers,
        query_slot_mapping=query_slot_mapping,
        output_context_positions=output_context_positions,
        context_slot_mapping=context_slot_mapping,
    )


@pytest.mark.parametrize("cp_size,cp_interleave", [(1, 1), (2, 1), (4, 1), (2, 16), (4, 8)])
def test_dflash_slot_mapping_matches_dcp_layout(cp_size: int, cp_interleave: int):
    for cp_rank in range(cp_size):
        result = _run_prepare(cp_size, cp_rank, cp_interleave)
        for req in range(result.num_reqs):
            start = int(result.query_start_loc[req])
            end = int(result.query_start_loc[req + 1])
            context_positions = result.context_positions[req]
            expected_context_slots = _ref_slots(
                context_positions,
                result.block_table[req],
                cp_size,
                cp_rank,
                cp_interleave,
            )
            torch.testing.assert_close(
                result.context_slot_mapping[start:end],
                expected_context_slots,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                result.output_context_positions[start:end],
                context_positions,
                rtol=0,
                atol=0,
            )

            query_start = req * NUM_QUERY_PER_REQ
            query_slots = result.query_slot_mapping[query_start : query_start + NUM_QUERY_PER_REQ]
            if cp_size == 1:
                last_valid_pos = int(context_positions[-1 - int(result.num_rejected[req])])
                query_positions = torch.arange(
                    last_valid_pos + 1,
                    last_valid_pos + 1 + NUM_QUERY_PER_REQ,
                    device=DEVICE,
                )
                expected_query_slots = _ref_slots(
                    query_positions,
                    result.block_table[req],
                    cp_size,
                    cp_rank,
                    cp_interleave,
                )
                torch.testing.assert_close(query_slots, expected_query_slots, rtol=0, atol=0)
            else:
                assert (query_slots == PAD_SLOT_ID).all()


@pytest.mark.parametrize("cp_size,cp_interleave", [(2, 1), (4, 1), (4, 8)])
def test_dflash_context_slots_partition_across_dcp(cp_size: int, cp_interleave: int):
    per_rank = [_run_prepare(cp_size, rank, cp_interleave) for rank in range(cp_size)]
    for token_idx in range(per_rank[0].num_tokens):
        owners = sum(int(result.context_slot_mapping[token_idx] != PAD_SLOT_ID) for result in per_rank)
        assert owners == 1
