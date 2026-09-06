# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from aphrodite.sampling_params import SamplingParams
from aphrodite.v1.phrase_guard.matcher import MAX_PENDING_TOKENS, PendingText, validate_phrases
from aphrodite.v1.phrase_guard.processor import RETRY_KEY, PhraseRetryProcessor
from aphrodite.v1.phrase_guard.worker import restore_requests
from aphrodite.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality

pytestmark = pytest.mark.cpu_test


def test_partial_phrase_is_held_then_rewound():
    guard = PendingText(("a testament to",))
    assert guard.feed(1, "This is ").ready == [1]
    assert guard.feed(2, "a test").ready == []
    assert guard.feed(3, "ament ").ready == []
    result = guard.feed(4, "to")
    assert result.ready == []
    assert result.rewind_to == 1
    assert result.blocked_token == 2
    assert guard.feed(5, "excellent").ready == [5]


def test_diverging_prefix_releases_without_delay():
    guard = PendingText(("hello world",))
    assert guard.feed(1, "hello ").ready == []
    assert guard.feed(2, "there").ready == [1, 2]


def test_match_inside_token_rewinds_entire_token():
    guard = PendingText(("forbidden",))
    result = guard.feed(9, "a forbidden sentence")
    assert result.rewind_to == 0
    assert result.blocked_token == 9
    assert result.ready == []


def test_overlapping_earlier_partial_cannot_leak():
    guard = PendingText(("abcdef", "bc"))
    assert guard.feed(1, "a").ready == []
    result = guard.feed(2, "bc")
    assert result.ready == []
    assert result.rewind_to == 0
    assert result.blocked_token == 1


def test_rollback_restores_prefix_at_published_token_boundary():
    guard = PendingText(("ab", "cd"))
    assert guard.feed(1, "a").ready == []
    assert guard.feed(2, "xc").ready == [1]
    assert guard.feed(3, "d").rewind_to == 1
    result = guard.feed(4, "b")
    assert result.ready == []
    assert result.rewind_to == 1
    assert guard.feed(5, "z").ready == [5]


def test_unicode_spanning_tokens_and_casefold():
    guard = PendingText(("é",))
    assert guard.feed(1, "").ready == []
    result = guard.feed(2, "É")
    assert result.rewind_to == 0
    assert result.blocked_token == 1
    guard = PendingText(("straße",))
    assert guard.feed(1, "STRAS").ready == []
    assert guard.feed(2, "SE").rewind_to == 0


def test_case_sensitive_and_literal_regex_characters():
    guard = PendingText(("[a-z]+",), case_sensitive=True)
    assert guard.feed(1, "abc").ready == [1]
    assert guard.feed(2, "[A-Z]+").ready == [2]
    assert guard.feed(3, "[a-z]+").rewind_to == 2


def test_incomplete_prefix_can_finish():
    guard = PendingText(("hello world",))
    assert guard.feed(1, "hello").ready == []
    assert guard.finish() == [1]
    assert guard.finish() == []


def test_zero_width_buffer_is_bounded():
    guard = PendingText(("hello",))
    for _ in range(MAX_PENDING_TOKENS):
        assert guard.feed(1, "").ready == []
    with pytest.raises(ValueError, match="pending-token"):
        guard.feed(1, "")


@pytest.mark.parametrize("value", [None, [], "hello", [""], [3], ["x" * 257]])
def test_invalid_phrase_lists(value):
    with pytest.raises(ValueError):
        validate_phrases(value)


def test_retry_mask_moves_with_request_and_only_applies_at_checkpoint():
    processor = PhraseRetryProcessor(None, torch.device("cpu"), False)
    output = [7, 8]
    params = SamplingParams(extra_args={RETRY_KEY: (2, [3, 5])})
    processor.update_state(BatchUpdate(2, [], [(0, params, [1], output)], []))
    processor.update_state(BatchUpdate(2, [], [], [(0, 1, MoveDirectionality.SWAP)]))
    logits = processor.apply(torch.zeros(2, 10))
    assert torch.isfinite(logits[0]).all()
    assert torch.isneginf(logits[1, [3, 5]]).all()
    output.append(9)
    assert torch.isfinite(processor.apply(torch.zeros(2, 10))).all()
    processor.update_state(BatchUpdate(1, [1], [], []))
    assert not processor.states


def test_exhausted_retry_returns_blocked_sentinel_only_for_affected_row():
    processor = PhraseRetryProcessor(None, torch.device("cpu"), False)
    params = SamplingParams(extra_args={RETRY_KEY: (0, [3, 5])})
    processor.update_state(BatchUpdate(2, [], [(0, params, [1], [])], []))
    logits = torch.zeros(2, 10)
    logits[0] = -torch.inf
    logits[0, 5] = 2.0
    processor.apply(logits)
    assert logits[0].argmax().item() == 3
    assert torch.isfinite(logits[0]).sum().item() == 1
    assert not logits[1].count_nonzero().item()


def test_restore_does_not_change_other_request_or_rng():
    from unittest.mock import Mock

    states = {
        rid: SimpleNamespace(
            output_token_ids=[1, 2, 3],
            sampling_params=None,
            persistent_data={"old": 1},
            prev_num_draft_len=0,
            generator=object(),
        )
        for rid in ("a", "b")
    }
    runner = SimpleNamespace(requests=states, input_batch=Mock())
    rng = states["a"].generator
    restore_requests(runner, {"a": ([1], SamplingParams())})
    assert states["a"].output_token_ids == [1]
    assert states["a"].generator is rng
    assert states["b"].output_token_ids == [1, 2, 3]
    assert states["b"].persistent_data == {"old": 1}
    runner.input_batch.remove_request.assert_called_once_with("a")
