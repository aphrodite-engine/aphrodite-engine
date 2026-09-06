# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from aphrodite.sampling_params import SamplingParams
from aphrodite.v1.phrase_guard.matcher import validate_phrases
from aphrodite.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from aphrodite.v1.sample.logits_processor.builtin import process_dict_updates

RETRY_KEY = "_sonar_phrase_retry"


class PhraseRetryProcessor(LogitsProcessor):
    """Mask failed alternatives at their checkpoint, never at another position."""

    def __init__(self, aphrodite_config, device, is_pin_memory):
        self.device = device
        self.states: dict[int, tuple[list[int], int, torch.Tensor]] = {}

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        args = sampling_params.extra_args or {}
        if "banned_strings" in args:
            validate_phrases(args["banned_strings"])
            case_sensitive = args.get("banned_strings_case_sensitive", False)
            if not isinstance(case_sensitive, (bool, int)) or case_sensitive not in (0, 1):
                raise ValueError("banned_strings_case_sensitive must be a boolean or 0/1")
            if sampling_params.stop:
                raise ValueError("Experimental banned_strings does not support text stop strings; use stop_token_ids")
            if (
                sampling_params.structured_outputs is not None
                or sampling_params.trace_decode_token_ids is not None
                or sampling_params.mirostat_mode
            ):
                raise ValueError("Experimental banned_strings does not support grammar, replay, or Mirostat")
        if RETRY_KEY in args:
            raise ValueError(f"{RETRY_KEY} is reserved for the phrase scheduler")

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        def add(params, prompt, output):
            retry = (params.extra_args or {}).get(RETRY_KEY)
            if retry is None:
                return None
            position, tokens = retry
            return output, position, torch.tensor(tokens, dtype=torch.long, device=self.device)

        process_dict_updates(self.states, batch_update, add)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for row, (output, position, tokens) in self.states.items():
            if len(output) == position:
                logits[row].index_fill_(0, tokens, float("-inf"))
                # Return a blocked token as an error sentinel if constraints
                # exhaust the row. The scheduler discards it before decoding.
                first = tokens[:1]
                logits[row].scatter_(0, first, torch.where(logits[row].amax() == -torch.inf, 0.0, logits[row][first]))
        return logits
