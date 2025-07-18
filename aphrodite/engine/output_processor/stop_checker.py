from typing import Callable, List, Optional, Tuple

from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.sequence import Sequence, SequenceStatus
from aphrodite.lora.request import LoRARequest
from aphrodite.transformers_utils.tokenizer import AnyTokenizer


class StopChecker:
    """AphroditeEngine helper class which separates out the logic involving stop
    checking. This checks things such as: whether the eos token was emitted,
    whether the max_tokens has been consumed, whether a stop string has been
    emitted, or if we have exceeded the max model len.
    """

    def __init__(self, max_model_len: int,
                 get_tokenizer_for_seq: Callable[[Sequence], AnyTokenizer]):
        # Do not use it directly, but use `self._get_max_model_len`.
        self._max_model_len = max_model_len
        self.get_tokenizer_for_seq = get_tokenizer_for_seq

    def _get_max_model_len(self, lora_req: Optional[LoRARequest]):
        if lora_req and lora_req.long_lora_max_len:
            return lora_req.long_lora_max_len
        else:
            return self._max_model_len

    def maybe_stop_sequence(
        self,
        seq: Sequence,
        new_char_count: int,
        sampling_params: SamplingParams,
        lora_req: Optional[LoRARequest] = None,
    ) -> None:
        """Stop the finished sequences.

       new_char_count is the number of chars added to the
           sequence's output text for the newly generated token
        """

        # Check if the minimum number of tokens has been generated yet;
        # skip the stop string/token checks if not
        if seq.get_output_len() < sampling_params.min_tokens:
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == seq.eos_token_id):
            # Remove the last EOS token unless explicitly specified
            # This prevents unintended exposure of the EOS token
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                seq.output_text = seq.output_text[:-new_char_count]
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if a stop token was encountered.
        # This assumes a single token produced per step.
        last_token_id = seq.get_last_token_id()
        if last_token_id in (sampling_params.stop_token_ids or ()):
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                # Remove last token
                seq.output_text = seq.output_text[:-new_char_count]
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = last_token_id
            return

        # Check if any stop strings are matched.
        stop = self.check_stop_strings(
            seq.output_text, new_char_count, sampling_params.stop,
            sampling_params.include_stop_str_in_output)
        if stop is not None:
            stop_str, truncate_to = stop
            if truncate_to != -1:
                seq.output_text = seq.output_text[:truncate_to]
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = stop_str
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self._get_max_model_len(lora_req):
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

    @staticmethod
    def check_stop_strings(
        output_text: str,
        new_char_count: int,
        stop: List[str],
        include_in_output: bool,
    ) -> Optional[Tuple[str, int]]:
        """Check if any stop strings are matched and truncate sequence
        output text accordingly.

        Returns tuple (stop_string, offset) if matched or else None.

        Where stop_string is the matched stop string and offset is the
        length to which output_text should be truncated, or -1 for no
        truncation.
        """
        if not new_char_count or not stop:
            return None

        for stop_str in stop:
            stop_string_len = len(stop_str)
            # Avoid searching already-searched text.
            stop_index = output_text.find(stop_str,
                                          1 - new_char_count - stop_string_len)
            if stop_index == -1:
                continue

            if include_in_output:
                # Truncate to end of stop string.
                stop_index += stop_string_len
                if stop_index >= len(output_text):
                    # No truncation required.
                    return stop_str, -1

            # Truncate the output text to either the beginning
            # or end of the stop string.
            return stop_str, stop_index
        return None
