import torch

from aphrodite.common.utils import (is_pin_memory_available,
                                    make_tensor_with_pad)
from aphrodite.v1.sample.metadata import SamplingMetadata
from aphrodite.v1.sample.ops.bad_words import apply_bad_words
from aphrodite.v1.sample.ops.dry import apply_all_dry
from aphrodite.v1.sample.ops.epsilon_cutoff import epsilon_cutoff
from aphrodite.v1.sample.ops.eta_cutoff import eta_cutoff
from aphrodite.v1.sample.ops.min_p import min_p
from aphrodite.v1.sample.ops.no_repeat_ngram import no_repeat_ngram
from aphrodite.v1.sample.ops.penalties import (apply_all_penalties,
                                               apply_min_token_penalties)
from aphrodite.v1.sample.ops.quadratic import quadratic
from aphrodite.v1.sample.ops.skew import skew
from aphrodite.v1.sample.ops.tfs import tfs
from aphrodite.v1.sample.ops.top_a import top_a
from aphrodite.v1.sample.ops.top_nsigma import top_nsigma
from aphrodite.v1.sample.ops.typical_p import typical_p
from aphrodite.v1.sample.ops.xtc import xtc


class SamplingOps:
    """Handles all sampling operations applied to logits."""

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.min_tokens:
            apply_min_token_penalties(logits,
                                      sampling_metadata.output_token_ids,
                                      sampling_metadata.min_tokens)
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                sampling_metadata.output_token_ids,
            )
        return logits

    def apply_dry(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Apply DRY sampling to the logits."""
        if (sampling_metadata.dry_multiplier is not None
                and sampling_metadata.prompt_token_ids is not None):

            # Convert output_token_ids to tensor
            _, vocab_size = logits.shape
            output_tokens_t = make_tensor_with_pad(
                sampling_metadata.output_token_ids,
                pad=vocab_size,
                device="cpu",
                dtype=torch.int64,
                pin_memory=is_pin_memory_available(),
            ).to(logits.device, non_blocking=True)

            # Ensure all required tensors are not None
            if (sampling_metadata.dry_base is not None
                    and sampling_metadata.dry_allowed_length is not None
                    and sampling_metadata.dry_sequence_breaker_ids is not None
                    and sampling_metadata.dry_ranges is not None
                    and sampling_metadata.dry_max_ngram is not None
                    and sampling_metadata.dry_max_occurrences is not None and
                    sampling_metadata.dry_early_exit_match_len is not None):

                logits = apply_all_dry(
                    logits,
                    sampling_metadata.prompt_token_ids,
                    output_tokens_t,
                    sampling_metadata.dry_multiplier,
                    sampling_metadata.dry_base,
                    sampling_metadata.dry_allowed_length,
                    sampling_metadata.dry_sequence_breaker_ids,
                    sampling_metadata.dry_ranges,
                    sampling_metadata.dry_max_ngram,
                    sampling_metadata.dry_max_occurrences,
                    sampling_metadata.dry_early_exit_match_len,
                )
        return logits

    def apply_no_repeat_ngram(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return no_repeat_ngram(logits, sampling_metadata)

    def apply_min_p(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return min_p(logits, sampling_metadata)

    def apply_top_a(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return top_a(logits, sampling_metadata)

    def apply_tfs(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return tfs(logits, sampling_metadata)

    def apply_eta_cutoff(

        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return eta_cutoff(logits, sampling_metadata)

    def apply_epsilon_cutoff(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return epsilon_cutoff(logits, sampling_metadata)

    def apply_typical_p(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return typical_p(logits, sampling_metadata)

    def apply_quadratic(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return quadratic(logits, sampling_metadata)

    def apply_xtc(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return xtc(logits, sampling_metadata)

    def apply_top_nsigma(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return top_nsigma(logits, sampling_metadata)

    def apply_skew(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return skew(logits, sampling_metadata)

    def apply_logits_bias(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        # TODO: this implementation is extremely inefficient.
        # One idea is implement this as a PyTorch C++ op, and we may
        # even optimize the logit_bias layout.

        # Get vocabulary size from logits
        vocab_size = logits.shape[-1]

        for i, logit_bias in enumerate(sampling_metadata.logit_bias):
            if logit_bias:
                for token_id, bias in logit_bias.items():
                    # Check token_id bounds to ensure within vocabulary
                    if token_id < 0 or token_id >= vocab_size:
                        raise ValueError(
                            f"token_id {token_id} in logit_bias contains "
                            f"out-of-vocab token id. Vocabulary size: "
                            f"{vocab_size}")
                    logits[i, token_id] += bias
        return logits

    def apply_allowed_token_ids(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask,
                                float("-inf"))
        return logits

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.bad_words_token_ids:
            apply_bad_words(
                logits,
                sampling_metadata.bad_words_token_ids,
                sampling_metadata.output_token_ids,
            )
        return logits
