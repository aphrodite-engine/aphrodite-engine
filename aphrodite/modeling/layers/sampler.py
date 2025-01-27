"""A layer that samples the next tokens from the model's outputs."""
import itertools
import warnings
from dataclasses import dataclass
from enum import IntEnum
from math import inf
from typing import Dict, List, Optional, Tuple, Union

import msgspec
import torch
import torch.nn as nn
from loguru import logger

import aphrodite._custom_ops as ops
import aphrodite.common.envs as envs
from aphrodite.common.sampling_params import SamplingType
from aphrodite.common.sequence import (CompletionSequenceGroupOutput, Logprob,
                                       PromptLogprobs, SampleLogprobs,
                                       SequenceOutput)
from aphrodite.common.utils import is_cpu
from aphrodite.modeling.sampling_metadata import (SamplingMetadata,
                                                  SamplingTensors,
                                                  SequenceGroupToSample)
from aphrodite.spec_decode.metrics import SpecDecodeWorkerMetrics

# (num_token_ids, num_parent_ids) per sequence group.
SampleResultType = List[Tuple[List[int], List[int]]]

# Types of temporary data structures used for
# computing sample_result
SampleMetadataType = Dict[SamplingType, Tuple[List[int],
                                              List[SequenceGroupToSample]]]
MultinomialSamplesType = Dict[SamplingType, torch.Tensor]
SampleResultsDictType = Dict[int, Tuple[List[int], List[int]]]


# Encapsulates temporary data structures for computing
# sample_result.
#
# * For multi-step scheduling: must be returned
#   by `Sampler.forward()` and used later to compute the pythonized
#   sample_result
#
# * For single-step scheduling: consumed immediately
#   inside `Sampler.forward()` to compute pythonized sample_result.
@dataclass
class SampleResultArgsType:
    sample_metadata: SampleMetadataType
    multinomial_samples: MultinomialSamplesType
    sample_results_dict: SampleResultsDictType
    sampling_metadata: SamplingMetadata
    greedy_samples: Optional[torch.Tensor]
    beam_search_logprobs: Optional[torch.Tensor]


# Union of non-deferred (single-step scheduling)
# vs deferred (multi-step scheduling)
# sample result types
MaybeDeferredSampleResultType = Union[SampleResultType, SampleResultArgsType]

# Abbreviation of the _sample() return type
SampleReturnType = Tuple[MaybeDeferredSampleResultType, Optional[torch.Tensor]]


class SamplerOutput(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """For each sequence group, we generate a list of SequenceOutput object,
    each of which contains one possible candidate for the next token.
    This data structure implements methods, so it can be used like a list, but
    also has optional fields for device tensors.
    """

    outputs: List[CompletionSequenceGroupOutput]

    # On-device tensor containing probabilities of each token.
    sampled_token_probs: Optional[torch.Tensor] = None

    # On-device tensor containing the logprobs of each token.
    logprobs: Optional["torch.Tensor"] = None

    # Holds either (1) the pythonized sampler result (single-step scheduling)
    # or (2) what will be arguments for later deferred pythonization of the
    # sampler result (muliti-step scheduling)
    deferred_sample_results_args: Optional[SampleResultArgsType] = None

    # On-device tensor containing the sampled token ids.
    sampled_token_ids: Optional[torch.Tensor] = None
    # CPU tensor containing the sampled token ids. Used during multi-step to
    # return the sampled token ids from last rank to AsyncLLMEngine to be
    # 'broadcasted' to all other PP ranks for next step.
    sampled_token_ids_cpu: Optional[torch.Tensor] = None

    # Spec decode metrics populated by workers.
    spec_decode_worker_metrics: Optional[SpecDecodeWorkerMetrics] = None

    # Optional last hidden states from the model.
    hidden_states: Optional[torch.Tensor] = None

    # Optional prefill hidden states from the model
    # (used for models like EAGLE).
    prefill_hidden_states: Optional[torch.Tensor] = None

    # Time taken in the forward pass for this across all workers
    model_forward_time: Optional[float] = None

    # Time taken in the model execute function. This will include model forward,
    # block/sync across workers, cpu-gpu sync time and sampling time.
    model_execute_time: Optional[float] = None

    def __getitem__(self, idx: int):
        return self.outputs[idx]

    def __setitem__(self, idx: int, value):
        self.outputs[idx] = value

    def __len__(self):
        return len(self.outputs)

    def __eq__(self, other: object):
        return isinstance(other,
                          self.__class__) and self.outputs == other.outputs

    def __repr__(self) -> str:
        """Show the shape of a tensor instead of its values to reduce noise.
        """
        sampled_token_probs_repr = ("None" if self.sampled_token_probs is None
                                    else self.sampled_token_probs.shape)
        sampled_token_ids_repr = ("None" if self.sampled_token_ids is None else
                                  self.sampled_token_ids.shape)
        return (
            f"SamplerOutput(outputs={self.outputs}, "
            f"sampled_token_probs={sampled_token_probs_repr}, "
            f"sampled_token_ids={sampled_token_ids_repr}, "
            f"spec_decode_worker_metrics={self.spec_decode_worker_metrics})")

# There isn't a "safe" temperature range for fp16 logits.
# This value was chosen because 1/2e-5 is just under the 65k fp16 max, meaning
# that this temperature well-uses the fp16 space after the logits are offset.
_TEMPERATURE_MINIMUM = 2e-5

# If enabled, we switch to a more performant implementation
# of top-k and top-p
APHRODITE_USE_SAMPLING_KERNELS = envs.APHRODITE_USE_SAMPLING_KERNELS


class SamplerID(IntEnum):
    # Mirror these in aphrodite/common/sampling_params.py
    # Values out of order to keep backwards compatibility
    # with Koboldcpp values
    DRY = 7
    PENALTIES = 6
    NO_REPEAT_NGRAM = 8
    TEMPERATURE = 5
    TOP_NSIGMA = 9
    TOP_P_TOP_K = 0
    TOP_A = 1
    MIN_P = 2
    TFS = 3
    ETA_CUTOFF = 10
    EPSILON_CUTOFF = 11
    TYPICAL_P = 4
    QUADRATIC = 12
    XTC = 13


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence, frequency and repetition penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).

    The structure of the logits tensor is coupled with the seq_groups in
    sampling_metadata. Typically, each sequence in each seq_group has one row in
    logits for the next token to be sampled; however, for a seq_group with a
    prompt request with the prompt_logprobs sampling parameter, there are rows
    in logits for each token in the input prompt.
    """

    def __init__(self):
        super().__init__()

        # Whether or not the SamplerOutput should have on-device tensors
        # containing the sampled token ids and probabilities. This is used by
        # speculative decoding.
        self.include_gpu_probs_tensor = False
        self.should_modify_greedy_probs_inplace = False

    def _init_sampling_tensors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ):
        """The goal here is to reuse sampling tensors between similar decode
        runs. This is possible because sampling logic does not change between
        decodes of the same sequences.
        """
        _, vocab_size = logits.shape

        # First free any existing stored sampling tensors.
        # This is necessary because some sampling tensors may
        # have pinned memory.
        self._sampling_tensors = None

        # Initialize new sampling tensors
        (sampling_tensors, do_penalties, do_no_repeat_ngrams, do_temperatures,
         do_top_p_top_k, do_top_as, do_min_p, do_tfss, do_eta_cutoffs,
         do_epsilon_cutoffs, do_typical_ps, do_quadratic, do_xtc, do_nsigmas,
         do_dry, do_skew, do_temp_last
         ) = SamplingTensors.from_sampling_metadata(
             sampling_metadata, vocab_size, logits.device, logits.dtype)

        self._sampling_tensors = sampling_tensors
        self._do_penalties = do_penalties
        self._do_no_repeat_ngrams = do_no_repeat_ngrams
        self._do_temperatures = do_temperatures
        self._do_top_p_top_k = do_top_p_top_k
        self._do_top_as = do_top_as
        self._do_min_p = do_min_p
        self._do_tfss = do_tfss
        self._do_eta_cutoffs = do_eta_cutoffs
        self._do_epsilon_cutoffs = do_epsilon_cutoffs
        self._do_typical_ps = do_typical_ps
        self._do_quadratic = do_quadratic
        self._do_xtc = do_xtc
        self._do_nsgimas = do_nsigmas
        self._do_dry = do_dry
        self._do_skew = do_skew
        self._do_temp_last = do_temp_last

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """
        Single-step scheduling:
        * Perform GPU-side sampling computation & compute
          GPU-side logprobs tensor
        * Pythonize sampling result & logprobs tensor
        Multi-step scheduling:
        * Perform GPU-side sampling computation & compute
          GPU-side logprobs tensor
        * Defer Pythonization of sampling result & logprobs
          tensor
        * Encapsulate arguments required for deferred Pythonization
          in the :class:`SamplerOutput` structure

        Args:
            logits: (num_tokens, vocab_size).
            sampling_metadata: Metadata for sampling.
        """
        assert logits is not None
        _, vocab_size = logits.shape

        # Prepare sampling tensors with pinned memory to avoid blocking.
        if not sampling_metadata.reuse_sampling_tensors:
            self._init_sampling_tensors(logits, sampling_metadata)
        elif self._do_penalties or self._do_dry:
            # In this case, the sampling tensors logic depends on
            # "output_tokens" of a sequence. As a result, we cannot
            # reuse sampling tensors, since "output_tokens" changes
            # between decode runs.
            self._init_sampling_tensors(logits, sampling_metadata)

        assert self._sampling_tensors is not None
        sampling_tensors = self._sampling_tensors
        do_penalties = self._do_penalties
        do_no_repeat_ngrams = self._do_no_repeat_ngrams
        do_temperatures = self._do_temperatures
        do_top_p_top_k = self._do_top_p_top_k
        do_top_as = self._do_top_as
        do_min_p = self._do_min_p
        do_tfss = self._do_tfss
        do_eta_cutoffs = self._do_eta_cutoffs
        do_epsilon_cutoffs = self._do_epsilon_cutoffs
        do_typical_ps = self._do_typical_ps
        do_quadratic = self._do_quadratic
        do_xtc = self._do_xtc
        do_nsigmas = self._do_nsgimas
        do_dry = self._do_dry
        do_skew = self._do_skew
        do_temp_last = self._do_temp_last

        logits = _apply_min_tokens_penalty(logits, sampling_metadata)
        banned_tokens = _get_custom_token_bans(sampling_metadata)
        logits = _apply_token_bans(logits, banned_tokens)

        sampler_order = None
        if sampling_metadata.seq_groups:
            sampler_order = sampling_metadata.seq_groups[
                0].sampling_params.sampler_priority

            # Warn if both custom order and temp_last are specified
            if (sampling_metadata.seq_groups and 
                sampling_metadata.seq_groups[0].is_prompt and
                sampler_order is not None and 
                do_temp_last):
                logger.warning(
                    "Both sampler_priority and temperature_last=True "
                    "were specified. Using custom sampler_priority order "
                    "and ignoring temperature_last.")

        if sampler_order is None:
            default_order = [
                SamplerID.DRY,
                SamplerID.PENALTIES,
                SamplerID.NO_REPEAT_NGRAM,
                SamplerID.TEMPERATURE,
                SamplerID.TOP_NSIGMA,
                SamplerID.TOP_P_TOP_K,
                SamplerID.TOP_A,
                SamplerID.MIN_P,
                SamplerID.TFS,
                SamplerID.ETA_CUTOFF,
                SamplerID.EPSILON_CUTOFF,
                SamplerID.TYPICAL_P,
                SamplerID.QUADRATIC,
                SamplerID.XTC,
            ]

            sampler_order = []
            for sampler_id in default_order:
                if sampler_id == SamplerID.TEMPERATURE and do_temp_last:
                    continue
                sampler_order.append(sampler_id)

                if sampler_id == SamplerID.XTC and do_temp_last:
                    sampler_order.append(SamplerID.TEMPERATURE)

        if sampling_metadata.seq_groups and sampling_metadata.seq_groups[
            0].is_prompt:
            logger.debug("Sampler execution order: ")
            for i, sampler_id in enumerate(sampler_order, 1):
                logger.debug(f"{i}. {SamplerID(sampler_id).name}")

            enabled_samplers = []
            # ruff: noqa: E701
            if do_penalties: enabled_samplers.append("PENALTIES")
            if do_no_repeat_ngrams: enabled_samplers.append("NO_REPEAT_NGRAM")
            if do_temperatures: enabled_samplers.append("TEMPERATURE")
            if do_top_p_top_k: enabled_samplers.append("TOP_P_TOP_K")
            if do_top_as: enabled_samplers.append("TOP_A")
            if do_min_p: enabled_samplers.append("MIN_P")
            if do_tfss: enabled_samplers.append("TFS")
            if do_eta_cutoffs: enabled_samplers.append("ETA_CUTOFF")
            if do_epsilon_cutoffs: enabled_samplers.append("EPSILON_CUTOFF")
            if do_typical_ps: enabled_samplers.append("TYPICAL_P")
            if do_quadratic: enabled_samplers.append("QUADRATIC")
            if do_xtc: enabled_samplers.append("XTC")
            if do_nsigmas: enabled_samplers.append("TOP_NSIGMA")
            if do_dry: enabled_samplers.append("DRY")
            if do_skew: enabled_samplers.append("SKEW")
            logger.debug(f"Enabled samplers: {', '.join(enabled_samplers)}")

        for sampler_id in sampler_order:
            if sampler_id == SamplerID.DRY and do_dry:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        f"Applying DRY with dry_multiplier: "
                        f"{sampling_tensors.dry_multipliers}.")
                logits = _apply_dry(
                    logits,
                    sampling_tensors.prompt_tokens,
                    sampling_tensors.output_tokens,
                    sampling_tensors.dry_multipliers,
                    sampling_tensors.dry_bases, 
                    sampling_tensors.dry_allowed_lengths,
                    sampling_tensors.dry_sequence_breaker_ids,
                    sampling_tensors.dry_ranges)

            elif sampler_id == SamplerID.PENALTIES and do_penalties:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        "Applying penalties with "
                        f"pres_pen: {sampling_tensors.presence_penalties}, "
                        f"freq_pen: {sampling_tensors.frequency_penalties}, "
                        f"rep_pen: {sampling_tensors.repetition_penalties}.")
                logits = _apply_penalties(
                    logits, sampling_tensors.prompt_tokens,
                    sampling_tensors.output_tokens,
                    sampling_tensors.presence_penalties,
                    sampling_tensors.frequency_penalties,
                    sampling_tensors.repetition_penalties)

            elif sampler_id == SamplerID.NO_REPEAT_NGRAM and \
                do_no_repeat_ngrams:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        "Applying no_repeat_ngram with no_repeat_ngram_size: "
                        f"{sampling_tensors.no_repeat_ngram_sizes}.")
                logits = _apply_no_repeat_ngram(
                    logits,
                    sampling_tensors.prompt_tokens,
                    sampling_tensors.no_repeat_ngram_sizes)

            elif sampler_id == SamplerID.TEMPERATURE and do_temperatures:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        "Applying temperatures with temperature: "
                        f"{sampling_tensors.temperatures}, "
                        f"dynatemp_min: {sampling_tensors.dynatemp_mins}, "
                        f"dynatemp_max: {sampling_tensors.dynatemp_maxs}, "
                        f"dynamtep_exp: {sampling_tensors.dynatemp_exps}.")
                _apply_temperatures(
                    logits, sampling_tensors.temperatures,
                    sampling_tensors.dynatemp_mins,
                    sampling_tensors.dynatemp_maxs,
                    sampling_tensors.dynatemp_exps)

            elif sampler_id == SamplerID.TOP_NSIGMA and do_nsigmas:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        "Applying Top-Nsigma with nsigma: "
                        f"{sampling_tensors.nsigmas}")
                logits = _apply_top_nsigma(
                    logits, sampling_tensors.nsigmas)

            elif sampler_id == SamplerID.TOP_P_TOP_K and do_top_p_top_k and \
                not APHRODITE_USE_SAMPLING_KERNELS:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        "Applying Top-p and Top-k with top-p: "
                        f"{sampling_tensors.top_ps}, top_k: "
                        f"{sampling_tensors.top_ks}.")
                logits = _apply_top_k_top_p(
                    logits, sampling_tensors.top_ps,
                    sampling_tensors.top_ks)

            elif sampler_id == SamplerID.TOP_A and do_top_as:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        "Applying Top-a with Top-a: "
                        f"{sampling_tensors.top_as}.")
                logits = _apply_top_a(
                    logits, sampling_tensors.top_as)

            elif sampler_id == SamplerID.MIN_P and do_min_p:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        "Applying Min-p with Min-p: "
                        f"{sampling_tensors.min_ps}.")
                logits = _apply_min_p(
                    logits, sampling_tensors.min_ps)

            elif sampler_id == SamplerID.TFS and do_tfss:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        "Applying Tail-Free Sampling with tfs: "
                        f"{sampling_tensors.tfss}.")
                logits = _apply_tfs(
                    logits, sampling_tensors.tfss)

            elif sampler_id == SamplerID.ETA_CUTOFF and do_eta_cutoffs:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        "Applying ETA Cutoff with eta_cutoff: "
                        f"{sampling_tensors.eta_cutoffs}.")
                logits = _apply_eta_cutoff(
                    logits, sampling_tensors.eta_cutoffs)

            elif sampler_id == SamplerID.EPSILON_CUTOFF and do_epsilon_cutoffs:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        "Applying Epsilon Cutoff with epsilon_cutoff: "
                        f"{sampling_tensors.epsilon_cutoffs}.")
                logits = _apply_epsilon_cutoff(
                    logits, sampling_tensors.epsilon_cutoffs)

            elif sampler_id == SamplerID.TYPICAL_P and do_typical_ps:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        "Applying Locally Typical Sampling with typical_p: "
                        f"{sampling_tensors.typical_ps}.")
                logits = _apply_typical_sampling(
                    logits, sampling_tensors.typical_ps)

            elif sampler_id == SamplerID.QUADRATIC and do_quadratic:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        "Applying Quadratic and Cubic Sampling with "
                        "smoothing_factors: "
                        f"{sampling_tensors.smoothing_factors},"
                        f" smoothing_curves: "
                        f"{sampling_tensors.smoothing_curves}.")
                logits = _apply_quadratic_sampling(
                    logits, sampling_tensors.smoothing_factors,
                    sampling_tensors.smoothing_curves)

            elif sampler_id == SamplerID.XTC and do_xtc:
                if (sampling_metadata.seq_groups and
                    sampling_metadata.seq_groups[0].is_prompt):
                    logger.debug(
                        "Applying Exclude Top Choices sampling with "
                        f"xtc_threshold: {sampling_tensors.xtc_thresholds}, "
                        "xtc_probability: "
                        f"{sampling_tensors.xtc_probabilities}.")
                logits = _apply_xtc_sampling(
                    logits, sampling_tensors.xtc_thresholds,
                    sampling_tensors.xtc_probabilities)


        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)

        # skew needs to be applied post-softmax
        if do_skew:
            if (sampling_metadata.seq_groups and
                sampling_metadata.seq_groups[0].is_prompt):
                logger.debug(
                    "Applying Skew sampling with skew: "
                    f"{sampling_tensors.skews}.")
            # reference: https://github.com/turboderp/exllamav2/commit/1de4cdd70b09208e7b4f17ee322c190e16f60efd
            cum_probs = torch.cumsum(probs, dim=-1)
            cum_probs = torch.pow(cum_probs, torch.exp(
                sampling_tensors.skews).unsqueeze(dim=1))
            probs = torch.diff(cum_probs, dim=-1,
                               prepend=torch.zeros_like(cum_probs[..., :1]))
            logits = torch.log(probs)

        # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        maybe_deferred_sample_results, maybe_sampled_tokens_tensor = _sample(
            probs,
            logprobs,
            sampling_metadata,
            sampling_tensors,
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
            modify_greedy_probs=self._should_modify_greedy_probs_inplace,
        )

        if self.include_gpu_probs_tensor:
            # Since we will defer sampler result Pythonization,
            # preserve GPU-side tensors in support of later
            # deferred pythonization of logprobs
            assert maybe_sampled_tokens_tensor is not None
            on_device_tensors = (probs, logprobs, maybe_sampled_tokens_tensor)
        else:
            # Since Pythonization has already happened, don't preserve
            # GPU-side tensors.
            on_device_tensors = None

        # Get the logprobs query results.
        prompt_logprobs = None
        sample_logprobs = None
        if not sampling_metadata.skip_sampler_cpu_output:
            # Pythonize logprobs now (GPU -> CPU); do not defer.
            assert not isinstance(maybe_deferred_sample_results,
                                  SampleResultArgsType)
            prompt_logprobs, sample_logprobs = get_logprobs(
                logprobs, sampling_metadata, maybe_deferred_sample_results)

        return _build_sampler_output(
            maybe_deferred_sample_results,
            sampling_metadata,
            prompt_logprobs,
            sample_logprobs,
            on_device_tensors=on_device_tensors,
            skip_sampler_cpu_output=sampling_metadata.skip_sampler_cpu_output)

    @property
    def _should_modify_greedy_probs_inplace(self) -> bool:
        """Whether or not the sampler should modify the probability distribution
        of greedily-sampled tokens such that multinomial sampling would sample
        the greedily-sampled token.

        In other words, if True then we set the probability of the greedily-
        sampled token to 1.

        This is used by speculative decoding, which requires that the sampling
        method be encoded into the probability distribution.
        """
        return self.should_modify_greedy_probs_inplace


def _get_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def _get_custom_token_bans(
        sampling_metadata: SamplingMetadata) -> List[List[int]]:
    assert sampling_metadata.seq_groups is not None
    banned_tokens: List[List[int]] = []
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        sampling_params = sampling_metadata.seq_groups[i].sampling_params
        seq_ids = seq_group.seq_ids
        custom_token_bans = sampling_params.custom_token_bans
        if (i < sampling_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            prompt_len = len(seq_group.prompt_logprob_indices)
            banned_tokens += [custom_token_bans] * (prompt_len - 1)
        banned_tokens += [custom_token_bans] * len(seq_ids)
    return banned_tokens


def _apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
                     output_tokens_tensor: torch.Tensor,
                     presence_penalties: torch.Tensor,
                     frequency_penalties: torch.Tensor,
                     repetition_penalties: torch.Tensor) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = _get_bin_counts_and_mask(prompt_tokens_tensor, vocab_size,
                                              num_seqs)
    output_bin_counts, output_mask = _get_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)

    repetition_penalties = repetition_penalties[:, None].repeat(1, vocab_size)
    repetition_penalties[~(prompt_mask | output_mask)] = 1.0
    logits = torch.where(logits > 0, logits / repetition_penalties,
                         logits * repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze_(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze_(dim=1) * output_mask
    return logits


def _apply_temperatures(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    dynatemp_mins: torch.Tensor,
    dynatemp_maxs: torch.Tensor,
    dynatemp_exps: torch.Tensor,
) -> None:
    dynatemp_mask = (dynatemp_mins != 0) | (dynatemp_maxs != 0)
    dynatemp_mins = dynatemp_mins[dynatemp_mask]
    dynatemp_maxs = dynatemp_maxs[dynatemp_mask]
    dynatemp_exps = dynatemp_exps[dynatemp_mask]

    dynatemp_logits = logits[dynatemp_mask]
    dynatemp_shifted_logits = torch.log_softmax(dynatemp_logits, dim=-1)
    dynatemp_probs = dynatemp_shifted_logits.exp()
    dynatemp_entropies = -(dynatemp_probs *
                           dynatemp_shifted_logits).nansum(dim=-1)
    dynatemp_max_entropies = torch.log_(
        (dynatemp_logits > float("-inf")).sum(dim=-1).float())
    normalized_entropies = dynatemp_entropies.div_(dynatemp_max_entropies)
    dyn_temp = (dynatemp_mins + (dynatemp_maxs - dynatemp_mins) *
                normalized_entropies.pow_(dynatemp_exps))
    temperatures[dynatemp_mask] = dyn_temp
  
    temperatures[temperatures.isnan()] = _TEMPERATURE_MINIMUM
    temperatures[temperatures <= _TEMPERATURE_MINIMUM] = _TEMPERATURE_MINIMUM
  
    # To prevent saturation of top logits, we shift the range to [-inf, 1]
    # Why align to 1, instead of 0? Because [0, 1] holds 25% of all floats.
    # Why mask? So we aren't potentially discarding data in milder temps.
    low_temps = temperatures < 0.1
    logits[low_temps] -= logits.max(dim=-1, keepdim=True).values[low_temps] - 1
    logits.div_(temperatures.unsqueeze(dim=1))


def _apply_token_bans(logits: torch.Tensor,
                      banned_tokens: List[List[int]]) -> torch.Tensor:
    for i, banned_token_ids in enumerate(banned_tokens):
        if i >= logits.size(0):
            break
        if not banned_token_ids:
            continue
        logits[i, banned_token_ids] = -float("inf")
    return logits


def _apply_min_tokens_penalty(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Apply min_tokens penalty which sets stop tokens to -inf if min_tokens
        have not been generated yet
    """
    # list of indices in logits that will be set to -inf
    logits_to_penalize = []
    logits_applied = 0
    for seq_group in sampling_metadata.seq_groups:
        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params

        sample_indices = seq_group.sample_indices
        logits_applied += len(sample_indices) + len(
            seq_group.prompt_logprob_indices)
        if not seq_group.do_sample:
            continue

        start_idx = sample_indices[0]
        min_tokens = sampling_params.min_tokens
        token_ids_to_penalize = sampling_params.all_stop_token_ids
        if min_tokens > 0 and token_ids_to_penalize:
            seqs_to_penalize = []
            for j, seq_id in enumerate(seq_ids):
                seq_data = seq_group.seq_data[seq_id]
                if len(seq_data.output_token_ids_array) < min_tokens:
                    seqs_to_penalize.append(j)

            if seqs_to_penalize:
                # convert to the index into logits
                seqs_to_penalize = [start_idx + j for j in seqs_to_penalize]
                # itertools.product pairs each seq index with every token id
                logits_to_penalize.extend(
                    itertools.product(seqs_to_penalize, token_ids_to_penalize))

    if logits_to_penalize:
        # use zip and * to group indices along each dimension
        # eg. [ (1,2), (1,3), (5,6) ] -> ( (1,1,5), (2,3,6) )
        logits[tuple(zip(*logits_to_penalize))] = -float("inf")

    # verifies that no rows in logits were missed unexpectedly
    assert logits_applied == logits.shape[0]
    return logits

def _apply_dry(
    logits: torch.Tensor,
    input_token_ids: torch.Tensor,
    output_token_ids: torch.Tensor,
    multipliers: torch.Tensor, 
    bases: torch.Tensor,
    allowed_lengths: torch.Tensor,
    sequence_breakers_ids: torch.Tensor,
    ranges: torch.Tensor,
) -> torch.Tensor:
    """
    Apply Don't Repeat Yourself (DRY) sampling to the logits.

    Reference: https://github.com/oobabooga/text-generation-webui/pull/5677 and
    https://github.com/AlpinDale/vllm/pull/1
    """

    VOCAB_SIZE = logits.size(-1)

    # ---- Key constants for DRY behavior ----
    MAX_NGRAM = 12            # Maximum length of match to check
    MAX_OCCURRENCES = 8       # How many occurrences of last_token we analyze
    EARLY_EXIT_MATCH_LEN = 8  # If we find this large a match, we stop searching
    # ----------------------------------------

    applies_to = multipliers.nonzero(as_tuple=True)[0]
    for irow in applies_to.tolist():
        prompt_len = len(input_token_ids[irow]) - (
            input_token_ids[irow] == VOCAB_SIZE).sum().item()
        output_len = len(output_token_ids[irow]) - (
            output_token_ids[irow] == VOCAB_SIZE).sum().item()

        token_seq = torch.cat(
            (input_token_ids[irow][:prompt_len],
             output_token_ids[irow][:output_len]),
            dim=0
        )

        range_limit = ranges[irow].item()
        if range_limit > 0:
            token_seq = token_seq[-range_limit:]

        if token_seq.size(0) < 2:
            continue

        last_token = token_seq[-1].item()
        if last_token in sequence_breakers_ids[irow]:
            continue

        break_mask = torch.zeros(len(token_seq),
                                 dtype=torch.bool, device=logits.device)
        for break_tok in sequence_breakers_ids[irow]:
            break_mask.logical_or_(token_seq == break_tok)

        max_ngram = 0
        for max_ngram in range(min(len(break_mask), MAX_NGRAM + 1)):
            if break_mask[-max_ngram - 1]:
                break

        min_ngram = allowed_lengths[irow].item()
        if max_ngram <= min_ngram:
            continue

        ngram_lens = torch.zeros(
            VOCAB_SIZE, dtype=torch.int32, device=logits.device)

        endpoint_indexes_all = torch.nonzero(
            token_seq == last_token, as_tuple=True)[0].tolist()
        if len(endpoint_indexes_all) < 2:
            continue
        endpoint_indexes = endpoint_indexes_all[:-1]

        if len(endpoint_indexes) > MAX_OCCURRENCES:
            endpoint_indexes = endpoint_indexes[-MAX_OCCURRENCES:]

        for idx in reversed(endpoint_indexes):
            if idx == len(token_seq) - 1:
                continue

            match_len = 0
            for unwind in range(1, min(idx, max_ngram) + 1):
                if break_mask[idx - unwind]:
                    break
                if token_seq[idx - unwind] != token_seq[-unwind - 1]:
                    break
                match_len = unwind

            if match_len > 0:
                next_tok = token_seq[idx + 1]
                new_len = match_len + 1

                ngram_lens[next_tok] = max(ngram_lens[next_tok].item(), new_len)

                if new_len >= EARLY_EXIT_MATCH_LEN:
                    break

        penalty_mask = ngram_lens > 0
        if penalty_mask.any():
            scales = bases[irow] ** (ngram_lens[penalty_mask] - min_ngram)
            logits[irow][penalty_mask] -= multipliers[irow] * scales

    return logits

def _apply_no_repeat_ngram(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    ngram_size: torch.Tensor,
) -> torch.Tensor:
    """Apply no-repeat-ngram penalty which sets logits to -inf for tokens that 
    would create a repeated n-gram.
    """
    if torch.all(ngram_size == 0):
        return logits

    batch_size = logits.shape[0]

    for i in range(batch_size):
        size = int(ngram_size[i].item())
        if size == 0:
            continue

        cur_len = len(input_ids[i])
        if cur_len < size:
            continue

        banned_tokens = _calc_banned_ngram_tokens(
            ngram_size=size,
            prev_input_ids=input_ids[i],
            cur_len=cur_len-1
        )

        if banned_tokens:
            logits[i, banned_tokens] = -float("inf")

    return logits

def _apply_top_k_top_p(
    logits: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    # Apply top-k.
    top_k_mask = logits_sort.size(1) - k.to(torch.long)
    # Get all the top_k values.
    top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
    top_k_mask = logits_sort < top_k_mask
    logits_sort.masked_fill_(top_k_mask, -float("inf"))

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    # at least one
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = torch.empty_like(logits_sort).scatter_(dim=-1,
                                                    index=logits_idx,
                                                    src=logits_sort)
    return logits


def _apply_min_p(
    logits: torch.Tensor,
    min_p: torch.Tensor,
) -> torch.Tensor:
    """
    Adapted from
    https://github.com/oobabooga/text-generation-webui/blob/3146124ec01f02c8fb1650a6517cf1b60b537aaf/modules/sampler_hijack.py#L16C17-L16C17
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    scaled_min_p = min_p.unsqueeze_(dim=1) * top_probs
    tokens_to_remove = probs < scaled_min_p
    logits = logits.masked_fill_(tokens_to_remove, -float("inf"))

    return logits


def _apply_top_a(
    logits: torch.Tensor,
    top_a: torch.Tensor,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    threshold = torch.pow(top_probs, 2) * top_a.unsqueeze_(dim=1)
    tokens_to_remove = probs < threshold
    logits = logits.masked_fill_(tokens_to_remove, -float("inf"))

    return logits


def _apply_tfs(
    logits: torch.Tensor,
    tfs: torch.Tensor,
) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)
    d2 = logits_sort.softmax(dim=-1).diff().diff().abs()
    normalized_d2 = d2 / torch.sum(d2, dim=-1, keepdim=True)
    curvature_cdf = torch.cumsum(normalized_d2, dim=-1)

    tfs_mask = curvature_cdf > tfs.unsqueeze(dim=-1)

    tfs_mask = torch.cat(
        (
            torch.zeros(
                logits.shape[0], 1, dtype=torch.bool, device=logits.device),
            tfs_mask,
            torch.ones(
                logits.shape[0], 1, dtype=torch.bool, device=logits.device),
        ),
        dim=-1,
    )

    logits_sort[tfs_mask] = -float("inf")
    logits = torch.gather(logits_sort,
                          dim=-1,
                          index=torch.argsort(logits_idx, dim=-1))

    return logits


def _apply_eta_cutoff(
    logits: torch.Tensor,
    eta_cutoff: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = torch.log_softmax(logits, dim=-1)
    probs = shifted_logits.exp()

    neg_entropy = (probs * shifted_logits).nansum(dim=-1)
    eps = torch.min(eta_cutoff,
                    torch.sqrt(eta_cutoff) *
                    torch.exp(neg_entropy)).unsqueeze(dim=1)

    eta_mask = probs < eps

    # guard against nulling out all the logits
    top_idx = torch.argmax(probs, dim=1, keepdim=True)
    eta_mask.scatter_(dim=1, index=top_idx, value=False)

    logits[eta_mask] = -float("inf")
    return logits


def _apply_epsilon_cutoff(
    logits: torch.Tensor,
    epsilon_cutoff: torch.Tensor,
) -> torch.Tensor:
    probs = logits.softmax(dim=-1)

    eps_mask = probs < epsilon_cutoff.unsqueeze(dim=1)

    # guard against nulling out all the logits
    top_idx = torch.argmax(probs, dim=1, keepdim=True)
    eps_mask.scatter_(dim=1, index=top_idx, value=False)

    logits[eps_mask] = -float("inf")
    return logits


def _apply_typical_sampling(
    logits: torch.Tensor,
    typical_p: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = torch.log_softmax(logits, dim=-1)
    probs = shifted_logits.exp()

    neg_entropy = (probs * shifted_logits).nansum(dim=-1, keepdim=True)

    surprisal_deviations = (neg_entropy - shifted_logits).abs()
    _, indices = torch.sort(surprisal_deviations)
    reordered_probs = probs.gather(-1, indices)
    typ_mask_sorted = reordered_probs.cumsum(dim=-1) >= typical_p.unsqueeze(
        dim=1)

    min_tokens_to_keep = 1
    # Keep at least min_tokens_to_keep
    typ_mask_sorted[..., :min_tokens_to_keep] = 0

    typ_mask = typ_mask_sorted.scatter(1, indices, typ_mask_sorted)
    logits[typ_mask] = -float("inf")
    return logits


def _apply_quadratic_sampling(
    logits: torch.Tensor,
    smoothing_factor: torch.Tensor,
    smoothing_curve: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a quadratic transformation to the logits based on the
    provided smoothing factors and curves. The transformation is
    centered around the maximum logit value in the batch.
    The transformation involves a quadratic and cubic term, with the
    cubic term controlled by the smoothing curve. The quadratic term is
    scaled by the smoothing factor, and the cubic term is scaled by the
    product of the smoothing factor and the smoothing curve.
    params:
        logits (torch.Tensor): The logits to be transformed.
        smoothing_factors (torch.Tensor): The factors to scale the quadratic
            term in the transformation.
        smoothing_curves (torch.Tensor): The factors to scale the cubic term
            in the transformation.
    returns:
        torch.Tensor: The transformed logits.
    Credits: @kalomaze
    """
    mask = smoothing_factor != 0

    smoothing_factor.unsqueeze_(dim=1)
    smoothing_curve.unsqueeze_(dim=1)
    k = smoothing_factor * (3 - smoothing_curve) / 2
    s = smoothing_factor * (smoothing_curve - 1) / 2

    quadlogits = logits[mask]  # limit to logits using this sampler
    max_logits = quadlogits.max(dim=-1, keepdim=True).values

    # Construct the delta from each logit to its new value
    diff = quadlogits - max_logits
    diff -= diff**2 * (s[mask] * diff - k[mask])
    diff[diff != diff] = 0  # Eliminate NaNs due to infs

    logits[mask] -= diff
    return logits


def _apply_xtc_sampling(
    logits: torch.Tensor,
    xtc_thresholds: torch.Tensor,
    xtc_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Apply Exclude Top Choices (XTC) sampling to the logits.
    Reference: https://github.com/oobabooga/text-generation-webui/pull/6335

    Args:
        logits: (num_tokens, vocab_size) The input logits.
        xtc_thresholds: (num_tokens,) The threshold for each token.
        xtc_probabilities: (num_tokens,) The probability of applying XTC
            for each token.

    Returns:
        torch.Tensor: The modified logits.
    """
    apply_xtc = torch.rand_like(xtc_probabilities) < xtc_probabilities

    if not apply_xtc.any():
        return logits

    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Find indices where the next probability is above the threshold
    # Skips the top choice, which later on becomes skipping the last choice.
    above_threshold = sorted_probs[..., 1:] >= xtc_thresholds.unsqueeze(-1)
    
    # Apply XTC only to rows where it should be applied
    for i in range(logits.shape[0]):
        if apply_xtc[i]:
            # Count logits above the threshold (skipping the first)
            indices_to_remove = above_threshold[i].count_nonzero(dim=-1).item()
            if indices_to_remove > 0:
                # Implies the top logit and at least one other is >= threshold.
                # Mask out above_thresh logits except the last/lowest one.
                logits[i].scatter_(
                    0, sorted_indices[i, :indices_to_remove], -float('inf'))

    return logits


def _apply_top_nsigma(
        logits: torch.Tensor,
        nsigma: torch.Tensor,
) -> torch.Tensor:
    """Apply top-nsigma truncation to the logits.
    
    Reference: https://arxiv.org/abs/2411.07641

    Args:
        logits: Logits of shape (num_tokens, vocab_size)
        nsigma: Number of standard deviations to use as threshold
    Returns:
        Modified logits with values below threshold set to -inf
    """
    std = logits.std(dim=-1, keepdim=True) 
    threshold = (logits.max(dim=-1, keepdim=True).values -
                 nsigma.unsqueeze(dim=1) * std)
    logits[logits < threshold] = float("-inf")

    return logits


def _greedy_sample(
    selected_seq_groups: List[SequenceGroupToSample],
    samples: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    """Run greedy sampling on a given samples.
    Args:
        selected_seq_groups: A list of sequence groups batched.
        samples: (num_selected_samples,) A tensor of samples. The length of
            samples could be smaller than selected_seq_groups if
            seq_group.do_sample is False.
    Returns:
        Tuple of (next_token_ids, parent_ids). The length of returned list is
        same as the length of selected_seq_groups. If the corresponding
        seq_group has do_sample=False, tuple contains ([], [])
    """
    samples = samples.tolist()
    sample_idx = 0
    results = []
    for seq_group in selected_seq_groups:
        if not seq_group.do_sample:
            results.append(([], []))
            continue

        seq_ids = seq_group.seq_ids
        num_parent_seqs = len(seq_ids)
        assert num_parent_seqs == 1, (
            "Greedy sampling should have only one seq.")
        parent_ids = list(range(num_parent_seqs))
        next_token_ids = [samples[sample_idx]]
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


def _random_sample(
    selected_seq_groups: List[SequenceGroupToSample],
    random_samples: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    """Run random sampling on a given samples.
    Args:
        selected_seq_groups: A list of sequence groups batched.
        random_samples: (num_selected_samples,) A tensor of samples. The
            length of samples could be smaller than selected_seq_groups if
            seq_group.do_sample is False.
    Returns:
        Tuple of (next_token_ids, parent_ids). The length of returned list is
        same as the length of selected_seq_groups. If the corresponding
        seq_group has do_sample=False, tuple contains ([], [])
    """
    # Find the maximum best_of value of the prompt phase requests.
    random_samples = random_samples.cpu()
    sample_idx = 0
    results = []
    for seq_group in selected_seq_groups:
        if not seq_group.do_sample:
            results.append(([], []))
            continue

        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params
        is_prompt = seq_group.is_prompt
        num_parent_seqs = len(seq_ids)
        if is_prompt:
            # Prompt phase.
            parent_ids = [0] * sampling_params.best_of
            next_token_ids = random_samples[
                sample_idx, :sampling_params.best_of].tolist()
        else:
            # Generation phase.
            parent_ids = list(range(num_parent_seqs))
            next_token_ids = random_samples[sample_idx:sample_idx +
                                            num_parent_seqs, 0].tolist()
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


def _beam_search_sample(
    selected_seq_groups: List[SequenceGroupToSample],
    logprobs: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    """Run beam sampling on a given samples.
    Args:
        selected_seq_groups: A list of sequence groups batched.
        logprobs: (num_selected_samples, vocab_size,) A tensor of logprob
        on selected sample indices.
    Returns:
        Tuple of (next_token_ids, parent_ids). The length of returned list is
        same as the length of selected_seq_groups. If the corresponding
        seq_group has do_sample=False, tuple contains ([], [])
    """
    # We sample 2 * beam_width candidates to make sure that with high
    # probability we can get `beam_width` candidates in addition to
    # the finished sequences for the next iteration. See
    # https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/utils/beam_search.py#L557-L563
    # for details. See also HF reference:
    # https://github.com/huggingface/transformers/blob/a4dd53d88e4852f023332d284ff07a01afcd5681/src/transformers/generation/utils.py#L3063-L3065
    #
    # NOTE: Beam search is not vectorized, so its speed can be slower than
    # other sampling methods.
    sample_idx = 0
    results = []
    for seq_group in selected_seq_groups:
        if not seq_group.do_sample:
            results.append(([], []))
            continue

        is_prompt = seq_group.is_prompt
        seq_ids, sampling_params = seq_group.seq_ids, seq_group.sampling_params
        num_parent_seqs = len(seq_ids)
        beam_width = sampling_params.best_of
        seq_group_logprobs = logprobs[sample_idx:sample_idx + num_parent_seqs]
        if is_prompt:
            # Prompt phase.
            assert num_parent_seqs == 1, (
                "Prompt input should have only one seq.")
            parent_ids = [0] * (2 * beam_width)
            _, next_token_ids = torch.topk(seq_group_logprobs[0],
                                           2 * beam_width)
            next_token_ids = next_token_ids.tolist()
        else:
            # Generation phase.
            cumulative_logprobs = [
                seq_group.seq_data[seq_id].cumulative_logprob
                for seq_id in seq_ids
            ]
            cumulative_logprobs = torch.tensor(
                cumulative_logprobs,
                dtype=torch.float,
                device=seq_group_logprobs.device)
            seq_group_logprobs = (seq_group_logprobs +
                                  cumulative_logprobs.unsqueeze(dim=1))
            _, topk_ids = torch.topk(seq_group_logprobs.flatten(),
                                     2 * beam_width)
            topk_ids = topk_ids.tolist()
            vocab_size = seq_group_logprobs.size(-1)
            parent_ids = [i // vocab_size for i in topk_ids]
            next_token_ids = [i % vocab_size for i in topk_ids]
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    assert sample_idx == logprobs.size(0)
    return results


# torch.multinomial forces a GPU<->CPU sync.
# Therefore, we use an optimized implementation instead.
# Note that we always sample with replacement.
# probs will be modified in place, but this is fine, as we pass
# in a copy already.
def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
    seq_groups: Optional[List[SequenceGroupToSample]] = None,
) -> torch.Tensor:
    if num_samples > 1:
        probs = probs.repeat_interleave(num_samples, dim=0)
    q = torch.empty_like(probs)
    if seq_groups is None:
        q.exponential_()
    else:
        sample_idx = 0
        for seq_group in seq_groups:
            seq_ids = seq_group.seq_ids
            stride = len(seq_ids) * num_samples
            assert seq_group.generator is not None
            q[sample_idx:sample_idx +
              stride].exponential_(generator=seq_group.generator)
            sample_idx += stride
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)


def _top_k_top_p_multinomial_with_kernels(
        probs: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor,
        num_samples: int, seq_groups: Optional[List[SequenceGroupToSample]]):
    max_top_k_round = 32
    if num_samples > 1:
        probs = probs.repeat_interleave(num_samples, dim=0)
        top_ks = top_ks.repeat_interleave(num_samples)
        top_ps = top_ps.repeat_interleave(num_samples)
    batch_size = probs.shape[0]
    uniform_samples = torch.empty((max_top_k_round, batch_size),
                                  device=probs.device)
    if seq_groups is None:
        uniform_samples.uniform_()
    else:
        sample_idx = 0
        for seq_group in seq_groups:
            seq_ids = seq_group.seq_ids
            stride = len(seq_ids) * num_samples
            assert seq_group.generator is not None
            uniform_samples[:, sample_idx:sample_idx +
                            stride].uniform_(generator=seq_group.generator)
            sample_idx += stride
    batch_next_token_ids, success = ops.top_k_top_p_sampling_from_probs(
        probs,
        uniform_samples,
        top_ks,
        top_ps,
    )
    if not success.all():
        warnings.warn("CUDA rejection sampling failed, fallback.",
                      stacklevel=1)
        probs = ops.top_k_renorm_prob(probs, top_ks)
        probs = ops.top_p_renorm_prob(probs, top_ps)
        batch_next_token_ids = ops.sampling_from_probs(
            probs, uniform_samples[0])
    return batch_next_token_ids.view(-1, num_samples)


def get_pythonized_sample_results(
        sample_result_args: SampleResultArgsType) -> SampleResultType:
    """This function consumes GPU-side sampler results and computes
    Pythonized CPU-side sampler results (GPU -> CPU sync.)
    Single-step scheduling: this function is invoked at sampling-time
    for immediate Pythonization.
    Multi-step scheduling: Pythonization is deferred until after multiple
    GPU-side steps have been completed.

    Args:
      sample_result_args: GPU-side inputs to the Pythonization process
    Returns:
      Pythonized sampler results
    """

    (
        sample_metadata,
        sampling_metadata,
        greedy_samples,
        multinomial_samples,
        beam_search_logprobs,
        sample_results_dict,
    ) = (
        sample_result_args.sample_metadata,
        sample_result_args.sampling_metadata,
        sample_result_args.greedy_samples,
        sample_result_args.multinomial_samples,
        sample_result_args.beam_search_logprobs,
        sample_result_args.sample_results_dict,
    )

    for sampling_type in SamplingType:
        if sampling_type not in sample_metadata:
            continue
        (seq_group_id, seq_groups) = sample_metadata[sampling_type]
        if sampling_type == SamplingType.GREEDY:
            sample_results = _greedy_sample(seq_groups, greedy_samples)
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            sample_results = _random_sample(seq_groups,
                                            multinomial_samples[sampling_type])
        elif sampling_type == SamplingType.BEAM:
            sample_results = _beam_search_sample(seq_groups,
                                                 beam_search_logprobs)
        sample_results_dict.update(zip(seq_group_id, sample_results))

    return [
        sample_results_dict.get(i, ([], []))
        for i in range(len(sampling_metadata.seq_groups))
    ]


def _sample_with_torch(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sampling_tensors: SamplingTensors,
    include_gpu_probs_tensor: bool,
    modify_greedy_probs: bool,
) -> SampleReturnType:
    """Torch-oriented _sample() implementation.

    Single-step scheduling: 
    * Perform GPU-side sampling computation
    * Immediately Pythonize sampling result

    Multi-step scheduling:
    * Perform GPU-side sampling computation
    * Defer Pythonization & preserve GPU-side
      tensors required for Pythonization
    """
    categorized_seq_group_ids = {t: [] for t in SamplingType}
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        sampling_params = seq_group.sampling_params
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)

    sample_results_dict: SampleResultsDictType = {}
    sample_metadata: SampleMetadataType = {}
    multinomial_samples: MultinomialSamplesType = {}
    greedy_samples: Optional[torch.Tensor] = None
    beam_search_logprobs: Optional[torch.Tensor] = None
    # Create output tensor for sampled token ids.
    if include_gpu_probs_tensor:
        sampled_token_ids_tensor = torch.empty(logprobs.shape[0],
                                               1,
                                               dtype=torch.long,
                                               device=logprobs.device)
    else:
        sampled_token_ids_tensor = None
    # Counterintuitively, having two loops here is actually faster.
    # The first loop can run without waiting on GPU<->CPU sync.
    for sampling_type in SamplingType:
        sample_indices = categorized_sample_indices[sampling_type]
        num_tokens = len(sample_indices)
        if num_tokens == 0:
            continue

        seq_group_id = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_id]
        sample_metadata[sampling_type] = (seq_group_id, seq_groups)
        long_sample_indices = sample_indices.long()
        if sampling_type == SamplingType.GREEDY:
            greedy_samples = torch.argmax(logprobs[long_sample_indices],
                                          dim=-1)
            if include_gpu_probs_tensor:
                # Store sampled tokens in output tensor.
                sampled_token_ids_tensor[
                    long_sample_indices] = greedy_samples.unsqueeze(-1)
            if modify_greedy_probs:
                # If required, modify the probabilities such that sampling from
                # the modified distribution would always sample the argmax
                # token id.
                _modify_greedy_probs_inplace(logprobs, probs,
                                             long_sample_indices,
                                             greedy_samples)

        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            max_best_of_in_batch = 1
            for seq_group in seq_groups:
                if seq_group.is_prompt:
                    sampling_params = seq_group.sampling_params
                    max_best_of_in_batch = max(max_best_of_in_batch,
                                               sampling_params.best_of)

            seq_groups_arg = (None if sampling_type == SamplingType.RANDOM else
                              seq_groups)
            if APHRODITE_USE_SAMPLING_KERNELS and not is_cpu():
                multinomial_samples[
                    sampling_type] = _top_k_top_p_multinomial_with_kernels(
                        probs[long_sample_indices],
                        sampling_tensors.top_ks[long_sample_indices],
                        sampling_tensors.top_ps[long_sample_indices],
                        max_best_of_in_batch,
                        seq_groups_arg,
                    )
            else:
                multinomial_samples[sampling_type] = _multinomial(
                    probs[long_sample_indices],
                    max_best_of_in_batch,
                    seq_groups=seq_groups_arg)

            if sampled_token_ids_tensor is not None:
                # Store sampled tokens in output tensor.
                sampled_token_ids_tensor[long_sample_indices] = \
                    multinomial_samples[sampling_type].to(torch.long)

        elif sampling_type == SamplingType.BEAM:
            beam_search_logprobs = logprobs[sample_indices]
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

    # Encapsulate arguments for computing Pythonized sampler
    # results, whether deferred or otherwise.
    maybe_deferred_args = SampleResultArgsType(
        sampling_metadata=sampling_metadata,
        sample_metadata=sample_metadata,
        multinomial_samples=multinomial_samples,
        greedy_samples=greedy_samples,
        beam_search_logprobs=beam_search_logprobs,
        sample_results_dict=sample_results_dict)

    if not sampling_metadata.skip_sampler_cpu_output:
        # GPU<->CPU sync happens here.
        # This also converts the sampler output to a Python object.
        # Return Pythonized sampler result & sampled token ids
        return get_pythonized_sample_results(
            maybe_deferred_args), sampled_token_ids_tensor
    else:
        # Defer sampler result Pythonization; return deferred
        # Pythonization args & sampled token ids
        return (
            maybe_deferred_args,
            sampled_token_ids_tensor,
        )


def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sampling_tensors: SamplingTensors,
    include_gpu_probs_tensor: bool,
    modify_greedy_probs: bool,
) -> SampleReturnType:
    """
    Args:
        probs: (num_query_tokens_in_batch, num_vocab)
        logprobs: (num_query_tokens_in_batch, num_vocab)
        sampling_metadata: The metadata for a batch for sampling.
        sampling_tensors: Tensors that include sampling related metadata.
    Returns:
        (next_token_ids, parent_seq_ids) for each seq group in a batch.
            If sampling is skipped, it returns ([], [])
        sampled_token_ids_tensor: A tensor of sampled token ids.    
    """
    return _sample_with_torch(
        probs,
        logprobs,
        sampling_metadata,
        sampling_tensors,
        include_gpu_probs_tensor=include_gpu_probs_tensor,
        modify_greedy_probs=modify_greedy_probs,
    )


def _get_ranks(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    This function calculates the ranks of the chosen tokens in a logprob tensor.
    Args:
        x (torch.Tensor): 2D logprob tensor of shape (N, M)
                        where N is the no. of tokens and M is the vocab dim.
        indices (torch.Tensor): List of chosen token indices.
    Returns:
        torch.Tensor: 1D tensor of shape (N,) where N is the no. of tokens.
                    Each element in the returned tensor represents the rank 
                    of the chosen token in the input logprob tensor.
    """
    vals = x[torch.arange(0, len(x), device=x.device, dtype=indices.dtype),
             indices]
    return (x > vals[:, None]).long().sum(1).add_(1)


def get_logprobs(
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sample_results: List[Tuple[List[int], List[int]]],
) -> Tuple[List[Optional[PromptLogprobs]], List[SampleLogprobs]]:
    """Return sample lobprobs and prompt logprobs.
    The logic consists of 3 parts.
    - Select indices to compute logprob from, ranks of token ids, and
        the top k token ids from logprobs.
    - Compute prompt logprobs if required.
    - Compute sample logprobs if required.
    Args:
        logprobs: (num_query_tokens_across_batch, num_vocab). Each query token's
            logprob per vocab. Sequence groups' query tokens are batched in a
            single flattened tensor. For example, assuming there are N
            seq groups, it is sorted by prefill tokens for seq_group_1 (if
            prompt logprob is enabled), decode tokens for seq_group_1 (if
            sampling is required), prefill tokens for seq_group_2, ...
        sampling_metadata: The sampling metadata.
        sample_results: (num_seq_groups) The tuple of (next_token_ids,
            parent_ids) for each sequence group. When beam search is enabled,
            sample_results can contain different number of seq_ids from
            sampling_metadata.seq_groups. It is because beam search creates
            2 * BEAM_WIDTH number of samples (whereas there are only up to
            BEAM_WIDTH number of seq_ids).
    Returns:
        A tuple of prompt and sample logprobs per sequence group in a batch.
    """
    # The index of query token to calculate logprobs. It includes both
    # prompt and sample logprob indices.
    query_indices: List[int] = []
    # The next token ids to get the logprob value from.
    next_token_ids: List[int] = []
    # The largest requested number of logprobs. We find logprobs as many as the
    # largest num logprobs in this API. If every logprobs is None, it will be
    # set to -1.
    largest_num_logprobs = -1
    # If beam search is enabled.
    use_beam_search = False

    # Select indices to compute logprob from, ranks of token ids, and the top
    # k token ids from logprobs.
    for (seq_group, sample_result) in zip(sampling_metadata.seq_groups,
                                          sample_results):
        sampling_params = seq_group.sampling_params

        # Update indices and tokens for prompt logprobs.
        if (seq_group.is_prompt
                and sampling_params.prompt_logprobs is not None):
            largest_num_logprobs = max(largest_num_logprobs,
                                       sampling_params.prompt_logprobs)
            next_prompt_tokens = _get_next_prompt_tokens(seq_group)
            query_indices.extend(seq_group.prompt_logprob_indices)
            next_token_ids.extend(next_prompt_tokens)

        # Update indices and next tokenes for sample logprob.
        if seq_group.do_sample:
            token_ids, parent_seq_ids = sample_result
            # NOTE: We cannot directly use sample_indices because
            # sample_indices only contain parent seq_ids of a previous step.
            # The current step may have different number of seq_ids, and
            # we can obtain it from `sample_result[1]`.
            query_idx = seq_group.sample_indices[0]
            query_indices.extend(
                [query_idx + parent_id for parent_id in parent_seq_ids])
            next_token_ids.extend(token_ids)

            if sampling_params.logprobs is not None:
                largest_num_logprobs = max(largest_num_logprobs,
                                           sampling_params.logprobs)

            use_beam_search = use_beam_search or sampling_params.use_beam_search

        assert len(next_token_ids) == len(query_indices)

    if len(query_indices) == 0:
        empty_sampled_logprob = []
        empty_prompt_logprob = None
        return [empty_prompt_logprob], [empty_sampled_logprob]

    selected_logprobs, ranks = None, None
    top_logprobs, top_token_ids = None, None

    # If largest_num_logprobs == -1, i.e. no logprobs are requested, we can
    # skip the whole logprob calculation.
    if largest_num_logprobs >= 0 or use_beam_search:
        query_indices_gpu = torch.tensor(query_indices, device=logprobs.device)
        next_token_ids_gpu = torch.tensor(next_token_ids,
                                          device=logprobs.device)

        # (num_selected_query_tokens, num_logprobs). Note that query_indices can
        # contain duplicates if beam search is enabled.
        selected_logprobs = logprobs[[
            query_indices_gpu,
            next_token_ids_gpu,
        ]]
        ranks = _get_ranks(
            logprobs[query_indices_gpu],
            next_token_ids_gpu,
        )
        assert selected_logprobs.shape[0] == ranks.shape[0]

        # We need to compute top k only if there exists logprobs > 0.
        if largest_num_logprobs > 0:
            # Logprobs of topk tokens for a batch of sequence groups.
            # (num_query_tokens_across_batch).
            top_logprobs, top_token_ids = torch.topk(logprobs,
                                                     largest_num_logprobs,
                                                     dim=-1)
            top_logprobs = top_logprobs.to('cpu')
            top_token_ids = top_token_ids.to('cpu')

        selected_logprobs = selected_logprobs.to('cpu')
        ranks = ranks.to('cpu')

    # Find prompt/sample logprobs.
    prompt_logprobs_per_seq_group: List[Optional[PromptLogprobs]] = []
    sample_logprobs_per_seq_group: List[SampleLogprobs] = []
    top_logprob_idx = 0
    selected_logprobs_idx = 0

    for seq_group, sample_result in zip(sampling_metadata.seq_groups,
                                        sample_results):
        (prompt_logprobs, top_logprob_idx,
         selected_logprobs_idx) = _get_prompt_logprob_if_needed(
             seq_group, selected_logprobs, ranks, top_token_ids, top_logprobs,
             selected_logprobs_idx, top_logprob_idx)
        prompt_logprobs_per_seq_group.append(prompt_logprobs)

        (sampled_logprobs, top_logprob_idx,
         selected_logprobs_idx) = _get_sampled_logprob_if_needed(
             seq_group, sample_result, selected_logprobs, ranks, top_token_ids,
             top_logprobs, selected_logprobs_idx, top_logprob_idx)
        sample_logprobs_per_seq_group.append(sampled_logprobs)

    return prompt_logprobs_per_seq_group, sample_logprobs_per_seq_group


def _get_prompt_logprob_if_needed(
    seq_group: SequenceGroupToSample,
    selected_logprobs: torch.Tensor,
    ranks: torch.Tensor,
    top_token_ids: torch.Tensor,
    top_logprobs: torch.Tensor,
    selected_logprobs_idx: int,
    top_logprob_idx: int,
):
    """Compute the prompt logprob from a sequence group if needed."""
    sampling_params = seq_group.sampling_params
    is_prompt = seq_group.is_prompt

    # Find prompt logprobs
    prompt_logprobs: Optional[PromptLogprobs] = None
    if is_prompt and sampling_params.prompt_logprobs is not None:
        prompt_logprobs = []
        num_logprobs = sampling_params.prompt_logprobs
        next_prompt_tokens = _get_next_prompt_tokens(seq_group)
        # Pre-select indexes and create a list. It is faster than calling .item
        # repetitively.
        selected_logprob_items = selected_logprobs[
            selected_logprobs_idx:selected_logprobs_idx +
            len(next_prompt_tokens)].tolist()
        rank_items = ranks[selected_logprobs_idx:selected_logprobs_idx +
                           len(next_prompt_tokens)].tolist()

        for idx, token_id in enumerate(next_prompt_tokens):
            # Calculate the prompt logprob of the real prompt tokens.
            # {token_id: (logprob, rank_from_vocab)}
            prompt_logprobs_dict: Dict[int, Tuple[float, int]] = {
                token_id: (selected_logprob_items[idx], rank_items[idx])
            }

            # Add top K prompt logprobs along with its rank.
            if num_logprobs > 0:
                top_ids = top_token_ids[
                    top_logprob_idx, :num_logprobs].tolist()
                top_probs = top_logprobs[
                    top_logprob_idx, :num_logprobs].tolist()
                # Top K is already sorted by rank, so we can use 1 ~
                # num_logprobs + 1 for rank.
                top_ranks = range(1, num_logprobs + 1)
                prompt_logprobs_dict.update({
                    top_id: (top_prob, rank)
                    for top_id, top_prob, rank in zip(top_ids, top_probs,
                                                      top_ranks)
                })
            prompt_logprobs.append({
                token_id: Logprob(*logprob_and_rank)
                for token_id, logprob_and_rank in prompt_logprobs_dict.items()
            })
            # + 1 to go to the next prompt token.
            top_logprob_idx += 1

        # + len(next_prompt_tokens) to go to the next prompt.
        selected_logprobs_idx += len(next_prompt_tokens)
    return prompt_logprobs, top_logprob_idx, selected_logprobs_idx


def _get_sampled_logprob_if_needed(
    seq_group: SequenceGroupToSample,
    sample_result: Tuple[List[int], List[int]],
    selected_logprobs: torch.Tensor,
    ranks: torch.Tensor,
    top_token_ids: torch.Tensor,
    top_logprobs: torch.Tensor,
    selected_logprobs_idx: int,
    top_logprob_idx: int,
):
    """Compute the sample logprob if needed."""
    seq_ids = seq_group.seq_ids
    num_logprobs = seq_group.sampling_params.logprobs
    use_beam_search = seq_group.sampling_params.use_beam_search
    sampled_logprobs: SampleLogprobs = []
    next_token_ids, parent_seq_ids = sample_result

    if seq_group.do_sample:
        assert len(next_token_ids) > 0
        if num_logprobs is None and not use_beam_search:
            for next_token_id in next_token_ids:
                # Use a dummy logprob
                sampled_logprobs.append({next_token_id: Logprob(inf)})
        else:
            # Pre-select items from tensor. tolist() is faster than repetitive
            # `.item()` calls.
            selected_logprob_items = selected_logprobs[
                selected_logprobs_idx:selected_logprobs_idx +
                len(next_token_ids)].tolist()
            rank_items = ranks[selected_logprobs_idx:selected_logprobs_idx +
                               len(next_token_ids)].tolist()
            for idx, (next_token_id, parent_id) in enumerate(
                    zip(next_token_ids, parent_seq_ids)):
                # Get the logprob of a sampled token.
                sampled_logprobs_dict = {
                    next_token_id:
                    (selected_logprob_items[idx], rank_items[idx])
                }
                if num_logprobs is not None and num_logprobs > 0:
                    # Get top K logprobs.
                    top_ids = top_token_ids[top_logprob_idx +
                                            parent_id, :num_logprobs].tolist()
                    top_probs = top_logprobs[
                        top_logprob_idx + parent_id, :num_logprobs].tolist()
                    # Top K is already sorted by rank, so we can use 1 ~
                    # num_logprobs + 1 for rank.
                    top_ranks = range(1, num_logprobs + 1)
                    sampled_logprobs_dict.update({
                        top_id: (top_prob, rank)
                        for top_id, top_prob, rank in zip(
                            top_ids, top_probs, top_ranks)
                    })

                sampled_logprobs.append({
                    token_id: Logprob(*logprob_and_rank)
                    for token_id, logprob_and_rank in
                    sampled_logprobs_dict.items()
                })

        # NOTE: This part of code is not intuitive. `selected_logprobs` include
        # logprobs for the current step, which has len(next_token_ids) tokens
        # per sequence group. `logprobs` includes logprobs from the previous
        # steps, which has len(seq_ids) tokens per sequence group.
        # Iterate to the next sequence group in a batch.
        selected_logprobs_idx += len(next_token_ids)
        # Iterate to the next sequence group in a batch.
        top_logprob_idx += len(seq_ids)
    return sampled_logprobs, top_logprob_idx, selected_logprobs_idx


def _modify_greedy_probs_inplace(logprobs: torch.Tensor, probs: torch.Tensor,
                                 sample_indices: torch.Tensor,
                                 greedy_samples: torch.Tensor) -> None:
    """Modify the probability distributions of the greedily-sampled tokens such
    that each sampled token has a "probability" of 1.0. This is required by
    speculative decoding, which depends on the sampling method being encoded
    within the probability distribution for correctness.
    # Why do we only need to do this for greedy sampling?
    Aphrodite's sampler performs the following steps for greedy or multinomial
    (random) sampling:
        1. Get logits from model.
        2. Modify logits according to per-sequence sampling parameters.
            - Multiply by temperature, top-k and top-p masking, penalize tokens
                according to their frequency, etc.
        3. Sample a token.
            - Random sampling simply samples from the modified probability
                distribution.
            - Greedy sampling performs `argmax` to obtain the token with the
                highest likelihood.
    
    Ignoring greedy sampling for a moment, we find that the computed probability
    distribution has the following property: we can sample from it independently
    and find that the token sampled by the Sampler has a frequency corresponding
    to how often we see it in our sampling. In other words, for tokens sampled
    with Aphrodite's random SamplingType, the computed probability distribution
    encodes the sampling methodology completely.
    Greedy sampling does not normally have this property. Aphrodite modifies
    logits according to sampling params, then performs `argmax`, then returns
    the sampled token and the computed probability distribution. If we sample
    from the distribution, we'll find the likelihood of the greedily-sampled
    token is not always 1.0.
    Since lossless speculative decoding requires that the sampling methodology
    be encoded within the probability distribution, we are motivated to modify
    the probability distribution such that the sampled token has probability 1
    when speculative decoding is used.
    NOTE: Alternatively, we could use an extremely low temperature to achieve
    greedy sampling using multinomial computation and unite the codepaths. This
    has implications on the overall design of the sampler, e.g. how to record
    accurate logprobs for the user, so this improvement is deferred to later.
    """
    # NOTE: logprobs are not modified so they can be returned to the user.
    probs[sample_indices, :] = 0
    probs[sample_indices, greedy_samples] = 1.0


def _build_sampler_output(
    maybe_deferred_sample_results: MaybeDeferredSampleResultType,
    sampling_metadata: SamplingMetadata,
    prompt_logprobs: Optional[List[Optional[PromptLogprobs]]],
    sample_logprobs: Optional[List[SampleLogprobs]],
    on_device_tensors: Optional[Tuple[torch.Tensor, torch.Tensor,
                                      torch.Tensor]],
    skip_sampler_cpu_output: bool = False,
) -> SamplerOutput:
    """Construct Python objects with the output of sampling.
    Args:
        on_device_tensors: Tuple containing on-device tensors with the
            probabilities used in sampling and the sampled token ids. This
            allows post-processing without copies to CPU/serialization, e.g. in
            speculative decoding rejection sampling.
    """
    sampler_output: List[CompletionSequenceGroupOutput] = []

    if skip_sampler_cpu_output:
        assert isinstance(maybe_deferred_sample_results, SampleResultArgsType)
        deferred_sample_results_args = maybe_deferred_sample_results
    else:
        assert prompt_logprobs is not None
        assert sample_logprobs is not None
        assert not isinstance(maybe_deferred_sample_results,
                              SampleResultArgsType)
        deferred_sample_results_args = None

        for (seq_group, sample_result, group_prompt_logprobs,
             group_sample_logprobs) in zip(sampling_metadata.seq_groups,
                                           maybe_deferred_sample_results,
                                           prompt_logprobs, sample_logprobs):
            seq_ids = seq_group.seq_ids
            next_token_ids, parent_ids = sample_result
            seq_outputs: List[SequenceOutput] = []
            for parent_id, next_token_id, logprobs in zip(
                    parent_ids, next_token_ids, group_sample_logprobs):
                seq_outputs.append(
                    SequenceOutput(seq_ids[parent_id], next_token_id,
                                   logprobs))
            sampler_output.append(
                CompletionSequenceGroupOutput(seq_outputs,
                                              group_prompt_logprobs))
    # If not specified, store None values in SamplerOutput.
    if on_device_tensors is not None:
        (sampled_token_probs, logprobs_tensor,
         sampled_token_ids) = on_device_tensors
    else:
        sampled_token_probs, logprobs_tensor, sampled_token_ids = (None, None,
                                                                   None)
    return SamplerOutput(
        outputs=sampler_output,
        sampled_token_probs=sampled_token_probs,
        sampled_token_ids=sampled_token_ids,
        logprobs=logprobs_tensor,
        deferred_sample_results_args=deferred_sample_results_args)


def _get_next_prompt_tokens(seq_group: SequenceGroupToSample) -> List[str]:
    """Get a list of next prompt tokens to compute logprob from a
        given sequence group.
    It is used to compute prompt logprob. Imagine you have logprob for each
    query token. Query token needs to know the next prompt token id to compute
    prompt logprob. This is a helper to obtain next prompt token ids.
    This API has to be used only when the caller knows seq_group is in prefill
    stage.
    Returns:
        A list of next prompt tokens to compute logprob.
    """
    assert seq_group.is_prompt, (
        "Caller should ensure the sequence group is in a prefill stage.")
    seq_ids = seq_group.seq_ids
    query_len = seq_group.query_len
    assert query_len is not None
    # prompt has only 1 seq id.
    assert len(seq_ids) == 1
    seq_data = seq_group.seq_data[seq_ids[0]]
    computed_len = seq_data.get_num_computed_tokens()
    prompt_tokens = seq_data.prompt_token_ids
    # +1 because we are looking for a next prompt token.
    next_token_index_start = computed_len + 1
    next_token_index_end = min(computed_len + query_len + 1,
                               len(prompt_tokens))
    next_prompt_tokens = prompt_tokens[
        next_token_index_start:next_token_index_end]
    return next_prompt_tokens

def _get_ngrams(
    ngram_size: int, 
    prev_input_ids: torch.Tensor
) -> Dict[Tuple[int, ...], List[int]]:
    """Get dictionary of ngrams and the tokens that followed them.

    Args:
        ngram_size: Size of ngrams to track
        prev_input_ids: 1D tensor of previous token ids

    Returns:
        Dictionary mapping ngram tuples to list of tokens that followed them
    """
    generated_ngrams = {}
    gen_tokens = prev_input_ids.tolist()

    for i in range(len(gen_tokens) - ngram_size + 1):
        ngram = tuple(gen_tokens[i:i + ngram_size - 1])
        next_token = gen_tokens[i + ngram_size - 1]
        if ngram in generated_ngrams:
            generated_ngrams[ngram].append(next_token)
        else:
            generated_ngrams[ngram] = [next_token]

    return generated_ngrams

def _get_generated_ngrams(
    banned_ngrams: Dict[Tuple[int, ...], List[int]], 
    prev_input_ids: torch.Tensor,
    ngram_size: int, 
    cur_len: int
) -> List[int]:
    """Get list of tokens that would create a repeated ngram if generated next.

    Args:
        banned_ngrams: Dictionary of previously seen ngrams and their next
            tokens
        prev_input_ids: Previous token ids
        ngram_size: Size of ngrams to check
        cur_len: Current position in sequence

    Returns:
        List of token ids that would create a repeat ngram
    """
    start_idx = cur_len + 1 - ngram_size
    current_ngram = tuple(prev_input_ids[start_idx:cur_len].tolist())

    return banned_ngrams.get(current_ngram, [])

def _calc_banned_ngram_tokens(
    ngram_size: int,
    prev_input_ids: torch.Tensor,
    cur_len: int
) -> List[int]:
    """Calculate tokens that would create repeated ngrams if generated next.

    Args:
        ngram_size: Size of ngrams to prevent repeating
        prev_input_ids: Previous token ids in sequence
        cur_len: Current position in sequence

    Returns:
        List of token ids that should be banned to prevent ngram repetition
    """
    if cur_len + 1 < ngram_size:
        return []

    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids)

    banned_tokens = _get_generated_ngrams(
        generated_ngrams,
        prev_input_ids, 
        ngram_size,
        cur_len
    )

    return banned_tokens


# def _apply_mirostat_v2(logits: torch.Tensor,
#                        sampling_tensors: SamplingTensors) -> torch.Tensor:
#     # Reduce our view to just the affected logits
#     logit_view = logits[sampling_tensors.miro_indices]

#     # Calculate surprise value per token
#     #  Convert nats to bits for compatibility with ooba/kobold parameters.
#     logit_surprise = torch.log_softmax(logit_view, dim=-1) / -math.log(2)

#     # Mask out "too-surprising" tokens (surprisal > mu)
#     mus = sampling_tensors.miro_mus
#     miro_mask = logit_surprise > mus.unsqueeze(dim=-1)

#     # Unmask most-likely logit to guarantee a selection.
#     maxinds = torch.argmax(logit_view, dim=-1, keepdim=True)
#     miro_mask.scatter_(dim=1, index=maxinds, value=False)

#     # Apply logit mask (effectively a top-k filter).
#     logit_view[miro_mask] = -float("inf")

#     # Project logit changes made to the view onto the original.
#     # I think this step might be redundant.
#     logits[sampling_tensors.miro_indices] = logit_view
#     return logits

# def _mirostat_store_args(logits: torch.Tensor, args: SamplingTensors,
#                          sample_results: List[Tuple[List[int], List[int]]],
#                          sampling_metadata: SamplingMetadata,
#                          output_metadata: OutputMetadata) -> None:
#     """Based on whichever token was finally sampled, we calculate the
#     final surprisal values to update the mus.

#     Because a single sequence can have multiple samples, we must fork
#     the mu accordingly."""
#     assert sampling_metadata.seq_groups is not None
#     seqid_to_tokens = {}
#     seqid_to_indices = {}
#     for (sids, _), (toks, parents) in zip(sampling_metadata.seq_groups,
#                                           sample_results):
#         for idx, (token, parent) in enumerate(zip(toks, parents)):
#             seqid_to_tokens.setdefault(sids[parent], []).append(token)
#             seqid_to_indices.setdefault(sids[parent], []).append(idx)

#     seqids = args.miro_seqids

#     picked_tokens = torch.tensor([seqid_to_tokens[x] for x in seqids],
#                                  device=logits.device,
#                                  dtype=torch.long)

#     # Clumsily, we recalculate token surprisals.
#     logits_view = logits[args.miro_indices]
#     picked_surprise = torch.gather(torch.log_softmax(logits_view, dim=-1),
#                                    dim=-1,
#                                    index=picked_tokens) / -math.log(2)

#     taus = args.miro_taus.unsqueeze(dim=-1)  # AKA target surprisals
#     etas = args.miro_etas.unsqueeze(dim=-1)  # AKA accumulation rates
#     mus = args.miro_mus.unsqueeze(dim=-1)  # AKA surprisal accumulators
#     nu_mus = mus - (picked_surprise - taus) * etas

#     # Record updated mu values for use in the next iteration
#     # Note how each mu is split into multiple based on the number of samples.
#     for seqid, seq_mus in zip(seqids, nu_mus):
#         for sample_idx, mu in zip(seqid_to_indices[seqid], seq_mus):
#             output_metadata.add(seqid, sample_idx, "miro_mu", mu)
