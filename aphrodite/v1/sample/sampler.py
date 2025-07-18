"""A layer that samples the next tokens from the model's outputs."""

import torch
import torch.nn as nn
from loguru import logger

from aphrodite.common.logger import log_once
from aphrodite.common.sampling_params import SamplerID
from aphrodite.v1.outputs import LogprobsTensors, SamplerOutput
from aphrodite.v1.sample.metadata import SamplingMetadata
from aphrodite.v1.sample.ops import SamplingOps
from aphrodite.v1.sample.ops.temperatures import apply_all_temperatures
from aphrodite.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

_SAMPLING_EPS = 1e-5

# Default sampler execution order (same as V0)
DEFAULT_SAMPLER_ORDER = [
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


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()
        self.sampling_ops = SamplingOps()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        # NOTE: Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        # TODO: provide option for logprobs post sampling.
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            raw_logprobs = self.compute_logprobs(logits)

        # Use float32 for the logits.
        logits = logits.to(torch.float32)

        # Apply allowed token ids, bad words, and logits bias
        # (these are not part of the priority system)
        logits = self.sampling_ops.apply_allowed_token_ids(
            logits, sampling_metadata)
        logits = self.sampling_ops.apply_bad_words(logits, sampling_metadata)
        logits = self.sampling_ops.apply_logits_bias(logits, sampling_metadata)

        # Apply samplers in priority order
        logits = self._execute_samplers_in_order(logits, sampling_metadata)

        # Sample the next token.
        sampled = self.sample(logits, sampling_metadata)
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        sampled = sampled.long()

        # Gather the logprobs of the topk and sampled token (if requested).
        # Get logprobs and rank tensors (if requested)
        logprobs_tensors = None if num_logprobs is None else \
            self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    def apply_temperature(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return apply_all_temperatures(logits, sampling_metadata)

    def _execute_samplers_in_order(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Execute samplers in the specified priority order.

        Args:
            logits: Input logits tensor
            sampling_metadata: Sampling metadata containing priority information

        Returns:
            Modified logits tensor after applying samplers in priority order
        """
        # Determine the sampler execution order
        sampler_order = sampling_metadata.sampler_priority
        do_temp_last = sampling_metadata.temperature_last

        if sampler_order is None:
            # Use default order with temperature_last handling
            sampler_order = []
            for sampler_id in DEFAULT_SAMPLER_ORDER:
                if sampler_id == SamplerID.TEMPERATURE and do_temp_last:
                    continue
                sampler_order.append(sampler_id)
                
                if sampler_id == SamplerID.XTC and do_temp_last:
                    sampler_order.append(SamplerID.TEMPERATURE)
        else:
            # Warn if both custom order and temp_last are specified
            if do_temp_last:
                log_once(
                    "WARNING",
                    "Both sampler_priority and temperature_last=True "
                    "were specified. Using custom sampler_priority order "
                    "and ignoring temperature_last.")

        # Log the execution order for debugging
        logger.debug("Sampler execution order: ")
        for i, sampler_id in enumerate(sampler_order, 1):
            logger.debug(f"{i}. {sampler_id.name}")

        # Execute samplers in the specified order
        for sampler_id in sampler_order:
            if sampler_id == SamplerID.DRY and \
                sampling_metadata.dry_multiplier is not None:
                logger.debug("Applying DRY with dry_multiplier: "
                             f"{sampling_metadata.dry_multiplier}")
                logits = self.sampling_ops.apply_dry(logits, sampling_metadata)
                
            elif sampler_id == SamplerID.PENALTIES and \
                not sampling_metadata.no_penalties:
                logger.debug("Applying penalties")
                logits = self.sampling_ops.apply_penalties(
                    logits, sampling_metadata)
                
            elif sampler_id == SamplerID.NO_REPEAT_NGRAM and \
                sampling_metadata.no_repeat_ngram_size is not None:
                logger.debug(
                    f"Applying no_repeat_ngram with size: "
                    f"{sampling_metadata.no_repeat_ngram_size}")
                logits = self.sampling_ops.apply_no_repeat_ngram(
                    logits, sampling_metadata)
                
            elif sampler_id == SamplerID.TEMPERATURE and \
                sampling_metadata.temperature is not None:
                logger.debug(
                    f"Applying temperature: {sampling_metadata.temperature}")
                logits = self.apply_temperature(logits, sampling_metadata)
                
            elif sampler_id == SamplerID.TOP_NSIGMA and \
                sampling_metadata.top_nsigma is not None:
                logger.debug(
                    f"Applying Top-Nsigma with nsigma: "
                    f"{sampling_metadata.top_nsigma}")
                logits = self.sampling_ops.apply_top_nsigma(
                    logits, sampling_metadata)
                
            elif sampler_id == SamplerID.TOP_P_TOP_K:
                # Apply top-k and top-p filtering to logits
                if sampling_metadata.top_k is not None:
                    logger.debug(
                        f"Applying Top-k with top_k: "
                        f"{sampling_metadata.top_k}")
                    # Apply top-k filtering to logits
                    for i, top_k_val in enumerate(sampling_metadata.top_k):
                        if top_k_val < logits.size(-1):
                            top_k_values, _ = torch.topk(logits[i], \
                                int(top_k_val.item()), dim=-1)
                            top_k_threshold = top_k_values[-1] if \
                                top_k_values.numel() > 0 else -float('inf')
                            logits[i] = torch.where(
                                logits[i] >= top_k_threshold, logits[i],
                                torch.tensor(-float('inf'), device=logits.device,
                                             dtype=logits.dtype))
                
                if sampling_metadata.top_p is not None:
                    logger.debug(
                        f"Applying Top-p with top_p: "
                        f"{sampling_metadata.top_p}")
                    # Apply top-p filtering to logits
                    for i, top_p_val in enumerate(sampling_metadata.top_p):
                        if top_p_val < 1.0:
                            sorted_logits, sorted_indices = torch.sort(
                                logits[i], descending=True, dim=-1)
                            cumulative_probs = torch.softmax(
                                sorted_logits, dim=-1).cumsum(dim=-1)
                            sorted_indices_to_remove = \
                                cumulative_probs > top_p_val
                            sorted_indices_to_remove[1:] = \
                                sorted_indices_to_remove[:-1].clone()
                            sorted_indices_to_remove[0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(
                                0, sorted_indices, sorted_indices_to_remove)
                            logits[i][indices_to_remove] = -float('inf')
                
            elif sampler_id == SamplerID.TOP_A and \
                sampling_metadata.top_a is not None:
                logger.debug(f"Applying Top-a with top_a: "
                    f"{sampling_metadata.top_a}")
                logits = self.sampling_ops.apply_top_a(
                    logits, sampling_metadata)
                
            elif sampler_id == SamplerID.MIN_P and \
                sampling_metadata.min_p is not None:
                logger.debug(f"Applying Min-p with min_p: "
                    f"{sampling_metadata.min_p}")
                logits = self.sampling_ops.apply_min_p(
                    logits, sampling_metadata)
                
            elif sampler_id == SamplerID.TFS and \
                sampling_metadata.tfs is not None:
                logger.debug(f"Applying TFS with tfs: {sampling_metadata.tfs}")
                logits = self.sampling_ops.apply_tfs(logits, sampling_metadata)
                
            elif sampler_id == SamplerID.ETA_CUTOFF and \
                sampling_metadata.eta_cutoff is not None:
                logger.debug(f"Applying ETA Cutoff with eta_cutoff: "
                    f"{sampling_metadata.eta_cutoff}")
                logits = self.sampling_ops.apply_eta_cutoff(
                    logits, sampling_metadata)
                
            elif sampler_id == SamplerID.EPSILON_CUTOFF and \
                sampling_metadata.epsilon_cutoff is not None:
                logger.debug(f"Applying Epsilon Cutoff with epsilon_cutoff: "
                    f"{sampling_metadata.epsilon_cutoff}")
                logits = self.sampling_ops.apply_epsilon_cutoff(
                    logits, sampling_metadata)
                
            elif sampler_id == SamplerID.TYPICAL_P and \
                sampling_metadata.typical_p is not None:
                logger.debug(f"Applying Typical P with typical_p: "
                    f"{sampling_metadata.typical_p}")
                logits = self.sampling_ops.apply_typical_p(
                    logits, sampling_metadata)
                
            elif sampler_id == SamplerID.QUADRATIC and \
                sampling_metadata.quadratic_smoothing_factor is not None:
                logger.debug(f"Applying Quadratic with smoothing_factor: "
                    f"{sampling_metadata.quadratic_smoothing_factor}")
                logits = self.sampling_ops.apply_quadratic(
                    logits, sampling_metadata)
                
            elif sampler_id == SamplerID.XTC and \
                sampling_metadata.xtc_threshold is not None:
                logger.debug(f"Applying XTC with threshold: "
                    f"{sampling_metadata.xtc_threshold}")
                logits = self.sampling_ops.apply_xtc(logits, sampling_metadata)

        return logits

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled

        assert sampling_metadata.temperature is not None

        # Apply sampling (multinomial sampling from the processed logits)
        random_sampled = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            None,  # top_k already applied in priority system
            None,  # top_p already applied in priority system
        )

        # Apply skew (after softmax, same as before)
        if sampling_metadata.skew is not None:
            # Convert logits back to probabilities for skew
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            probs = self.sampling_ops.apply_skew(probs, sampling_metadata)
            # Convert back to logits
            logits = torch.log(probs)

        if greedy_sampled is None:
            return random_sampled

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled

    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    def gather_logprobs(
        self,
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs,
                                                 num_logprobs,
                                                 dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = (logprobs >= token_logprobs).sum(-1)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)
