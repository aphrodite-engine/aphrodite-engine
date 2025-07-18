from dataclasses import dataclass
from typing import Optional, List

import torch

from aphrodite.common.sampling_params import SamplerID


@dataclass
class SamplingMetadata:

    # Temperature
    temperature: Optional[torch.Tensor]
    dynatemp_min: Optional[torch.Tensor]
    dynatemp_max: Optional[torch.Tensor]
    dynatemp_exp: Optional[torch.Tensor]

    all_greedy: bool
    all_random: bool

    # Alphabet sampling
    top_p: Optional[torch.Tensor]
    top_k: Optional[torch.Tensor]
    min_p: Optional[torch.Tensor]
    top_a: Optional[torch.Tensor]

    # DRY
    dry_multiplier: Optional[torch.Tensor]
    dry_base: Optional[torch.Tensor]
    dry_allowed_length: Optional[torch.Tensor]
    dry_sequence_breaker_ids: Optional[torch.Tensor]
    dry_ranges: Optional[torch.Tensor]
    dry_max_ngram: Optional[torch.Tensor]
    dry_max_occurrences: Optional[torch.Tensor]
    dry_early_exit_match_len: Optional[torch.Tensor]

    # No repeat ngram
    no_repeat_ngram_size: Optional[torch.Tensor]

    # Tail-Free Sampling
    tfs: Optional[torch.Tensor]

    # Eta Cutoff
    eta_cutoff: Optional[torch.Tensor]

    # Epsilon Cutoff
    epsilon_cutoff: Optional[torch.Tensor]

    # Typical Sampling
    typical_p: Optional[torch.Tensor]

    # Quadratic Sampling
    quadratic_smoothing_factor: Optional[torch.Tensor]
    quadratic_smoothing_curve: Optional[torch.Tensor]

    # XTC Sampling
    xtc_threshold: Optional[torch.Tensor]
    xtc_probability: Optional[torch.Tensor]

    # Top-nsigma Sampling
    top_nsigma: Optional[torch.Tensor]

    # Skew
    skew: Optional[torch.Tensor]

    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: Optional[int]

    no_penalties: bool
    prompt_token_ids: Optional[torch.Tensor]
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: list[list[int]]

    # req_index -> (min_tokens, stop_token_ids)
    min_tokens: dict[int, tuple[int, set[int]]]

    logit_bias: list[Optional[dict[int, float]]]

    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    allowed_token_ids_mask: Optional[torch.Tensor]

    # req_index -> bad_words_token_ids
    bad_words_token_ids: dict[int, list[list[int]]]

    # Sampler priority and temperature_last for priority-based execution
    sampler_priority: Optional[List[SamplerID]] = None
    temperature_last: bool = False
