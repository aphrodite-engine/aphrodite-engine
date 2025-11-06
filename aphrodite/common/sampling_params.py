"""Sampling parameters for text generation."""

import copy
import warnings
from dataclasses import field
from enum import Enum, IntEnum
from functools import cached_property
from typing import Annotated, Any

import msgspec
from pydantic.dataclasses import dataclass

from aphrodite.config import SchedulerConfig
from aphrodite.logger import init_logger
from aphrodite.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5
_MAX_TEMP = 1e-2


class SamplingType(IntEnum):
    GREEDY = 0
    RANDOM = 1
    RANDOM_SEED = 2
    BEAM = 3


# maybe make msgspec?
@dataclass
class StructuredOutputsParams:
    # One of these fields will be used to build a logit processor.
    json: str | dict | None = None
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    # These are other options that can be set.
    disable_fallback: bool = False
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: str | None = None
    structural_tag: str | None = None

    _backend: str | None = field(default=None, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""
    _backend_was_auto: bool = field(default=False, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        count = sum(
            [
                self.json is not None,
                self.regex is not None,
                self.choice is not None,
                self.grammar is not None,
                self.json_object is not None,
                self.structural_tag is not None,
            ]
        )
        if count > 1:
            raise ValueError(
                "You can only use one kind of structured outputs constraint "
                f"but multiple are specified: {self.__dict__}"
            )

    def all_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, field) is None
            for field in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
                "structural_tag",
            )
        )

    def all_non_structural_tag_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, field) is None
            for field in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
            )
        )


@dataclass
class GuidedDecodingParams(StructuredOutputsParams):
    def __post_init__(self):
        warnings.warn(
            "GuidedDecodingParams is deprecated. This will be removed in "
            "v0.12.0 or v1.0.0, which ever is soonest. Please use "
            "StructuredOutputsParams instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__post_init__()


class RequestOutputKind(Enum):
    # Return entire output so far in every RequestOutput
    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    DELTA = 1
    # Do not return intermediate RequestOuputs
    FINAL_ONLY = 2


class SamplerID(IntEnum):
    # Mirror these in aphrodite/modeling/layers/sampler.py
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

    @classmethod
    def from_str(cls, value: str | int) -> "SamplerID":
        """Convert string or int to SamplerID enum.

        Args:
            value: String name (case-insensitive) or integer value

        Returns:
            SamplerID enum value

        Raises:
            ValueError: If value cannot be converted to SamplerID
        """
        if isinstance(value, int):
            return cls(value)

        try:
            return cls[value.upper()]
        except KeyError as e:
            valid_names = [x.name for x in cls]
            raise ValueError(f"Invalid sampler name '{value}'. Must be one of: {valid_names}") from e


class SamplingParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support multiple additional samplers which are not supported
    by OpenAI.
    """

    n: int = 1
    """Number of outputs to return for the given prompt request.
    NOTE:
        `AsyncLLM` streams outputs by default. When `n > 1`, all `n` outputs
        are generated and streamed cumulatively per request. To see all `n`
        outputs upon completion, use `output_kind=RequestOutputKind.FINAL_ONLY`
        in `SamplingParams`."""
    best_of: int | None = None
    """
    Number of output sequences that are generated from the prompt.
    From these `best_of` sequences, the top `n` sequences are returned.
    `best_of` must be greater than or equal to `n`. By default,
    `best_of` is set to `n`.
    """
    _real_n: int | None = None
    presence_penalty: float = 0.0
    """
    Float that penalizes new tokens based on whether they
    appear in the generated text so far. Values > 0 encourage the model
    to use new tokens, while values < 0 encourage the model to repeat
    tokens.
    """
    frequency_penalty: float = 0.0
    """
    Float that penalizes new tokens based on their
    frequency in the generated text so far. Values > 0 encourage the
    model to use new tokens, while values < 0 encourage the model to
    repeat tokens.
    """
    repetition_penalty: float = 1.0
    """
    Float that penalizes new tokens based on their
    frequency in the generated text so far.
    freq_pen is applied additively while
    rep_pen is applied multiplicatively.
    Must be in [1, inf). set to 1 to disable the effect.
    """
    no_repeat_ngram_size: int = 0
    """
    Size of the n-grams to prevent repeating.
    1 would mean no token can appear twice.
    2 would mean no pair of consecutive tokens can appear twice.
    """
    temperature: float = 1.0
    """
    Float that controls the randomness of the sampling. Lower
    values make the model more deterministic, while higher values make
    the model more random. Zero means greedy sampling.
    """
    dynatemp_min: float = 0.0
    """
    Minimum temperature for dynamic temperature sampling.
    """
    dynatemp_max: float = 0.0
    """
    Maximum temperature for dynamic temperature sampling.
    """
    dynatemp_exponent: float = 1.0
    """
    Exponent for dynamic temperature sampling.
    """
    temperature_last: bool = False
    """
    Whether to use temperature as the last sampler in the sampling
    pipeline. Ignored if sampler_priority is set.
    """
    top_p: float = 1.0
    """
    Float that controls the cumulative probability of the top tokens
    to consider. Must be in (0, 1]. set to 1 to consider all tokens.
    """
    top_k: int = 0
    """
    Integer that controls the number of top tokens to consider. set
    to 0 (or -1) to consider all tokens.
    """
    top_a: float = 0.0
    """
    Float that controls the cutoff for Top-A sampling.
    Exact cutoff is top_a*max_prob**2. Must be in [0,inf], 0 to disable.
    """
    min_p: float = 0.0
    """
    Float that controls the cutoff for min-p sampling.
    Exact cutoff is min_p*max_prob. Must be in [0,1], 0 to disable.
    """
    tfs: float = 1.0
    """
    Float that controls the cumulative approximate curvature of the
    distribution to retain for Tail Free Sampling.
    Must be in (0, 1]. set to 1 to disable
    """
    eta_cutoff: float = 0.0
    """
    Float that controls the cutoff threshold for Eta sampling
    (a form of entropy adaptive truncation sampling)
    threshold is computed as min(eta, sqrt(eta)*entropy(probs)).
    Specified in units of 1e-4. set to 0 to disable
    """
    epsilon_cutoff: float = 0.0
    """
    Float that controls the cutoff threshold for
    Epsilon sampling (simple probability threshold truncation).
    Specified in units of 1e-4. set to 0 to disable.
    """
    typical_p: float = 1.0
    """
    Float that controls the cumulative probability of tokens
    closest in surprise to the expected surprise to consider.
    Must be in (0, 1]. set to 1 to disable.
    """
    smoothing_factor: float = 0.0
    """Smoothing factor for Quadratic Sampling."""
    smoothing_curve: float = 1.0
    """Smoothing curve for Quadratic (Cubic) Sampling."""
    seed: int | None = None
    """Random seed to use for the generation."""
    use_beam_search: bool = False
    """Whether to use beam search instead of sampling."""
    length_penalty: float = 1.0
    """Float that penalizes sequences based on their length.
    Used in beam search.
    """
    early_stopping: bool | str = False
    """
    Controls the stopping condition for beam search. It
    accepts the following values: `True`, where the generation stops as
    soon as there are `best_of` complete candidates; `False`, where an
    heuristic is applied and the generation stops when is it very
    unlikely to find better candidates; `"never"`, where the beam search
    procedure only stops when there cannot be better candidates
    (canonical beam search algorithm).
    """
    stop: None | str | list[str] = None
    """
    list of strings that stop the generation when they are generated.
    The returned output will not contain the stop strings.
    """
    stop_token_ids: list[int] | None = None
    """
    list of tokens that stop the generation when they are
    generated. The returned output will contain the stop tokens unless
    the stop tokens are special tokens.
    """
    include_stop_str_in_output: bool = False
    """
    Whether to include the stop strings in
    output text. Defaults to False.
    """
    ignore_eos: bool = False
    """
    Whether to ignore the EOS token and continue generating
    tokens after the EOS token is generated.
    """
    max_tokens: int | None = 16
    """
    Maximum number of tokens to generate per output sequence.
    """
    min_tokens: int = 0
    """
    Minimum number of tokens to generate per output sequence
    before EOS or stop tokens are generated.
    """
    logprobs: int | None = None
    """
    Number of log probabilities to return per output token.
    When set to None, no probability is returned. If set to a non-None
    value, the result includes the log probabilities of the specified
    number of most likely tokens, as well as the chosen tokens.
    Note that the implementation follows the OpenAI API: The API will
    always return the log probability of the sampled token, so there
    may be up to `logprobs+1` elements in the response.
    When set to -1, return all `vocab_size` log probabilities.
    """
    prompt_logprobs: int | None = None
    """
    Number of log probabilities to return per prompt token.
    """
    detokenize: bool = True
    """
    Whether to detokenize the output. Defaults to True.
    """
    custom_token_bans: list[int] | None = None
    """
    list of token IDs to ban from generating
    """
    token_ban_ranges: list[tuple[list[int], int, int]] | None = None
    """
    list of tuples (tokens, start, length) to ban from
    generating. start=0 means start from first output token.
    """
    skip_special_tokens: bool = True
    """
    Whether to skip special tokens in the output.
    defaults to true.
    """
    spaces_between_special_tokens: bool = True
    """
    Whether to add spaces between special
    tokens in the output. Defaults to True.
    """
    # Optional[list[LogitsProcessor]] type.
    # We use Any here because the type above
    # is not supported by msgspec.
    logits_processors: Any | None = None
    """
    list of functions that modify logits based on
    previously generated tokens, and optionally prompt tokens as
    a first argument.
    """
    truncate_prompt_tokens: Annotated[int, msgspec.Meta(ge=-1)] | None = None
    """
    If set to an integer k, will use only the last
    k tokens from the prompt (i.e. left-truncation). Defaults to None
    (i.e. no truncation).
    """
    xtc_threshold: float = 0.1
    """
    In XTC sampling, if 2 or more tokens have probability
    above this threshold, consider removing all but the last one.
    """
    xtc_probability: float = 0
    """
    Probability that the removal will actually happen.
    0 disables the sampler, 1 makes it always happen.
    """
    nsigma: float = 0.0
    """
    Number of standard deviations from the maximum logit to use
    as a cutoff threshold. Tokens with logits below
    (max_logit - nsgima * std_dev) are filtered out. Higher values
    (e.g. 3.0) keep more tokens, lower values (e.g. 1.0) are more
    selective. Must be positive. 0 to disable.
    """
    mirostat_mode: int = 0
    """
    Can either be 0 (disabled) or 2 (Mirostat v2).
    """
    mirostat_tau: float = 0.0
    """
    Target "surprisal" that mirostat works towards.
    Range [0, inf).
    """
    mirostat_eta: float = 0.0
    """
    Rate at which mirostat updates its internal surprisal value.
    Range [0, inf).
    """
    dry_multiplier: float = 0.0
    """
    Float that controls the magnitude of the DRY sampling
    penalty. Higher values create stronger penalties against
    repetition. The penalty is multiplied by this value before being
    applied. Must be non-negative. 0 disables the sampler.
    """
    dry_base: float = 1.75
    """
    Base for the exponential growth of the DRY sampling penalty.
    Controls how quickly the penalty increases with longer repeated
    sequences. Must be greater than 1. Higher values (e.g. 2.0) create
    more aggressive penalties for longer repetitions. Defaults to 1.75.
    """
    dry_allowed_length: int = 2
    """
    Maximum number of tokens that can be repeated
    without incurring a DRY sampling penalty. Sequences longer than
    this will be penalized exponentially. Must be at least 1.
    Defaults to 2.
    """
    dry_sequence_breaker_ids: list[int] = []
    """
    list of token IDs that stop
    the matching of repeated content. These tokens will break up the
    input into sections where repetition is evaluated separately.
    Common examples are newlines, quotes, and other structural tokens.
    Defaults to None.
    """
    dry_range: int = 0
    """
    The range of tokens (input + output) to apply the DRY
    sampler.
    """
    dry_max_ngram: int = 12
    """
    Maximum length of match to check in DRY sampling.
    """
    dry_max_occurrences: int = 8
    """
    How many occurrences of last_token we analyze in DRY sampling.
    """
    dry_early_exit_match_len: int = 8
    """
    If we find this large a match in DRY sampling, we stop
    searching.
    """
    skew: float = 0.0
    """
    Bias the token selection towards higher or lower probability
    tokens. Defaults to 0 (disabled).
    """
    sampler_priority: list[int] | None = None
    """
    A list of integers to control the order in which
    samplers are applied.
    """
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE
    """
    The type of output to generate. 0 is cumulative, which means
    return the entire output so far in every RequestOutput, 1
    is delta, which means return only deltas in each RequestOutput,
    and 2 is final_only, which means do not return intermediate
    request outputs.
    """
    # The below fields are not supposed to be used as an input.
    # They are set in post_init.
    output_text_buffer_length: int = 0
    _all_stop_token_ids: set[int] = msgspec.field(default_factory=set)
    _bad_words_token_ids: list[list[int]] | None = None

    # Fields used to construct logits processors
    structured_outputs: StructuredOutputsParams | None = None
    """Parameters for configuring structured outputs."""
    guided_decoding: GuidedDecodingParams | None = None
    """
    If provided, the engine will construct a guided
    decoding logits processor from these parameters. Defaults to None.
    """
    logit_bias: dict[int, float] | None = None
    """
    If provided, the engine will construct a logits processor
    that applies these logit biases. Defaults to None.
    """
    allowed_token_ids: list[int] | None = None
    """
    If provided, the engine will construct a logits
    processor which only retains scores for the given token ids.
    Defaults to None.
    """
    extra_args: dict[str, Any] | None = None
    """
    Extra arguments to pass to the engine. Defaults to None.
    """
    bad_words: list[str] | None = None
    """
    list of words that are not allowed to be generated.
    More precisely, only the last token of a corresponding
    token sequence is not allowed when the next generated token
    can complete the sequence.
    """
    banned_phrases_token_ids: list[list[int]] | None = None
    """
    list of token sequences that are not allowed to be generated.
    """
    enable_deepconf: bool | None = False
    """
    Enable DeepConf (Deep Think with Confidence) for confidence-based early
    stopping. When enabled, the model will automatically stop generation when
    its confidence drops below a certain threshold, as measured by a sliding
    window of token-level confidence scores. This helps prevent low-quality
    continuations and improves reasoning efficiency.
    The confidence is calculated as the negative average of the logprobs of
    non-sampled candidate tokens, where higher confidence indicates the model
    was more certain about its token choice.
    """
    deepconf_window_size: int | None = 2048
    """
    Size of the sliding window for confidence calculation in DeepConf.
    This parameter controls how many recent tokens are considered when
    computing the moving average confidence score. A larger window provides
    more stable confidence estimates but may be less responsive to recent
    confidence drops.
    The window uses a sliding average approach where each token's confidence is
    calculated as the negative average of the logprobs of all non-sampled
    candidate tokens. The system maintains a running sum and deque to
    efficiently compute the moving average without storing all historical
    values. Default: 2048 tokens.
    """
    deepconf_threshold: float | None = 17
    """
    Confidence threshold for early stopping in DeepConf.
    When the moving average confidence over the sliding window drops below this
    threshold, generation will stop early. This threshold should be calibrated
    based on the specific model and task.
    The threshold is compared against the average confidence of the last
    `deepconf_window_size` tokens. Lower thresholds are more conservative and
    allow generation to continue longer, while higher thresholds trigger
    earlier stopping for potentially better efficiency.
    The optimal threshold varies by model and dataset. The DeepConf paper found
    that thresholds around 17 worked well across multiple models and
    mathematical reasoning tasks, achieving up to 84.7% token reduction
    while maintaining or improving accuracy. Default: 17.
    """

    @staticmethod
    def from_optional(
        n: int | None = None,
        best_of: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        temperature: float | None = None,
        dynatemp_min: float | None = None,
        dynatemp_max: float | None = None,
        dynatemp_exponent: float | None = None,
        temperature_last: bool | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        top_a: float | None = None,
        min_p: float | None = None,
        tfs: float | None = None,
        eta_cutoff: float | None = None,
        epsilon_cutoff: float | None = None,
        typical_p: float | None = None,
        smoothing_factor: float | None = None,
        smoothing_curve: float | None = None,
        seed: int | None = None,
        use_beam_search: bool | None = None,
        length_penalty: float | None = None,
        early_stopping: bool | str | None = None,
        stop: None | str | list[str] = None,
        stop_token_ids: list[int] | None = None,
        include_stop_str_in_output: bool | None = None,
        ignore_eos: bool | None = None,
        max_tokens: int | None = None,
        min_tokens: int | None = None,
        logprobs: int | None = None,
        prompt_logprobs: int | None = None,
        detokenize: bool | None = None,
        custom_token_bans: list[int] | None = None,
        token_ban_ranges: list[tuple[list[int], int, int]] | None = None,
        skip_special_tokens: bool | None = None,
        spaces_between_special_tokens: bool | None = None,
        logits_processors: Any | None = None,
        truncate_prompt_tokens: int | None = None,
        xtc_threshold: float | None = None,
        xtc_probability: float | None = None,
        nsigma: float | None = None,
        mirostat_mode: int | None = None,
        mirostat_tau: float | None = None,
        mirostat_eta: float | None = None,
        dry_multiplier: float | None = None,
        dry_base: float | None = None,
        dry_allowed_length: int | None = None,
        dry_sequence_breaker_ids: list[int] | None = None,
        dry_range: int | None = None,
        dry_max_ngram: int | None = None,
        dry_max_occurrences: int | None = None,
        dry_early_exit_match_len: int | None = None,
        skew: float | None = None,
        sampler_priority: list[int] | None = None,
        output_kind: RequestOutputKind | None = None,
        structured_outputs: StructuredOutputsParams | None = None,
        guided_decoding: GuidedDecodingParams | None = None,
        logit_bias: dict[int, float] | None = None,
        allowed_token_ids: list[int] | None = None,
        extra_args: dict[str, Any] | None = None,
        bad_words: list[str] | None = None,
        banned_phrases_token_ids: list[list[int]] | None = None,
        enable_deepconf: bool | None = False,
        deepconf_window_size: int | None = 2048,
        deepconf_threshold: float | None = 17,
    ) -> "SamplingParams":
        if logit_bias is not None:
            # Convert token_id to integer
            # Clamp the bias between -100 and 100 per OpenAI API spec
            logit_bias = {int(token): min(100.0, max(-100.0, bias)) for token, bias in logit_bias.items()}
        if guided_decoding is not None:
            warnings.warn(
                "guided_decoding is deprecated. This will be removed in "
                "v0.12.0 or v1.0.0, which ever is soonest. Please use "
                "structured_outputs instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            structured_outputs = guided_decoding
            guided_decoding = None

        return SamplingParams(
            n=1 if n is None else n,
            best_of=best_of,
            presence_penalty=0.0 if presence_penalty is None else presence_penalty,
            frequency_penalty=0.0 if frequency_penalty is None else frequency_penalty,
            repetition_penalty=1.0 if repetition_penalty is None else repetition_penalty,
            no_repeat_ngram_size=0 if no_repeat_ngram_size is None else no_repeat_ngram_size,
            temperature=1.0 if temperature is None else temperature,
            dynatemp_min=0.0 if dynatemp_min is None else dynatemp_min,
            dynatemp_max=0.0 if dynatemp_max is None else dynatemp_max,
            dynatemp_exponent=1.0 if dynatemp_exponent is None else dynatemp_exponent,
            temperature_last=False if temperature_last is None else temperature_last,
            top_p=1.0 if top_p is None else top_p,
            top_k=-1 if top_k is None else top_k,
            top_a=0.0 if top_a is None else top_a,
            min_p=0.0 if min_p is None else min_p,
            tfs=1.0 if tfs is None else tfs,
            eta_cutoff=0.0 if eta_cutoff is None else eta_cutoff,
            epsilon_cutoff=0.0 if epsilon_cutoff is None else epsilon_cutoff,
            typical_p=1.0 if typical_p is None else typical_p,
            smoothing_factor=0.0 if smoothing_factor is None else smoothing_factor,
            smoothing_curve=1.0 if smoothing_curve is None else smoothing_curve,
            seed=seed,
            use_beam_search=(False if use_beam_search is None else use_beam_search),
            length_penalty=1.0 if length_penalty is None else length_penalty,
            early_stopping=False if early_stopping is None else early_stopping,
            stop=stop,
            stop_token_ids=stop_token_ids if stop_token_ids is not None else [],
            include_stop_str_in_output=(False if include_stop_str_in_output is None else include_stop_str_in_output),
            ignore_eos=False if ignore_eos is None else ignore_eos,
            max_tokens=16 if max_tokens is None else max_tokens,
            min_tokens=0 if min_tokens is None else min_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            detokenize=True if detokenize is None else detokenize,
            custom_token_bans=custom_token_bans,
            token_ban_ranges=token_ban_ranges,
            skip_special_tokens=(True if skip_special_tokens is None else skip_special_tokens),
            spaces_between_special_tokens=(
                True if spaces_between_special_tokens is None else spaces_between_special_tokens
            ),
            logits_processors=logits_processors,
            truncate_prompt_tokens=truncate_prompt_tokens,
            xtc_threshold=0.1 if xtc_threshold is None else xtc_threshold,
            xtc_probability=0 if xtc_probability is None else xtc_probability,
            nsigma=0.0 if nsigma is None else nsigma,
            mirostat_mode=0 if mirostat_mode is None else mirostat_mode,
            mirostat_tau=0.0 if mirostat_tau is None else mirostat_tau,
            mirostat_eta=0.0 if mirostat_eta is None else mirostat_eta,
            dry_multiplier=0.0 if dry_multiplier is None else dry_multiplier,
            dry_base=1.75 if dry_base is None else dry_base,
            dry_allowed_length=2 if dry_allowed_length is None else dry_allowed_length,
            dry_sequence_breaker_ids=([] if dry_sequence_breaker_ids is None else dry_sequence_breaker_ids),
            dry_range=0 if dry_range is None else dry_range,
            dry_max_ngram=12 if dry_max_ngram is None else dry_max_ngram,
            dry_max_occurrences=8 if dry_max_occurrences is None else dry_max_occurrences,
            dry_early_exit_match_len=8 if dry_early_exit_match_len is None else dry_early_exit_match_len,
            skew=0.0 if skew is None else skew,
            sampler_priority=[] if sampler_priority is None else sampler_priority,
            output_kind=(RequestOutputKind.CUMULATIVE if output_kind is None else output_kind),
            structured_outputs=structured_outputs,
            logit_bias=logit_bias,
            allowed_token_ids=allowed_token_ids,
            extra_args=extra_args,
            bad_words=bad_words,
            banned_phrases_token_ids=banned_phrases_token_ids,
            enable_deepconf=enable_deepconf,
            deepconf_window_size=deepconf_window_size,
            deepconf_threshold=deepconf_threshold,
        )

    default_values = {
        "n": 1,
        "best_of": None,
        "_real_n": None,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "temperature": 1.0,
        "dynatemp_min": 0.0,
        "dynatemp_max": 0.0,
        "dynatemp_exponent": 1.0,
        "temperature_last": False,
        "top_p": 1.0,
        "top_k": 0,
        "top_a": 0.0,
        "min_p": 0.0,
        "tfs": 1.0,
        "eta_cutoff": 0.0,
        "epsilon_cutoff": 0.0,
        "typical_p": 1.0,
        "smoothing_factor": 0.0,
        "smoothing_curve": 1.0,
        "seed": None,
        "use_beam_search": False,
        "length_penalty": 1.0,
        "early_stopping": False,
        "stop": [],
        "stop_token_ids": [],
        "ignore_eos": False,
        "max_tokens": 16,
        "min_tokens": 0,
        "logprobs": None,
        "prompt_logprobs": None,
        "detokenize": True,
        "custom_token_bans": None,
        "token_ban_ranges": None,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
        "include_stop_str_in_output": False,
        "truncate_prompt_tokens": None,
        "xtc_threshold": 0.1,
        "xtc_probability": 0,
        "nsigma": 0.0,
        "mirostat_mode": 0,
        "mirostat_tau": 0.0,
        "mirostat_eta": 0.0,
        "dry_multiplier": 0.0,
        "dry_base": 1.75,
        "dry_allowed_length": 2,
        "dry_sequence_breaker_ids": [],
        "dry_range": 0,
        "dry_max_ngram": 12,
        "dry_max_occurrences": 8,
        "dry_early_exit_match_len": 8,
        "skew": 0.0,
        "sampler_priority": [],
        "output_kind": RequestOutputKind.CUMULATIVE,
        "guided_decoding": None,
        "logit_bias": None,
        "allowed_token_ids": None,
        "bad_words": None,
        "extra_args": None,
        "banned_phrases_token_ids": None,
        "enable_deepconf": False,
        "deepconf_window_size": 2048,
        "deepconf_threshold": 17,
    }

    def __post_init__(self) -> None:
        # how we deal with `best_of``:
        # if `best_of`` is not set, we default to `n`;
        # if `best_of`` is set, we set `n`` to `best_of`,
        # and set `_real_n`` to the original `n`.
        # when we return the result, we will check
        # if we need to return `n` or `_real_n` results
        if self.best_of:
            if self.best_of < self.n:
                raise ValueError(
                    f"best_of must be greater than or equal to n, got n={self.n} and best_of={self.best_of}."
                )
            self._real_n = self.n
            self.n = self.best_of
        if 0 < self.temperature < _MAX_TEMP:
            logger.warning(
                "temperature %s is less than %s, "
                "which may cause numerical errors NaN or inf in tensors. We "
                "have maxed it out to %s.",
                self.temperature,
                _MAX_TEMP,
                _MAX_TEMP,
            )
            self.temperature = max(self.temperature, _MAX_TEMP)
        if self.seed == -1:
            self.seed = None
        else:
            self.seed = self.seed
        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]
        else:
            self.stop = list(self.stop)
        if self.stop_token_ids is None:
            self.stop_token_ids = []
        else:
            self.stop_token_ids = list(self.stop_token_ids)
        self.logprobs = 1 if self.logprobs is True else self.logprobs
        self.prompt_logprobs = 1 if self.prompt_logprobs is True else self.prompt_logprobs

        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        if self.stop and not self.include_stop_str_in_output:
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1

        if self.logit_bias is not None:
            logit_bias = {int(token): bias for token, bias in self.logit_bias.items()}
            self.logit_bias = logit_bias

        self._verify_args()
        if self.use_beam_search:
            self._verify_beam_search()
        else:
            self._verify_non_beam_search()
            if self.temperature < _SAMPLING_EPS:
                # Zero temperature means greedy sampling.
                self.top_p = 1.0
                self.top_k = -1
                self.min_p = 0.0
                self.top_a = 0.0
                self._verify_greedy_sampling()
        # eos_token_id is added to this by the engine
        self._all_stop_token_ids = set(self.stop_token_ids)

        if self.guided_decoding is not None:
            warnings.warn(
                "guided_decoding is deprecated. This will be removed in "
                "v0.12.0 or v1.0.0, which ever is soonest. Please use "
                "structured_outputs instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.structured_outputs = self.guided_decoding
            self.guided_decoding = None

    def _verify_args(self) -> None:
        if not isinstance(self.n, int):
            raise ValueError(f"n must be an int, but is of type {type(self.n)}")
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if self.best_of is not None:
            if not isinstance(self.best_of, int):
                raise ValueError(f"best_of must be an integer, got {type(self.best_of)}")
            if self.best_of < 1:
                raise ValueError(f"best_of must be at least 1, got {self.best_of}")
            if self.best_of < self.n:
                raise ValueError(
                    f"best_of must be greater than or equal to n, got n={self.n} and best_of={self.best_of}."
                )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(f"presence_penalty must be in [-2, 2], got {self.presence_penalty}.")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}.")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be in [1, inf), got {self.repetition_penalty}.")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        # quietly accept -1 as disabled, but prefer 0
        if self.top_k < -1:
            raise ValueError(f"top_k must be 0 (disable), or at least 1, got {self.top_k}.")
        if self.top_a < 0:
            raise ValueError(f"top_a must be non negative, got {self.top_a}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if not 0.0 < self.tfs <= 1.0:
            raise ValueError(f"tfs must be in (0, 1], got {self.tfs}.")
        if self.epsilon_cutoff < 0.0 or self.epsilon_cutoff > 1000.0:
            raise ValueError(f"epsilon_cutoff must be in [0, 1000], got {self.epsilon_cutoff}.")
        # pylint: disable=unneeded-not
        if not self.eta_cutoff >= 0:
            raise ValueError(f"eta_cutoff must be non negative, got {self.eta_cutoff}.")
        if not 0.0 <= self.typical_p <= 1.0:
            raise ValueError(f"typical_p must be in (0, 1], got {self.typical_p}.")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {self.max_tokens}.")
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be greater than or equal to 0, got {self.min_tokens}.")
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens must be less than or equal to max_tokens={self.max_tokens}, got {self.min_tokens}."
            )
        if self.logprobs is not None and self.logprobs != -1 and self.logprobs < 0:
            raise ValueError(f"logprobs must be non-negative or -1, got {self.logprobs}.")
        if self.prompt_logprobs is not None and self.prompt_logprobs != -1 and self.prompt_logprobs < 0:
            raise ValueError(f"prompt_logprobs must be non-negative or -1, got {self.prompt_logprobs}.")
        if self.truncate_prompt_tokens is not None and (
            self.truncate_prompt_tokens == 0 or self.truncate_prompt_tokens < -1
        ):
            raise ValueError(f"truncate_prompt_tokens must be an integer >= 1 or -1, got {self.truncate_prompt_tokens}")
        assert isinstance(self.stop, list)
        if any(not stop_str for stop_str in self.stop):
            raise ValueError("stop cannot contain an empty string.")
        if self.stop and not self.detokenize:
            raise ValueError(
                "stop strings are only supported when detokenize is True. set detokenize=True to use stop."
            )
        if self.xtc_threshold < 0.0:
            raise ValueError(f"xtc_threshold must be non-negative, got {self.xtc_threshold}.")
        if not 0.0 <= self.xtc_probability <= 1.0:
            raise ValueError(f"xtc_probability must be in [0, 1], got {self.xtc_probability}.")
        if self.nsigma < 0.0:
            raise ValueError(f"nsigma must be non-negative, got {self.nsigma}.")
        if self.mirostat_mode not in [0, 2]:
            raise ValueError(f"mirostat_mode must be 0 or 2, got {self.mirostat_mode}.")
        if self.mirostat_tau < 0.0:
            raise ValueError(f"mirostat_tau must be non-negative, got {self.mirostat_tau}.")
        if self.mirostat_eta < 0.0:
            raise ValueError(f"mirostat_eta must be non-negative, got {self.mirostat_eta}.")
        if self.dry_multiplier < 0.0:
            raise ValueError(f"dry_multiplier must be non-negative, got {self.dry_multiplier}.")
        if self.dry_base <= 1.0:
            raise ValueError(f"dry_base must be greater than 1, got {self.dry_base}.")
        if self.dry_allowed_length < 0:
            raise ValueError(f"dry_allowed_length must be non-negative, got {self.dry_allowed_length}.")
        if self.dry_range < 0:
            raise ValueError(f"dry_range must be non-negative, got {self.dry_range}.")
        if self.dry_max_ngram < 0:
            raise ValueError(f"dry_max_ngram must be non-negative, got {self.dry_max_ngram}.")
        if self.dry_max_occurrences < 0:
            raise ValueError(f"dry_max_occurrences must be non-negative, got {self.dry_max_occurrences}.")
        if self.dry_early_exit_match_len < 0:
            raise ValueError(f"dry_early_exit_match_len must be non-negative, got {self.dry_early_exit_match_len}.")
        if self.skew < 0.0:
            raise ValueError(f"skew must be non-negative, got {self.skew}.")
        if self.custom_token_bans is not None and not isinstance(self.custom_token_bans, list):
            raise ValueError("custom_token_bans must be a list of integers")
        if self.token_ban_ranges is not None and not isinstance(self.token_ban_ranges, list):
            raise ValueError("token_ban_ranges must be a list of tuples")
        if self.bad_words is None:
            self.bad_words = []
        else:
            self.bad_words = list(self.bad_words)

        if self.banned_phrases_token_ids is not None and not isinstance(self.banned_phrases_token_ids, list):
            raise ValueError("banned_phrases_token_ids must be a list of lists of integers")

        if self.enable_deepconf:
            if self.deepconf_window_size < 1:
                raise ValueError(f"deepconf_window_size must be at least 1, got {self.deepconf_window_size}.")
            if self.deepconf_threshold < 0:
                raise ValueError(f"deepconf_threshold must be non-negative, got {self.deepconf_threshold}.")

        if self.sampler_priority is not None:
            if not self.sampler_priority:
                self.sampler_priority = None
                return

            if not isinstance(self.sampler_priority, list):
                raise ValueError("sampler_priority must be a list of integers or strings")

            try:
                self.sampler_priority = [SamplerID.from_str(x) for x in self.sampler_priority]
                provided_samplers = set(self.sampler_priority)
            except ValueError as e:
                raise ValueError(f"Invalid sampler ID in priority list: {e}") from e

            required_samplers = set(SamplerID)
            if not required_samplers.issubset(provided_samplers):
                missing = required_samplers - provided_samplers
                missing_names = [s.name for s in missing]
                raise ValueError(f"Missing required samplers in priority list: {missing_names}")

        if self.best_of != self._real_n and self.output_kind == (RequestOutputKind.DELTA):
            raise ValueError("best_of must equal n to use output_kind=DELTA")

    def _verify_beam_search(self) -> None:
        if self.best_of == 1:
            raise ValueError(f"best_of must be greater than 1 when using beam search. Got {self.best_of}.")
        if self.temperature > _SAMPLING_EPS:
            raise ValueError("temperature must be 0 when using beam search.")
        if self.top_p < 1.0 - _SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using beam search.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using beam search.")
        if self.early_stopping not in [True, False, "never"]:
            raise ValueError(f"early_stopping must be True, False, or 'never', got {self.early_stopping}.")

    def _verify_non_beam_search(self) -> None:
        if self.early_stopping is not False:
            raise ValueError("early_stopping is not effective and must be False when not using beam search.")
        if self.length_penalty < 1.0 - _SAMPLING_EPS or self.length_penalty > 1.0 + _SAMPLING_EPS:
            raise ValueError(
                "length_penalty is not effective and must be the default value of 1.0 when not using beam search."
            )

    def _verify_greedy_sampling(self) -> None:
        if self.top_p < 1.0 - _SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using greedy sampling.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using greedy sampling.")

    def _verify_with_scheduler_config(self, scheduler_config: "SchedulerConfig") -> None:
        if scheduler_config.single_user_mode:
            if self.n > 1:
                raise ValueError("n must be 1 in single user mode.")
            if self.use_beam_search:
                raise ValueError("beam search is not supported in single user mode.")

    def update_from_generation_config(
        self, generation_config: dict[str, Any], model_eos_token_id: int | None = None
    ) -> None:
        """Update if there are non-default values from generation_config"""

        if model_eos_token_id is not None:
            # Add the eos token id into the sampling_params to support
            # min_tokens processing.
            self._all_stop_token_ids.add(model_eos_token_id)

        # Update eos_token_id for generation
        if (eos_ids := generation_config.get("eos_token_id")) is not None:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            if model_eos_token_id is not None:
                # We don't need to include the primary eos_token_id in
                # stop_token_ids since it's handled separately for stopping
                # purposes.
                eos_ids.discard(model_eos_token_id)
            if eos_ids:
                self._all_stop_token_ids.update(eos_ids)
                if not self.ignore_eos:
                    assert isinstance(self.stop_token_ids, list)
                    eos_ids.update(self.stop_token_ids)
                    self.stop_token_ids = list(eos_ids)

    def update_from_tokenizer(self, tokenizer: AnyTokenizer) -> None:
        if not self.bad_words:
            return
        self._bad_words_token_ids = []
        for bad_word in self.bad_words:
            # To prohibit words both at the beginning
            # and in the middle of text
            # (related to add_prefix_space tokenizer parameter)
            for add_prefix_space in [False, True]:
                prefix = " " if add_prefix_space else ""
                prompt = prefix + bad_word.lstrip()
                prompt_token_ids = tokenizer.encode(text=prompt, add_special_tokens=False)

                # If no space at the beginning
                # or if prefix space produces a new word token
                if (not add_prefix_space) or (
                    add_prefix_space
                    and prompt_token_ids[0] != self._bad_words_token_ids[-1][0]
                    and len(prompt_token_ids) == len(self._bad_words_token_ids[-1])
                ):
                    self._bad_words_token_ids.append(prompt_token_ids)

        invalid_token_ids = [
            token_id
            for bad_words_token_ids in self._bad_words_token_ids
            for token_id in bad_words_token_ids
            if token_id < 0 or token_id > tokenizer.max_token_id
        ]
        if len(invalid_token_ids) > 0:
            raise ValueError(
                f"The model vocabulary size is {tokenizer.max_token_id + 1},"
                f" but the following tokens"
                f" were specified as bad: {invalid_token_ids}."
                f" All token id values should be integers satisfying:"
                f" 0 <= token_id <= {tokenizer.max_token_id}."
            )

    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.use_beam_search:
            return SamplingType.BEAM
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM

    @property
    def all_stop_token_ids(self) -> set[int]:
        return self._all_stop_token_ids

    @property
    def bad_words_token_ids(self) -> list[list[int]] | None:
        # For internal use only. Backward compatibility not guaranteed
        return self._bad_words_token_ids

    def clone(self) -> "SamplingParams":
        """Deep copy, but maybe not the LogitsProcessor objects.
        LogitsProcessor objects may contain an arbitrary, nontrivial amount of
        data that is expensive to copy. However, if not copied, the processor
        needs to support parallel decoding for multiple sequences
        """

        logit_processor_refs = (
            None
            if self.logits_processors is None
            else {id(lp): lp.clone() if hasattr(lp, "clone") else lp for lp in self.logits_processors}
        )
        return copy.deepcopy(self, memo=logit_processor_refs)

    def __repr__(self) -> str:
        repr_str = "SamplingParams("
        for param, default_value in self.default_values.items():
            current_value = getattr(self, param)
            if (
                param != "guided_decoding"
                or current_value is None
                or not all(
                    getattr(current_value, field) is None
                    for field in ["json", "regex", "choice", "grammar", "json_object", "backend", "whitespace_pattern"]
                )
            ) and current_value != default_value:
                repr_str += f"{param}={current_value}, "
        repr_str = repr_str.rstrip(", ") + ")"
        return repr_str


class BeamSearchParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Beam search parameters for text generation."""

    beam_width: int
    max_tokens: int
    ignore_eos: bool = False
    temperature: float = 0.0
    length_penalty: float = 1.0
    include_stop_str_in_output: bool = False
