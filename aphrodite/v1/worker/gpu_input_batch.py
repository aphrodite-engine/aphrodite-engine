# Datastructures defining an input batch

from dataclasses import dataclass, field
from typing import Any, Optional, cast

import numpy as np
import torch
from typing_extensions import deprecated

from aphrodite.common.pooling_params import PoolingParams
from aphrodite.common.sampling_params import (SamplerID, SamplingParams,
                                              SamplingType)
from aphrodite.lora.request import LoRARequest
from aphrodite.multimodal.inputs import (MultiModalKwargsItem,
                                         MultiModalKwargsItems,
                                         PlaceholderRange)
from aphrodite.utils import swap_dict_values
from aphrodite.v1.outputs import LogprobsTensors
from aphrodite.v1.pool.metadata import PoolingMetadata
from aphrodite.v1.sample.logits_processor import (BatchUpdateBuilder,
                                                  LogitsProcessors,
                                                  MoveDirectionality)
from aphrodite.v1.sample.metadata import SamplingMetadata
from aphrodite.v1.spec_decode.utils import is_spec_decode_unsupported
from aphrodite.v1.utils import copy_slice
from aphrodite.v1.worker.block_table import MultiGroupBlockTable

_SAMPLING_EPS = 1e-5


@dataclass
class CachedRequestState:

    req_id: str
    prompt_token_ids: list[int]
    mm_kwargs: list[MultiModalKwargsItem]
    mm_positions: list[PlaceholderRange]
    mm_hashes: list[str]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]
    generator: Optional[torch.Generator]

    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]

    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[int] = None

    lora_request: Optional[LoRARequest] = None

    _tokens_to_mask: list[int] = field(default_factory=list)

    # Persistent metadata for mirostat
    persistent_data: dict[str, Any] = None

    def __post_init__(self):
        self.num_prompt_tokens = len(self.prompt_token_ids)
        if self.persistent_data is None:
            self.persistent_data = {}

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    # Temporary back-compatibility for plugins that define model runner
    @property
    @deprecated("`mm_inputs` is superseded by `mm_kwargs` and will be "
                "removed in a later version. Please use `mm_kwargs` instead.")
    def mm_inputs(self) -> list[MultiModalKwargsItems]:
        return [
            MultiModalKwargsItems.from_seq([item]) for item in self.mm_kwargs
        ]

    def get_token_id(self, idx: int) -> int:
        if idx < self.num_prompt_tokens:
            return self.prompt_token_ids[idx]
        return self.output_token_ids[idx - self.num_prompt_tokens]


class InputBatch:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],  # The block_size of each kv cache group
        logitsprocs: Optional[LogitsProcessors] = None,
        is_spec_decode: bool = False,
        is_pooling_model: bool = False,
    ):
        self.is_pooling_model = is_pooling_model
        self.is_spec_decode = is_spec_decode
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size

        self._req_ids: list[Optional[str]] = []
        self.req_id_to_index: dict[str, int] = {}

        # TODO: This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        # This buffer is not directly transferred to the GPU, so it does not
        # need to be pinned.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs, ),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.num_computed_tokens_cpu = \
            self.num_computed_tokens_cpu_tensor.numpy()

        # Block table.
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,
        )

        # Sampling-related.
        self.temperature = torch.empty((max_num_reqs, ),
                                       dtype=torch.float32,
                                       device=device)
        self.temperature_cpu_tensor = torch.empty((max_num_reqs, ),
                                                  dtype=torch.float32,
                                                  device="cpu",
                                                  pin_memory=pin_memory)
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()

        # Dynatemp parameters
        self.dynatemp_min = torch.empty((max_num_reqs, ),
                                         dtype=torch.float32,
                                         device=device)
        self.dynatemp_min_cpu_tensor = torch.empty((max_num_reqs, ),
                                                    dtype=torch.float32,
                                                    device="cpu",
                                                    pin_memory=pin_memory)
        self.dynatemp_min_cpu = self.dynatemp_min_cpu_tensor.numpy()
        self.dynatemp_max = torch.empty((max_num_reqs, ),
                                         dtype=torch.float32,
                                         device=device)
        self.dynatemp_max_cpu_tensor = torch.empty((max_num_reqs, ),
                                                    dtype=torch.float32,
                                                    device="cpu",
                                                    pin_memory=pin_memory)
        self.dynatemp_max_cpu = self.dynatemp_max_cpu_tensor.numpy()
        self.dynatemp_exp = torch.empty((max_num_reqs, ),
                                         dtype=torch.float32,
                                         device=device)
        self.dynatemp_exp_cpu_tensor = torch.empty((max_num_reqs, ),
                                                    dtype=torch.float32,
                                                    device="cpu",
                                                    pin_memory=pin_memory)
        self.dynatemp_exp_cpu = self.dynatemp_exp_cpu_tensor.numpy()
        self.dynatemp_reqs: set[str] = set()

        self.top_p = torch.empty((max_num_reqs, ),
                                 dtype=torch.float32,
                                 device=device)
        self.top_p_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        self.top_p_reqs: set[str] = set()

        self.top_k = torch.empty((max_num_reqs, ),
                                 dtype=torch.int32,
                                 device=device)
        self.top_k_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        self.top_k_reqs: set[str] = set()

        # IDs of requests which do not support spec decoding
        self.spec_decode_unsupported_reqs: set[str] = set()

        # Top-a related data structures
        self.top_a = torch.empty((max_num_reqs, ),
                                 dtype=torch.float32,
                                 device=device)
        self.top_a_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.top_a_cpu = self.top_a_cpu_tensor.numpy()
        self.top_a_reqs: set[str] = set()

        # TFS related data structures
        self.tfs = torch.empty((max_num_reqs, ),
                               dtype=torch.float32,
                               device=device)
        self.tfs_cpu_tensor = torch.empty((max_num_reqs, ),
                                          dtype=torch.float32,
                                          device="cpu",
                                          pin_memory=pin_memory)
        self.tfs_cpu = self.tfs_cpu_tensor.numpy()
        self.tfs_reqs: set[str] = set()

        # Eta cutoff related data structures
        self.eta_cutoff = torch.empty((max_num_reqs, ),
                                      dtype=torch.float32,
                                      device=device)
        self.eta_cutoff_cpu_tensor = torch.empty((max_num_reqs, ),
                                                 dtype=torch.float32,
                                                 device="cpu",
                                                 pin_memory=pin_memory)
        self.eta_cutoff_cpu = self.eta_cutoff_cpu_tensor.numpy()
        self.eta_cutoff_reqs: set[str] = set()

        # Epsilon cutoff related data structures
        self.epsilon_cutoff = torch.empty((max_num_reqs, ),
                                          dtype=torch.float32,
                                          device=device)
        self.epsilon_cutoff_cpu_tensor = torch.empty((max_num_reqs, ),
                                                     dtype=torch.float32,
                                                     device="cpu",
                                                     pin_memory=pin_memory)
        self.epsilon_cutoff_cpu = self.epsilon_cutoff_cpu_tensor.numpy()
        self.epsilon_cutoff_reqs: set[str] = set()

        # Typical p related data structures
        self.typical_p = torch.empty((max_num_reqs, ),
                                     dtype=torch.float32,
                                     device=device)
        self.typical_p_cpu_tensor = torch.empty((max_num_reqs, ),
                                                dtype=torch.float32,
                                                device="cpu",
                                                pin_memory=pin_memory)
        self.typical_p_cpu = self.typical_p_cpu_tensor.numpy()
        self.typical_p_reqs: set[str] = set()

        # Quadratic related data structures
        self.quadratic_smoothing_factor = torch.empty((max_num_reqs, ),
                                                      dtype=torch.float32,
                                                      device=device)
        self.quadratic_smoothing_factor_cpu_tensor = torch.empty(
            (max_num_reqs, ),
            dtype=torch.float32,
            device="cpu",
            pin_memory=pin_memory)
        self.quadratic_smoothing_factor_cpu = (
            self.quadratic_smoothing_factor_cpu_tensor.numpy())
        self.quadratic_smoothing_curve = torch.empty((max_num_reqs, ),
                                                     dtype=torch.float32,
                                                     device=device)
        self.quadratic_smoothing_curve_cpu_tensor = torch.empty(
            (max_num_reqs, ),
            dtype=torch.float32,
            device="cpu",
            pin_memory=pin_memory)
        self.quadratic_smoothing_curve_cpu = (
            self.quadratic_smoothing_curve_cpu_tensor.numpy())
        self.quadratic_reqs: set[str] = set()

        # XTC related data structures
        self.xtc_threshold = torch.empty((max_num_reqs, ),
                                         dtype=torch.float32,
                                         device=device)
        self.xtc_threshold_cpu_tensor = torch.empty((max_num_reqs, ),
                                                    dtype=torch.float32,
                                                    device="cpu",
                                                    pin_memory=pin_memory)
        self.xtc_threshold_cpu = self.xtc_threshold_cpu_tensor.numpy()
        self.xtc_probability = torch.empty((max_num_reqs, ),
                                           dtype=torch.float32,
                                           device=device)
        self.xtc_probability_cpu_tensor = torch.empty((max_num_reqs, ),
                                                      dtype=torch.float32,
                                                      device="cpu",
                                                      pin_memory=pin_memory)
        self.xtc_probability_cpu = self.xtc_probability_cpu_tensor.numpy()
        self.xtc_reqs: set[str] = set()

        # Top-nsigma related data structures
        self.top_nsigma = torch.empty((max_num_reqs, ),
                                      dtype=torch.float32,
                                      device=device)
        self.top_nsigma_cpu_tensor = torch.empty((max_num_reqs, ),
                                                 dtype=torch.float32,
                                                 device="cpu",
                                                 pin_memory=pin_memory)
        self.top_nsigma_cpu = self.top_nsigma_cpu_tensor.numpy()
        self.top_nsigma_reqs: set[str] = set()

        # Mirostat related data structures
        self.mirostat_mode = torch.empty((max_num_reqs, ),
                                         dtype=torch.int32,
                                         device=device)
        self.mirostat_mode_cpu_tensor = torch.empty((max_num_reqs, ),
                                                    dtype=torch.int32,
                                                    device="cpu",
                                                    pin_memory=pin_memory)
        self.mirostat_mode_cpu = self.mirostat_mode_cpu_tensor.numpy()
        self.mirostat_tau = torch.empty((max_num_reqs, ),
                                        dtype=torch.float32,
                                        device=device)
        self.mirostat_tau_cpu_tensor = torch.empty((max_num_reqs, ),
                                                   dtype=torch.float32,
                                                   device="cpu",
                                                   pin_memory=pin_memory)
        self.mirostat_tau_cpu = self.mirostat_tau_cpu_tensor.numpy()
        self.mirostat_eta = torch.empty((max_num_reqs, ),
                                        dtype=torch.float32,
                                        device=device)
        self.mirostat_eta_cpu_tensor = torch.empty((max_num_reqs, ),
                                                   dtype=torch.float32,
                                                   device="cpu",
                                                   pin_memory=pin_memory)
        self.mirostat_eta_cpu = self.mirostat_eta_cpu_tensor.numpy()
        self.mirostat_reqs: set[str] = set()

        # Skew related data structures
        self.skew = torch.empty((max_num_reqs, ),
                                dtype=torch.float32,
                                device=device)
        self.skew_cpu_tensor = torch.empty((max_num_reqs, ),
                                           dtype=torch.float32,
                                           device="cpu",
                                           pin_memory=pin_memory)
        self.skew_cpu = self.skew_cpu_tensor.numpy()
        self.skew_reqs: set[str] = set()

        # DRY related data structures
        self.dry_multiplier = torch.empty((max_num_reqs, ),
                                          dtype=torch.float32,
                                          device=device)
        self.dry_multiplier_cpu_tensor = torch.empty((max_num_reqs, ),
                                                     dtype=torch.float32,
                                                     device="cpu",
                                                     pin_memory=pin_memory)
        self.dry_multiplier_cpu = self.dry_multiplier_cpu_tensor.numpy()
        self.dry_base = torch.empty((max_num_reqs, ),
                                    dtype=torch.float32,
                                    device=device)
        self.dry_base_cpu_tensor = torch.empty((max_num_reqs, ),
                                               dtype=torch.float32,
                                               device="cpu",
                                               pin_memory=pin_memory)
        self.dry_base_cpu = self.dry_base_cpu_tensor.numpy()
        self.dry_allowed_length = torch.empty((max_num_reqs, ),
                                              dtype=torch.int32,
                                              device=device)
        self.dry_allowed_length_cpu_tensor = torch.empty((max_num_reqs, ),
                                                         dtype=torch.int32,
                                                         device="cpu",
                                                         pin_memory=pin_memory)
        self.dry_allowed_length_cpu = self.dry_allowed_length_cpu_tensor.numpy()
        self.dry_sequence_breaker_ids: dict[int, list[int]] = {}
        self.dry_ranges = torch.empty((max_num_reqs, ),
                                      dtype=torch.int32,
                                      device=device)
        self.dry_ranges_cpu_tensor = torch.empty((max_num_reqs, ),
                                                 dtype=torch.int32,
                                                 device="cpu",
                                                 pin_memory=pin_memory)
        self.dry_ranges_cpu = self.dry_ranges_cpu_tensor.numpy()
        self.dry_max_ngram = torch.empty((max_num_reqs, ),
                                         dtype=torch.int32,
                                         device=device)
        self.dry_max_ngram_cpu_tensor = torch.empty((max_num_reqs, ),
                                                    dtype=torch.int32,
                                                    device="cpu",
                                                    pin_memory=pin_memory)
        self.dry_max_ngram_cpu = self.dry_max_ngram_cpu_tensor.numpy()
        self.dry_max_occurrences = torch.empty((max_num_reqs, ),
                                               dtype=torch.int32,
                                               device=device)
        self.dry_max_occurrences_cpu_tensor = torch.empty((max_num_reqs, ),
                                                          dtype=torch.int32,
                                                          device="cpu",
                                                          pin_memory=pin_memory)
        self.dry_max_occurrences_cpu = (
            self.dry_max_occurrences_cpu_tensor.numpy())
        self.dry_early_exit_match_len = torch.empty((max_num_reqs, ),
                                                    dtype=torch.int32,
                                                    device=device)
        self.dry_early_exit_match_len_cpu_tensor = torch.empty((max_num_reqs, ),
                                                               dtype=torch.int32,
                                                               device="cpu",
                                                               pin_memory=pin_memory)
        self.dry_early_exit_match_len_cpu = (
            self.dry_early_exit_match_len_cpu_tensor.numpy())
        self.dry_reqs: set[str] = set()

        # No repeat ngram related data structures
        self.no_repeat_ngram_size = torch.empty((max_num_reqs, ),
                                                dtype=torch.int32,
                                                device=device)
        self.no_repeat_ngram_size_cpu_tensor = torch.empty((max_num_reqs, ),
                                                           dtype=torch.int32,
                                                           device="cpu",
                                                           pin_memory=pin_memory)
        self.no_repeat_ngram_size_cpu = (
            self.no_repeat_ngram_size_cpu_tensor.numpy())
        self.no_repeat_ngram_reqs: set[str] = set()

        # Frequency penalty related data structures
        self.frequency_penalties = torch.empty((max_num_reqs, ),
                                               dtype=torch.float,
                                               device=device)
        self.frequency_penalties_cpu_tensor = torch.empty(
            (max_num_reqs, ),
            dtype=torch.float,
            device="cpu",
            pin_memory=pin_memory)
        self.frequency_penalties_cpu = \
            self.frequency_penalties_cpu_tensor.numpy()
        self.frequency_penalties_reqs: set[str] = set()

        # Presence penalty related data structures
        self.presence_penalties = torch.empty((max_num_reqs, ),
                                              dtype=torch.float,
                                              device=device)
        self.presence_penalties_cpu_tensor = torch.empty((max_num_reqs, ),
                                                         dtype=torch.float,
                                                         device="cpu",
                                                         pin_memory=pin_memory)
        self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy(
        )
        self.presence_penalties_reqs: set[str] = set()

        # Repetition penalty related data structures
        self.repetition_penalties = torch.empty((max_num_reqs, ),
                                                dtype=torch.float,
                                                device=device)
        self.repetition_penalties_cpu_tensor = torch.empty(
            (max_num_reqs, ),
            dtype=torch.float,
            device="cpu",
            pin_memory=pin_memory)
        self.repetition_penalties_cpu = \
            self.repetition_penalties_cpu_tensor.numpy()
        self.repetition_penalties_reqs: set[str] = set()

        # lora related
        self.request_lora_mapping = np.zeros((self.max_num_reqs, ),
                                             dtype=np.int32)
        self.lora_id_to_request_ids: dict[int, set[str]] = {}
        self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

        # req_index -> generator
        # NOTE: The indices of the requests that do not have their own
        # generator should not be included in the dictionary.
        self.generators: dict[int, torch.Generator] = {}

        self.num_logprobs: dict[str, int] = {}
        # NOTE: num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}

        # To accumulate prompt logprobs tensor chunks across prefill steps.
        self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}

        # Internal representation of per-step batch state changes, used for
        # reordering persistent batch and generating logitsprocs batch state
        # updates. Should reset each step.
        self.batch_update_builder = BatchUpdateBuilder()

        # TODO convert this to LogitsProcessor
        self.has_allowed_token_ids: set[str] = set()
        # NOTE(lufang): In the mask tensor, if the corresponding token allowed,
        # the value is False. Since we use masked_fill_ to set -inf.
        self.allowed_token_ids_mask: Optional[torch.Tensor] = None
        self.allowed_token_ids_mask_cpu_tensor: Optional[torch.Tensor] = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        self.logits_processing_needs_token_ids = np.zeros(max_num_reqs,
                                                          dtype=bool)

        self.req_output_token_ids: list[Optional[list[int]]] = []

        # Store provided logitsprocs. If none are provided, initialize empty
        # data structure
        self.logitsprocs = logitsprocs or LogitsProcessors()

        # Sampler priority and temperature_last for priority-based execution
        self.sampler_priority: list[Optional[list[int]]] = [None] * max_num_reqs
        self.temperature_last: list[bool] = [False] * max_num_reqs

        # Persistent metadata for mirostat
        self.persistent_data: dict[int, dict[str, Any]] = {}

        # This is updated each time the batch constituents change.
        self.sampling_metadata = self._make_sampling_metadata()

        self.pooling_params: dict[str, PoolingParams] = {}

        # Cached reference to the GPU tensor of previously sampled tokens
        self.prev_sampled_token_ids: Optional[torch.Tensor] = None
        self.prev_sampled_token_ids_invalid_indices: Optional[set[int]] = None
        self.prev_req_id_to_index: Optional[dict[str, int]] = None

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    def _register_add_request(self, request: "CachedRequestState") -> int:
        """Track add-request operations for logits processors.
        Not applicable to pooling models.
        """

        # Fill the next empty index if there is one.
        if (new_req_index := self.batch_update_builder.pop_removed()) is None:
            # Append to end otherwise.
            new_req_index = self.num_reqs

        assert new_req_index < self.max_num_reqs
        self.batch_update_builder.batch_changed = True
        if request.sampling_params:
            # Detailed added request metadata is only required for non-pooling
            # models, to support logitsprocs.
            self.batch_update_builder.added.append(
                (new_req_index, request.sampling_params,
                 request.prompt_token_ids, request.output_token_ids))

        return new_req_index

    def add_request(
        self,
        request: "CachedRequestState",
    ) -> int:
        req_index = self._register_add_request(request)

        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        self.token_ids_cpu[
            req_index, :num_prompt_tokens] = request.prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index,
                           start_idx:end_idx] = request.output_token_ids
        # Number of token ids in token_ids_cpu.
        # NOTE: This may include spec decode tokens.
        self.num_tokens[req_index] = request.num_tokens
        # Number of tokens without spec decode tokens.
        self.num_tokens_no_spec[req_index] = request.num_tokens

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids, req_index)

        if sampling_params := request.sampling_params:
            if (self.is_spec_decode
                    and is_spec_decode_unsupported(sampling_params)):
                self.spec_decode_unsupported_reqs.add(req_id)
            if sampling_params.sampling_type == SamplingType.GREEDY:
                # Avoid later division by zero.
                self.temperature_cpu[req_index] = -1.0
                self.greedy_reqs.add(req_id)
            else:
                self.temperature_cpu[req_index] = sampling_params.temperature
                self.random_reqs.add(req_id)

            # Add dynatemp parameters
            self.dynatemp_min_cpu[req_index] = sampling_params.dynatemp_min
            self.dynatemp_max_cpu[req_index] = sampling_params.dynatemp_max
            self.dynatemp_exp_cpu[req_index] = sampling_params.dynatemp_exponent
            if (sampling_params.dynatemp_min > _SAMPLING_EPS or
                sampling_params.dynatemp_max > _SAMPLING_EPS):
                self.dynatemp_reqs.add(req_id)

            self.top_p_cpu[req_index] = sampling_params.top_p
            if sampling_params.top_p < 1:
                self.top_p_reqs.add(req_id)
            top_k = sampling_params.top_k
            if 0 < top_k < self.vocab_size:
                self.top_k_reqs.add(req_id)
            else:
                top_k = self.vocab_size
            self.top_k_cpu[req_index] = top_k
            self.frequency_penalties_cpu[
                req_index] = sampling_params.frequency_penalty
            if sampling_params.frequency_penalty != 0.0:
                self.frequency_penalties_reqs.add(req_id)
            self.presence_penalties_cpu[
                req_index] = sampling_params.presence_penalty
            if sampling_params.presence_penalty != 0.0:
                self.presence_penalties_reqs.add(req_id)
            self.repetition_penalties_cpu[
                req_index] = sampling_params.repetition_penalty
            if sampling_params.repetition_penalty != 1.0:
                self.repetition_penalties_reqs.add(req_id)

            self.top_a_cpu[req_index] = sampling_params.top_a
            if sampling_params.top_a > 0:
                self.top_a_reqs.add(req_id)

            self.tfs_cpu[req_index] = sampling_params.tfs
            if sampling_params.tfs < 1.0:
                self.tfs_reqs.add(req_id)

            self.eta_cutoff_cpu[req_index] = sampling_params.eta_cutoff
            if sampling_params.eta_cutoff > 0:
                self.eta_cutoff_reqs.add(req_id)

            self.epsilon_cutoff_cpu[req_index] = sampling_params.epsilon_cutoff
            if sampling_params.epsilon_cutoff > 0:
                self.epsilon_cutoff_reqs.add(req_id)

            self.typical_p_cpu[req_index] = sampling_params.typical_p
            if sampling_params.typical_p < 1.0:
                self.typical_p_reqs.add(req_id)

            self.quadratic_smoothing_factor_cpu[req_index] = \
                sampling_params.smoothing_factor
            self.quadratic_smoothing_curve_cpu[req_index] = \
                sampling_params.smoothing_curve
            if sampling_params.smoothing_factor > 0:
                self.quadratic_reqs.add(req_id)

            self.xtc_threshold_cpu[req_index] = sampling_params.xtc_threshold
            self.xtc_probability_cpu[req_index] = sampling_params.xtc_probability
            if sampling_params.xtc_probability > 0:
                self.xtc_reqs.add(req_id)

            self.top_nsigma_cpu[req_index] = sampling_params.nsigma
            if sampling_params.nsigma > 0:
                self.top_nsigma_reqs.add(req_id)

            self.mirostat_mode_cpu[req_index] = sampling_params.mirostat_mode
            self.mirostat_tau_cpu[req_index] = sampling_params.mirostat_tau
            self.mirostat_eta_cpu[req_index] = sampling_params.mirostat_eta
            if sampling_params.mirostat_mode == 2:
                self.mirostat_reqs.add(req_id)

            self.skew_cpu[req_index] = sampling_params.skew
            if sampling_params.skew != 0:
                self.skew_reqs.add(req_id)

            self.dry_multiplier_cpu[req_index] = sampling_params.dry_multiplier
            self.dry_base_cpu[req_index] = sampling_params.dry_base
            self.dry_allowed_length_cpu[req_index] = \
                sampling_params.dry_allowed_length
            self.dry_ranges_cpu[req_index] = sampling_params.dry_range
            self.dry_max_ngram_cpu[req_index] = sampling_params.dry_max_ngram
            self.dry_max_occurrences_cpu[req_index] = \
                sampling_params.dry_max_occurrences
            self.dry_early_exit_match_len_cpu[req_index] = \
                sampling_params.dry_early_exit_match_len
            if sampling_params.dry_multiplier > 0:
                self.dry_reqs.add(req_id)
                if sampling_params.dry_sequence_breaker_ids:
                    self.dry_sequence_breaker_ids[req_index] = \
                        sampling_params.dry_sequence_breaker_ids

            self.no_repeat_ngram_size_cpu[req_index] = \
                sampling_params.no_repeat_ngram_size
            if sampling_params.no_repeat_ngram_size > 0:
                self.no_repeat_ngram_reqs.add(req_id)

            self.sampler_priority[req_index] = sampling_params.sampler_priority
            self.temperature_last[req_index] = sampling_params.temperature_last

            # NOTE: self.generators should not include the requests that
            # do not have their own generator.
            if request.generator is not None:
                self.generators[req_index] = request.generator

            if sampling_params.logprobs is not None:
                self.num_logprobs[req_id] = sampling_params.logprobs
            if sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = sampling_params.prompt_logprobs
            if sampling_params.logit_bias is not None:
                self.logit_bias[req_index] = sampling_params.logit_bias

            if sampling_params.allowed_token_ids:
                self.has_allowed_token_ids.add(req_id)
                if self.allowed_token_ids_mask_cpu_tensor is None:
                    # Lazy allocation for this tensor, which can be large.
                    # False means we don't fill with -inf.
                    self.allowed_token_ids_mask = torch.zeros(self.max_num_reqs,
                                                            self.vocab_size,
                                                            dtype=torch.bool,
                                                            device=self.device)
                    self.allowed_token_ids_mask_cpu_tensor = torch.zeros(
                        self.max_num_reqs,
                        self.vocab_size,
                        dtype=torch.bool,
                        device="cpu")
                self.allowed_token_ids_mask_cpu_tensor[req_index] = True
                # False means we don't fill with -inf.
                self.allowed_token_ids_mask_cpu_tensor[req_index][
                    sampling_params.allowed_token_ids] = False

            if sampling_params.bad_words_token_ids:
                self.bad_words_token_ids[
                    req_index] = sampling_params.bad_words_token_ids

        # Add request lora ID
        if request.lora_request:
            lora_id = request.lora_request.lora_int_id
            if lora_id not in self.lora_id_to_request_ids:
                self.lora_id_to_request_ids[lora_id] = set()

            self.request_lora_mapping[req_index] = lora_id
            self.lora_id_to_request_ids[lora_id].add(request.req_id)
            self.lora_id_to_lora_request[lora_id] = request.lora_request
        else:
            # No LoRA
            self.request_lora_mapping[req_index] = 0

        self.persistent_data[req_index] = request.persistent_data.copy()

        return req_index

    def remove_request(self, req_id: str) -> Optional[int]:
        """This method must always be followed by a call to condense().
        Args:
          req_id: request to remove
        Returns:
          Removed request index, or `None` if `req_id` not recognized
        """

        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self.batch_update_builder.removed_append(req_index)
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        # LoRA
        lora_id = self.request_lora_mapping[req_index]
        if lora_id != 0:
            lora_req_ids = self.lora_id_to_request_ids[lora_id]
            lora_req_ids.discard(req_id)
            if not lora_req_ids:
                del self.lora_id_to_request_ids[lora_id]
                del self.lora_id_to_lora_request[lora_id]
            self.request_lora_mapping[req_index] = 0

        if self.is_pooling_model:
            self.pooling_params.pop(req_id, None)
            return req_index

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.spec_decode_unsupported_reqs.discard(req_id)
        self.dynatemp_reqs.discard(req_id)
        self.top_a_reqs.discard(req_id)
        self.tfs_reqs.discard(req_id)
        self.eta_cutoff_reqs.discard(req_id)
        self.epsilon_cutoff_reqs.discard(req_id)
        self.typical_p_reqs.discard(req_id)
        self.quadratic_reqs.discard(req_id)
        self.xtc_reqs.discard(req_id)
        self.top_nsigma_reqs.discard(req_id)
        self.mirostat_reqs.discard(req_id)
        self.skew_reqs.discard(req_id)
        self.dry_reqs.discard(req_id)
        self.no_repeat_ngram_reqs.discard(req_id)
        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.num_prompt_logprobs.pop(req_id, None)
        self.in_progress_prompt_logprobs_cpu.pop(req_id, None)

        self.has_allowed_token_ids.discard(req_id)
        if self.allowed_token_ids_mask_cpu_tensor is not None:
            # False means we don't fill with -inf.
            self.allowed_token_ids_mask_cpu_tensor[req_index].fill_(False)
        self.bad_words_token_ids.pop(req_index, None)

        # Clean up persistent data
        self.persistent_data.pop(req_index, None)

        return req_index

    def swap_states(self, i1: int, i2: int) -> None:
        old_id_i1 = self._req_ids[i1]
        old_id_i2 = self._req_ids[i2]
        self._req_ids[i1], self._req_ids[i2] =\
            self._req_ids[i2], self._req_ids[i1] # noqa
        self.req_output_token_ids[i1], self.req_output_token_ids[i2] =\
            self.req_output_token_ids[i2], self.req_output_token_ids[i1]
        assert old_id_i1 is not None and old_id_i2 is not None
        self.req_id_to_index[old_id_i1], self.req_id_to_index[old_id_i2] =\
            self.req_id_to_index[old_id_i2], self.req_id_to_index[old_id_i1]
        self.num_tokens[i1], self.num_tokens[i2] =\
            self.num_tokens[i2], self.num_tokens[i1]
        self.num_tokens_no_spec[i1], self.num_tokens_no_spec[i2] =\
            self.num_tokens_no_spec[i2], self.num_tokens_no_spec[i1]
        self.num_prompt_tokens[i1], self.num_prompt_tokens[i2] =\
            self.num_prompt_tokens[i2], self.num_prompt_tokens[i1]
        self.num_computed_tokens_cpu[i1], self.num_computed_tokens_cpu[i2] =\
            self.num_computed_tokens_cpu[i2], self.num_computed_tokens_cpu[i1]

        # NOTE: the following is unsafe
        # self.token_ids_cpu[i1, ...], self.token_ids_cpu[i2, ...], =\
        #     self.token_ids_cpu[i2, ...], self.token_ids_cpu[i1, ...]
        # instead, we need to temporiarily copy the data for one of the indices
        # TODO(lucas): optimize this by only copying valid indices
        tmp = self.token_ids_cpu[i1, ...].copy()
        self.token_ids_cpu[i1, ...] = self.token_ids_cpu[i2, ...]
        self.token_ids_cpu[i2, ...] = tmp

        self.block_table.swap_row(i1, i2)

        self.request_lora_mapping[i1], self.request_lora_mapping[i2] = \
            self.request_lora_mapping[i2], self.request_lora_mapping[i1]

        if self.is_pooling_model:
            # Sampling and logits parameters don't apply to pooling models.
            return

        # For autoregressive models, track detailed request reordering info
        # to support logitsprocs.
        self.batch_update_builder.moved.append(
            (i1, i2, MoveDirectionality.SWAP))

        self.temperature_cpu[i1], self.temperature_cpu[i2] =\
            self.temperature_cpu[i2], self.temperature_cpu[i1]
        self.dynatemp_min_cpu[i1], self.dynatemp_min_cpu[i2] =\
            self.dynatemp_min_cpu[i2], self.dynatemp_min_cpu[i1]
        self.dynatemp_max_cpu[i1], self.dynatemp_max_cpu[i2] =\
            self.dynatemp_max_cpu[i2], self.dynatemp_max_cpu[i1]
        self.dynatemp_exp_cpu[i1], self.dynatemp_exp_cpu[i2] =\
            self.dynatemp_exp_cpu[i2], self.dynatemp_exp_cpu[i1]
        self.top_p_cpu[i1], self.top_p_cpu[i2] =\
            self.top_p_cpu[i2], self.top_p_cpu[i1]
        self.top_k_cpu[i1], self.top_k_cpu[i2] =\
            self.top_k_cpu[i2], self.top_k_cpu[i1]
        self.frequency_penalties_cpu[i1], self.frequency_penalties_cpu[i2] =\
            self.frequency_penalties_cpu[i2], self.frequency_penalties_cpu[i1]
        self.presence_penalties_cpu[i1], self.presence_penalties_cpu[i2] =\
            self.presence_penalties_cpu[i2], self.presence_penalties_cpu[i1]
        self.repetition_penalties_cpu[i1], self.repetition_penalties_cpu[i2] =\
            self.repetition_penalties_cpu[i2], self.repetition_penalties_cpu[i1]
        self.top_a_cpu[i1], self.top_a_cpu[i2] =\
            self.top_a_cpu[i2], self.top_a_cpu[i1]
        self.tfs_cpu[i1], self.tfs_cpu[i2] =\
            self.tfs_cpu[i2], self.tfs_cpu[i1]
        self.eta_cutoff_cpu[i1], self.eta_cutoff_cpu[i2] =\
            self.eta_cutoff_cpu[i2], self.eta_cutoff_cpu[i1]
        self.epsilon_cutoff_cpu[i1], self.epsilon_cutoff_cpu[i2] =\
            self.epsilon_cutoff_cpu[i2], self.epsilon_cutoff_cpu[i1]
        self.typical_p_cpu[i1], self.typical_p_cpu[i2] =\
            self.typical_p_cpu[i2], self.typical_p_cpu[i1]
        self.quadratic_smoothing_factor_cpu[i1], \
            self.quadratic_smoothing_factor_cpu[i2] =\
            self.quadratic_smoothing_factor_cpu[i2], \
                self.quadratic_smoothing_factor_cpu[i1]
        self.quadratic_smoothing_curve_cpu[i1], \
            self.quadratic_smoothing_curve_cpu[i2] =\
            self.quadratic_smoothing_curve_cpu[i2], \
                self.quadratic_smoothing_curve_cpu[i1]
        self.xtc_threshold_cpu[i1], self.xtc_threshold_cpu[i2] =\
            self.xtc_threshold_cpu[i2], self.xtc_threshold_cpu[i1]
        self.xtc_probability_cpu[i1], self.xtc_probability_cpu[i2] =\
            self.xtc_probability_cpu[i2], self.xtc_probability_cpu[i1]
        self.top_nsigma_cpu[i1], self.top_nsigma_cpu[i2] =\
            self.top_nsigma_cpu[i2], self.top_nsigma_cpu[i1]
        self.mirostat_mode_cpu[i1], self.mirostat_mode_cpu[i2] =\
            self.mirostat_mode_cpu[i2], self.mirostat_mode_cpu[i1]
        self.mirostat_tau_cpu[i1], self.mirostat_tau_cpu[i2] =\
            self.mirostat_tau_cpu[i2], self.mirostat_tau_cpu[i1]
        self.mirostat_eta_cpu[i1], self.mirostat_eta_cpu[i2] =\
            self.mirostat_eta_cpu[i2], self.mirostat_eta_cpu[i1]
        self.skew_cpu[i1], self.skew_cpu[i2] =\
            self.skew_cpu[i2], self.skew_cpu[i1]
        self.dry_multiplier_cpu[i1], self.dry_multiplier_cpu[i2] =\
            self.dry_multiplier_cpu[i2], self.dry_multiplier_cpu[i1]
        self.dry_base_cpu[i1], self.dry_base_cpu[i2] =\
            self.dry_base_cpu[i2], self.dry_base_cpu[i1]
        self.dry_allowed_length_cpu[i1], self.dry_allowed_length_cpu[i2] =\
            self.dry_allowed_length_cpu[i2], self.dry_allowed_length_cpu[i1]
        self.dry_ranges_cpu[i1], self.dry_ranges_cpu[i2] =\
            self.dry_ranges_cpu[i2], self.dry_ranges_cpu[i1]
        self.dry_max_ngram_cpu[i1], self.dry_max_ngram_cpu[i2] =\
            self.dry_max_ngram_cpu[i2], self.dry_max_ngram_cpu[i1]
        self.dry_max_occurrences_cpu[i1], self.dry_max_occurrences_cpu[i2] =\
            self.dry_max_occurrences_cpu[i2], self.dry_max_occurrences_cpu[i1]
        self.dry_early_exit_match_len_cpu[i1], \
            self.dry_early_exit_match_len_cpu[i2] =\
            self.dry_early_exit_match_len_cpu[i2], \
                self.dry_early_exit_match_len_cpu[i1]
        self.no_repeat_ngram_size_cpu[i1], self.no_repeat_ngram_size_cpu[i2] =\
            self.no_repeat_ngram_size_cpu[i2], self.no_repeat_ngram_size_cpu[i1]

        swap_dict_values(self.generators, i1, i2)
        swap_dict_values(self.bad_words_token_ids, i1, i2)

        if self.allowed_token_ids_mask_cpu_tensor is not None:
            self.allowed_token_ids_mask_cpu_tensor[i1], \
                self.allowed_token_ids_mask_cpu_tensor[i2] =\
                self.allowed_token_ids_mask_cpu_tensor[i2], \
                    self.allowed_token_ids_mask_cpu_tensor[i1]

        # Swap persistent data
        self.persistent_data[i1], self.persistent_data[i2] = \
            self.persistent_data[i2], self.persistent_data[i1]

    def condense(self) -> None:
        """Slide non-empty requests down into lower, empty indices.
        Any consecutive empty indices at the very end of the list are not
        filled.
        Args:
          empty_req_indices: empty indices which may be filled.
        Returns:
          swaps: list of (from,to) swap tuples for moved requests
          empty_req_indices: indices not filled by condensation
        """
        num_reqs = self.num_reqs

        if not (empty_req_indices := self.batch_update_builder.removed):
            # All removed requests were replaced by added requests, or else no
            # requests were removed at all. No condense() needed
            return
        if num_reqs == 0:
            # The batched states are empty.
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            return

        # NOTE: This function assumes that the empty_req_indices
        # is sorted in descending order.
        last_req_index = num_reqs + len(empty_req_indices) - 1
        while empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = self.batch_update_builder.peek_removed()
            assert empty_index is not None
            if empty_index >= last_req_index:
                break

            # Move active request down into empty request
            # index.
            self.batch_update_builder.pop_removed()
            req_id = self._req_ids[last_req_index]
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            # Move persistent data
            self.persistent_data[empty_index] = \
                self.persistent_data[last_req_index]
            self.persistent_data[last_req_index] = {}

            # Copy token data
            num_tokens = self.num_tokens[last_req_index]
            self.token_ids_cpu[empty_index, :num_tokens] = self.token_ids_cpu[
                last_req_index, :num_tokens]
            self.num_tokens[empty_index] = num_tokens
            self.num_tokens_no_spec[empty_index] = self.num_tokens_no_spec[
                last_req_index]
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[
                last_req_index]
            self.num_computed_tokens_cpu[
                empty_index] = self.num_computed_tokens_cpu[last_req_index]

            # Update the block table.
            self.block_table.move_row(last_req_index, empty_index)

            self.request_lora_mapping[empty_index] = self.request_lora_mapping[
                last_req_index]

            if self.is_pooling_model:
                last_req_index -= 1
                # Samping state not used by pooling models.
                continue

            # Autoregressive models require detailed tracking of condense
            # operations to support logitsprocs
            self.batch_update_builder.moved.append(
                (last_req_index, empty_index,
                 MoveDirectionality.UNIDIRECTIONAL))

            self.temperature_cpu[empty_index] = self.temperature_cpu[
                last_req_index]
            self.dynatemp_min_cpu[empty_index] = \
                self.dynatemp_min_cpu[last_req_index]
            self.dynatemp_max_cpu[empty_index] = \
                self.dynatemp_max_cpu[last_req_index]
            self.dynatemp_exp_cpu[empty_index] = \
                self.dynatemp_exp_cpu[last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            self.frequency_penalties_cpu[
                empty_index] = self.frequency_penalties_cpu[last_req_index]
            self.presence_penalties_cpu[
                empty_index] = self.presence_penalties_cpu[last_req_index]
            self.repetition_penalties_cpu[
                empty_index] = self.repetition_penalties_cpu[last_req_index]
            self.top_a_cpu[empty_index] = self.top_a_cpu[last_req_index]
            self.tfs_cpu[empty_index] = self.tfs_cpu[last_req_index]
            self.eta_cutoff_cpu[empty_index] = \
                self.eta_cutoff_cpu[last_req_index]
            self.epsilon_cutoff_cpu[empty_index] = \
                self.epsilon_cutoff_cpu[last_req_index]
            self.typical_p_cpu[empty_index] = self.typical_p_cpu[last_req_index]
            self.quadratic_smoothing_factor_cpu[empty_index] = \
                self.quadratic_smoothing_factor_cpu[last_req_index]
            self.quadratic_smoothing_curve_cpu[empty_index] = \
                self.quadratic_smoothing_curve_cpu[last_req_index]
            self.xtc_threshold_cpu[empty_index] = \
                self.xtc_threshold_cpu[last_req_index]
            self.xtc_probability_cpu[empty_index] = \
                self.xtc_probability_cpu[last_req_index]
            self.top_nsigma_cpu[empty_index] = \
                self.top_nsigma_cpu[last_req_index]
            self.mirostat_mode_cpu[empty_index] = self.mirostat_mode_cpu[
                last_req_index]
            self.mirostat_tau_cpu[empty_index] = self.mirostat_tau_cpu[
                last_req_index]
            self.mirostat_eta_cpu[empty_index] = self.mirostat_eta_cpu[
                last_req_index]
            self.skew_cpu[empty_index] = self.skew_cpu[last_req_index]
            self.dry_multiplier_cpu[empty_index] = \
                self.dry_multiplier_cpu[last_req_index]
            self.dry_base_cpu[empty_index] = self.dry_base_cpu[last_req_index]
            self.dry_allowed_length_cpu[empty_index] = \
                self.dry_allowed_length_cpu[last_req_index]
            self.dry_ranges_cpu[empty_index] = \
                self.dry_ranges_cpu[last_req_index]
            self.dry_max_ngram_cpu[empty_index] = \
                self.dry_max_ngram_cpu[last_req_index]
            self.dry_max_occurrences_cpu[empty_index] = \
                self.dry_max_occurrences_cpu[last_req_index]
            self.dry_early_exit_match_len_cpu[empty_index] = \
                self.dry_early_exit_match_len_cpu[last_req_index]
            self.no_repeat_ngram_size_cpu[empty_index] = \
                self.no_repeat_ngram_size_cpu[last_req_index]
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            if self.allowed_token_ids_mask_cpu_tensor is not None:
                self.allowed_token_ids_mask_cpu_tensor[
                    empty_index] = self.allowed_token_ids_mask_cpu_tensor[
                        last_req_index]

            bad_words_token_ids = self.bad_words_token_ids.pop(
                last_req_index, None)
            if bad_words_token_ids is not None:
                self.bad_words_token_ids[empty_index] = bad_words_token_ids
            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

        # Trim lists to the batch size.
        del self._req_ids[num_reqs:]
        del self.req_output_token_ids[num_reqs:]

    def refresh_metadata(self):
        """Apply any batch updates to sampling metadata."""
        if self.is_pooling_model:
            batch_changed = self.batch_update_builder.reset()
            if batch_changed:
                self.sampling_metadata = self._make_sampling_metadata()
            return

        # For non-pooling models - generate and apply logitsprocs update;
        # reset batch update tracking.
        # Update sampling metadata if batch state is changed.
        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
        for logit_proc in self.logitsprocs.all:
            logit_proc.update_state(batch_update)
        if batch_update:
            self.sampling_metadata = self._make_sampling_metadata()

    def _make_sampling_metadata(self) -> SamplingMetadata:
        num_reqs = self.num_reqs
        if not self.all_greedy:
            copy_slice(self.temperature_cpu_tensor, self.temperature, num_reqs)
            temperature = self.temperature[:num_reqs]
        else:
            temperature = None

        if not self.no_dynatemp:
            copy_slice(self.dynatemp_min_cpu_tensor,
                       self.dynatemp_min, num_reqs)
            copy_slice(self.dynatemp_max_cpu_tensor,
                       self.dynatemp_max, num_reqs)
            copy_slice(self.dynatemp_exp_cpu_tensor,
                       self.dynatemp_exp, num_reqs)
        if not self.no_top_p:
            copy_slice(self.top_p_cpu_tensor, self.top_p, num_reqs)
        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)

        if not self.no_top_a:
            copy_slice(self.top_a_cpu_tensor, self.top_a, num_reqs)
        if not self.no_tfs:
            copy_slice(self.tfs_cpu_tensor, self.tfs, num_reqs)
        if not self.no_eta_cutoff:
            copy_slice(self.eta_cutoff_cpu_tensor, self.eta_cutoff, num_reqs)
        if not self.no_epsilon_cutoff:
            copy_slice(self.epsilon_cutoff_cpu_tensor,
                       self.epsilon_cutoff, num_reqs)
        if not self.no_typical_p:
            copy_slice(self.typical_p_cpu_tensor, self.typical_p, num_reqs)
        if not self.no_quadratic:
            copy_slice(self.quadratic_smoothing_factor_cpu_tensor,
                       self.quadratic_smoothing_factor, num_reqs)
            copy_slice(self.quadratic_smoothing_curve_cpu_tensor,
                       self.quadratic_smoothing_curve, num_reqs)
        if not self.no_xtc:
            copy_slice(self.xtc_threshold_cpu_tensor,
                       self.xtc_threshold, num_reqs)
            copy_slice(self.xtc_probability_cpu_tensor,
                       self.xtc_probability, num_reqs)
        if not self.no_top_nsigma:
            copy_slice(self.top_nsigma_cpu_tensor, self.top_nsigma, num_reqs)
        if not self.no_mirostat:
            copy_slice(self.mirostat_mode_cpu_tensor,
                       self.mirostat_mode, num_reqs)
            copy_slice(self.mirostat_tau_cpu_tensor,
                       self.mirostat_tau, num_reqs)
            copy_slice(self.mirostat_eta_cpu_tensor,
                       self.mirostat_eta, num_reqs)
        if not self.no_skew:
            copy_slice(self.skew_cpu_tensor, self.skew, num_reqs)
        if not self.no_dry:
            copy_slice(self.dry_multiplier_cpu_tensor,
                       self.dry_multiplier, num_reqs)
            copy_slice(self.dry_base_cpu_tensor, self.dry_base, num_reqs)
            copy_slice(self.dry_allowed_length_cpu_tensor,
                       self.dry_allowed_length, num_reqs)
            copy_slice(self.dry_ranges_cpu_tensor, self.dry_ranges, num_reqs)
            copy_slice(self.dry_max_ngram_cpu_tensor,
                       self.dry_max_ngram, num_reqs)
            copy_slice(self.dry_max_occurrences_cpu_tensor,
                       self.dry_max_occurrences, num_reqs)
            copy_slice(self.dry_early_exit_match_len_cpu_tensor,
                       self.dry_early_exit_match_len, num_reqs)
        if not self.no_no_repeat_ngram:
            copy_slice(self.no_repeat_ngram_size_cpu_tensor,
                       self.no_repeat_ngram_size, num_reqs)

        if not self.no_penalties:
            # Since syncing these tensors is expensive only copy them
            # if necessary i.e. if there are requests which require
            # penalties to be applied during sampling.
            copy_slice(self.frequency_penalties_cpu_tensor,
                       self.frequency_penalties, num_reqs)
            copy_slice(self.presence_penalties_cpu_tensor,
                       self.presence_penalties, num_reqs)
            copy_slice(self.repetition_penalties_cpu_tensor,
                       self.repetition_penalties, num_reqs)

        needs_prompt_token_ids = (
            not self.no_penalties
            or self.logits_processing_needs_token_ids[:num_reqs].any())
        if needs_prompt_token_ids:
            # The prompt tokens are used only for applying penalties or
            # step pooling during the sampling/pooling process.
            # Hence copy these tensors only when there are requests which
            # need penalties/step_pooler to be applied.
            prompt_token_ids = self._make_prompt_token_ids_tensor()
        else:
            prompt_token_ids = None

        allowed_token_ids_mask: Optional[torch.Tensor] = None
        if not self.no_allowed_token_ids:
            assert self.allowed_token_ids_mask is not None
            copy_slice(self.allowed_token_ids_mask_cpu_tensor,
                       self.allowed_token_ids_mask, num_reqs)
            allowed_token_ids_mask = self.allowed_token_ids_mask[:num_reqs]

        # Process sampler priority - convert integers to SamplerID enums
        sampler_priority_processed = None
        if any(self.sampler_priority[:num_reqs]):
            # Take the first non-None priority list as the batch priority
            for priority_list in self.sampler_priority[:num_reqs]:
                if priority_list is not None:
                    sampler_priority_processed = [
                        SamplerID(pid) for pid in priority_list
                    ]
                    break

        return SamplingMetadata(
            temperature=temperature,
            dynatemp_min=(
                None if self.no_dynatemp else self.dynatemp_min[:num_reqs]),
            dynatemp_max=(
                None if self.no_dynatemp else self.dynatemp_max[:num_reqs]),
            dynatemp_exp=(
                None if self.no_dynatemp else self.dynatemp_exp[:num_reqs]),
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=None if self.no_top_p else self.top_p[:num_reqs],
            top_k=None if self.no_top_k else self.top_k[:num_reqs],
            top_a=None if self.no_top_a else self.top_a[:num_reqs],
            dry_multiplier=(
                None if self.no_dry else self.dry_multiplier[:num_reqs]),
            dry_base=(
                None if self.no_dry else self.dry_base[:num_reqs]),
            dry_allowed_length=(
                None if self.no_dry else self.dry_allowed_length[:num_reqs]),
            dry_sequence_breaker_ids=(
                None if self.no_dry else
                self._make_dry_sequence_breaker_ids_tensor(num_reqs)),
            dry_ranges=None if self.no_dry else self.dry_ranges[:num_reqs],
            dry_max_ngram=(
                None if self.no_dry else self.dry_max_ngram[:num_reqs]),
            dry_max_occurrences=(
                None if self.no_dry else self.dry_max_occurrences[:num_reqs]),
            dry_early_exit_match_len=(
                None if self.no_dry else
                self.dry_early_exit_match_len[:num_reqs]),
            no_repeat_ngram_size=(
                None if self.no_no_repeat_ngram else
                self.no_repeat_ngram_size[:num_reqs]),
            tfs=None if self.no_tfs else self.tfs[:num_reqs],
            eta_cutoff=(
                None if self.no_eta_cutoff else self.eta_cutoff[:num_reqs]),
            epsilon_cutoff=(
                None if self.no_epsilon_cutoff else
                self.epsilon_cutoff[:num_reqs]),
            typical_p=None if self.no_typical_p else self.typical_p[:num_reqs],
            quadratic_smoothing_factor=(
                None if self.no_quadratic else
                self.quadratic_smoothing_factor[:num_reqs]),
            quadratic_smoothing_curve=(
                None if self.no_quadratic else
                self.quadratic_smoothing_curve[:num_reqs]),
            xtc_threshold=(
                None if self.no_xtc else self.xtc_threshold[:num_reqs]),
            xtc_probability=(
                None if self.no_xtc else self.xtc_probability[:num_reqs]),
            top_nsigma=(
                None if self.no_top_nsigma else self.top_nsigma[:num_reqs]),
            mirostat_mode=(
                None if self.no_mirostat else self.mirostat_mode[:num_reqs]),
            mirostat_tau=(
                None if self.no_mirostat else self.mirostat_tau[:num_reqs]),
            mirostat_eta=(
                None if self.no_mirostat else self.mirostat_eta[:num_reqs]),
            skew=None if self.no_skew else self.skew[:num_reqs],
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[:num_reqs],
            presence_penalties=self.presence_penalties[:num_reqs],
            repetition_penalties=self.repetition_penalties[:num_reqs],
            output_token_ids=cast(list[list[int]], self.req_output_token_ids),
            no_penalties=self.no_penalties,
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=self.bad_words_token_ids,
            sampler_priority=sampler_priority_processed,
            temperature_last=any(self.temperature_last[:num_reqs]),
            logitsprocs=self.logitsprocs,
            persistent_data=self.persistent_data,
        )

    def get_pooling_params(self) -> list[PoolingParams]:
        assert len(self.req_ids) == len(self.pooling_params)
        return [self.pooling_params[req_id] for req_id in self.req_ids]

    def get_pooling_metadata(self) -> PoolingMetadata:
        pooling_params = self.get_pooling_params()

        return PoolingMetadata(
            prompt_lens=torch.from_numpy(
                self.num_prompt_tokens[:self.num_reqs]),
            prompt_token_ids=self.sampling_metadata.prompt_token_ids,
            pooling_params=pooling_params,
        )

    def _make_prompt_token_ids_tensor(self) -> torch.Tensor:
        num_reqs = self.num_reqs
        max_prompt_len = self.num_prompt_tokens[:num_reqs].max()
        prompt_token_ids_cpu_tensor = torch.empty(
            (self.num_reqs, max_prompt_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        prompt_token_ids = prompt_token_ids_cpu_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[:num_reqs, :max_prompt_len]
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        for i in range(num_reqs):
            prompt_token_ids[i, self.num_prompt_tokens[i]:] = self.vocab_size
        return prompt_token_ids_cpu_tensor.to(device=self.device,
                                              non_blocking=True)

    def _make_dry_sequence_breaker_ids_tensor(
        self, num_reqs: int) -> torch.Tensor:
        """Convert dry_sequence_breaker_ids dict to padded tensor like V0."""
        if not self.dry_sequence_breaker_ids:
            return torch.empty((num_reqs, 0), device=self.device,
                                dtype=torch.long)

        # Find max length
        max_len = max(
            len(ids) for ids in self.dry_sequence_breaker_ids.values())

        # Create padded tensor
        tensor_data = []
        for i in range(num_reqs):
            if i in self.dry_sequence_breaker_ids:
                ids = self.dry_sequence_breaker_ids[i]
                # Pad with zeros
                padded_ids = ids + [0] * (max_len - len(ids))
                tensor_data.append(padded_ids)
            else:
                tensor_data.append([0] * max_len)

        return torch.tensor(tensor_data, device=self.device, dtype=torch.long)

    def make_lora_inputs(
        self, num_scheduled_tokens: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        """
        Given the num_scheduled_tokens for each request in the batch, return
        datastructures used to activate the current LoRAs.
        Returns:
            1. prompt_lora_mapping: A tuple of size self.num_reqs where,
               prompt_lora_mapping[i] is the LoRA id to use for the ith prompt.
            2. token_lora_mapping: A tuple of size np.sum(num_scheduled_tokens)
               where, token_lora_mapping[i] is the LoRA id to use for ith token.
            3. lora_requests: Set of relevant LoRA requests.
        """

        req_lora_mapping = self.request_lora_mapping[:self.num_reqs]
        prompt_lora_mapping = tuple(req_lora_mapping)
        token_lora_mapping = tuple(
            req_lora_mapping.repeat(num_scheduled_tokens))
        active_lora_requests: set[LoRARequest] = set(
            self.lora_id_to_lora_request.values())

        return prompt_lora_mapping, token_lora_mapping, active_lora_requests

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        return len(self.top_k_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        return (len(self.presence_penalties_reqs) == 0
                and len(self.frequency_penalties_reqs) == 0
                and len(self.repetition_penalties_reqs) == 0)

    @property
    def max_num_logprobs(self) -> Optional[int]:
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def no_prompt_logprob(self) -> bool:
        return not self.num_prompt_logprobs

    @property
    def no_allowed_token_ids(self) -> bool:
        return len(self.has_allowed_token_ids) == 0

    @property
    def no_dynatemp(self) -> bool:
        return len(self.dynatemp_reqs) == 0

    @property
    def no_top_a(self) -> bool:
        return len(self.top_a_reqs) == 0

    @property
    def no_tfs(self) -> bool:
        return len(self.tfs_reqs) == 0

    @property
    def no_eta_cutoff(self) -> bool:
        return len(self.eta_cutoff_reqs) == 0

    @property
    def no_epsilon_cutoff(self) -> bool:
        return len(self.epsilon_cutoff_reqs) == 0

    @property
    def no_typical_p(self) -> bool:
        return len(self.typical_p_reqs) == 0

    @property
    def no_quadratic(self) -> bool:
        return len(self.quadratic_reqs) == 0

    @property
    def no_xtc(self) -> bool:
        return len(self.xtc_reqs) == 0

    @property
    def no_top_nsigma(self) -> bool:
        return len(self.top_nsigma_reqs) == 0

    @property
    def no_mirostat(self) -> bool:
        return len(self.mirostat_reqs) == 0

    @property
    def no_skew(self) -> bool:
        return len(self.skew_reqs) == 0

    @property
    def no_dry(self) -> bool:
        return len(self.dry_reqs) == 0

    @property
    def no_no_repeat_ngram(self) -> bool:
        return len(self.no_repeat_ngram_reqs) == 0
