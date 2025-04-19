from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch import nn

from aphrodite.common.config import (CacheConfig, DeviceConfig, ModelConfig,
                                     ParallelConfig, SchedulerConfig)
from aphrodite.common.sequence import (SamplerOutput, SequenceData,
                                       SequenceGroupMetadata)
from aphrodite.common.utils import pad_to_max_length
from aphrodite.modeling import SamplingMetadata
from aphrodite.modeling.model_loader.qaic import get_qaic_model


class QaicModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        cache_config : CacheConfig,
        device_config: DeviceConfig,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.device_config = device_config

        if model_config is not None and model_config.get_sliding_window():
            logger.warning("Sliding window is not supported on qaic. "
                           "The model will run without sliding window.")
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device
        assert self.device.type == "cpu", (
            f"Run with Device as cpu. Passed device type is {self.device}"
        )
        self.pin_memory = False

        # Lazy initialization.
        self.model: nn.Module  # initialize after load_model.

    def load_model(self) -> None:
        self.model = get_qaic_model(self.model_config,
                                parallel_config=self.parallel_config,
                                scheduler_config=self.scheduler_config,
                                cache_config=self.cache_config,
                                device_config=self.device_config)

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[np.ndarray] = []
        input_positions: List[np.ndarray] = []
        input_block_ids: List[int] = []
        seq_lens: List[int] = []

        max_seq_len = self.model.model.seq_len
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids: List[int] = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id: int = seq_ids[0]

            seq_data: SequenceData = seq_group_metadata.seq_data[seq_id]
            prompt_tokens: List[int] = seq_data.get_token_ids()
            seq_len = len(prompt_tokens)
            seq_lens.append(seq_len)

            assert seq_group_metadata.block_tables is not None
            block_table = seq_group_metadata.block_tables[seq_id]
            assert len(block_table) == 1
            input_block_id = block_table[0]
            input_block_ids.append(input_block_id)

            # pad input
            prompt_positions: List[int] = list(range(seq_len))
            if seq_len <= max_seq_len:
                prompt_tokens = np.asarray(
                                    pad_to_max_length(prompt_tokens,
                                                      max_seq_len, 2)
                                    ).reshape(1, max_seq_len)
                prompt_positions = np.asarray(
                                    pad_to_max_length(prompt_positions,
                                                      max_seq_len, -1)
                                    ).reshape(1, max_seq_len)
            else:
                # model_runner will perform chunking
                prompt_tokens = np.asarray(
                                    prompt_tokens
                                    ).reshape(1, -1)
                prompt_positions = np.asarray(
                                    prompt_positions
                                    ).reshape(1, -1)

            input_tokens.append(prompt_tokens)
            input_positions.append(prompt_positions)

        assert max_seq_len > 0

        return input_tokens, input_positions, input_block_ids, seq_lens

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = [] # holds token ids of each sequence
        input_positions: List[List[int]] = []
        input_block_ids: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids: List[int] = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data: SequenceData = seq_group_metadata.seq_data[seq_id]
                generation_token: int = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len: int = seq_data.get_len()
                position: int = seq_len - 1 # TODO: needs changing for qpc
                input_positions.append([position])

                assert seq_group_metadata.block_tables is not None
                block_table = seq_group_metadata.block_tables[seq_id]
                assert len(block_table) == 1
                input_block_id = block_table[0]
                input_block_ids.append(input_block_id)

        assert len(input_tokens[0]) == 1


        input_tokens = [np.asarray(x_i).reshape(1,1) for x_i in input_tokens]
        input_positions = [np.asarray(x_i).reshape(1,1
                                                   ) for x_i in input_positions]

        return input_tokens, input_positions, input_block_ids

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, SamplingMetadata]:
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, input_block_ids,
             seq_lens) = self._prepare_prompt(seq_group_metadata_list)
        else:
            (input_tokens, input_positions, input_block_ids
             ) = self._prepare_decode(seq_group_metadata_list)
            seq_lens = []
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            # query_lens is not needed if chunked prefill is not
            # supported. Since qaic worker doesn't support chunked prefill
            # just use seq_lens instead.
            seq_lens,
            self.device,
            self.pin_memory)

        return (input_tokens, input_positions, input_block_ids,
                sampling_metadata)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, input_block_ids, sampling_metadata
         ) = self.prepare_input_tensors(seq_group_metadata_list)

        # Compute the logits.
        is_prompt: bool = seq_group_metadata_list[0].is_prompt
        hidden_states: torch.Tensor = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            batch_indices=input_block_ids,
            is_prompt=is_prompt
        )

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Sample the next token.
        output: Optional[SamplerOutput] = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return output

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()
