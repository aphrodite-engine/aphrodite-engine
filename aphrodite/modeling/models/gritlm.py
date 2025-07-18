from array import array
from typing import Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from aphrodite.attention.backends.xformers import XFormersImpl
from aphrodite.common.config import AphroditeConfig, ModelConfig
from aphrodite.common.sequence import (IntermediateTensors, PoolerOutput,
                                       PoolingSequenceGroupOutput)
from aphrodite.forward_context import get_forward_context
from aphrodite.modeling.layers.pooler import PoolerHead
from aphrodite.modeling.models.llama import LlamaForCausalLM
from aphrodite.modeling.pooling_metadata import PoolingMetadata, PoolingTensors
from aphrodite.transformers_utils.tokenizer import cached_tokenizer_from_config

from .interfaces import SupportsV0Only


class GritLMPooler(nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.model_config = model_config

        tokenizer = cached_tokenizer_from_config(self.model_config)

        # Collect the tokens needed for pattern matching.
        # "▁<" is different from "_<". The former uses "▁" to indicate that
        # the next token is the start of a word.
        # "<0x0A>" is the newline token (i.e. "\n")."
        self.token_ids = {
            tok: tokenizer.convert_tokens_to_ids([tok])[0]
            for tok in ["<s>", "▁<", "<", "|", "embed", ">", "<0x0A>", "user"]
        }

        def tokens_to_ids(tokens: list[str]) -> array:
            return array("i", [self.token_ids[token] for token in tokens])

        self.user_pattern_ids = tokens_to_ids(
            ["▁<", "|", "user", "|", ">", "<0x0A>"])
        self.embed_newline_pattern_ids = tokens_to_ids(
            ["<0x0A>", "<", "|", "embed", "|", ">", "<0x0A>"])
        self.embed_pattern_ids = tokens_to_ids(
            ["▁<", "|", "embed", "|", ">", "<0x0A>"])

        self.head = PoolerHead(normalize=True, softmax=False)

    def _find_array(self, arr: array, target: array, start_idx: int) -> int:
        """
        Find the first occurrence of target in arr starting from start_idx.

        Args:
        arr: The array to search within
        target: The consecutive subsequence to find
        start_idx: The starting index to search from

        Returns:
        int: The index of the first occurrence of target in arr.
        """
        if start_idx < 0:
            raise ValueError("start_idx must be non-negative")
        if not target or not arr:
            raise ValueError("Empty arr or target not allowed")

        target_len = len(target)
        for i in range(start_idx, len(arr) - target_len + 1):
            if arr[i:i + target_len] == target:
                return i
        return -1

    def _get_instruction_len(self, prompt_token_ids: array) -> int:
        """
        Get the length of the instruction in the prompt.

        We do a pattern matching to find the instruction in the prompt,
        and then return the length of the instruction.

        The pattern matching is done using integers instead of strings
        because the prompt is given as a list of token IDs.
        """

        instruction_len = 0

        # Return no instruction in case of missing BOS token.
        if prompt_token_ids[0] != self.token_ids["<s>"]:
            logger.warning("BOS token not found in prompt, "
                           "thus using empty string for instruction. "
                           "GritLM requires BOS token in prompt.")
            return instruction_len

        # If user pattern is found in the prompt, that means there should be
        # a newline token before the embed pattern.
        embed_pattern_ids = self.embed_pattern_ids
        if self._find_array(prompt_token_ids,
                            self.user_pattern_ids,
                            start_idx=1) == 1:
            embed_pattern_ids = self.embed_newline_pattern_ids

        # Find the embed pattern in the prompt.
        found_embed_pattern_idx = self._find_array(prompt_token_ids,
                                                   embed_pattern_ids,
                                                   start_idx=1)

        if found_embed_pattern_idx != -1:
            instruction_len = found_embed_pattern_idx + len(embed_pattern_ids)
        else:
            logger.warning("Query instruction not found in prompt, "
                           "thus using BOS token as instruction instead. "
                           "GritLM requires query instruction in prompt.")
            instruction_len = 1

        return instruction_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """
        Pool the hidden states by summing the embeddings of
        non-instruction tokens.
        """
        prompts_token_ids = [
            token_ids.prompt_token_ids_array
            for _, token_ids in pooling_metadata.seq_data.items()
        ]

        instruction_lens = torch.tensor(
            [
                self._get_instruction_len(prompt_token_ids)
                for prompt_token_ids in prompts_token_ids
            ],
            device=hidden_states.device,
        )

        prompt_lens = PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

        mask = torch.zeros_like(hidden_states, dtype=torch.bool)

        start_idx = 0
        for prompt_len, instruction_len in zip(prompt_lens, instruction_lens):
            end_idx = start_idx + prompt_len
            mask[start_idx + instruction_len:end_idx] = True
            start_idx = end_idx

        masked_hidden_states = hidden_states.masked_fill(~mask, 0.0)

        sum_embeddings = torch.zeros(len(prompt_lens),
                                     hidden_states.size(1),
                                     device=hidden_states.device)

        start_idx = 0
        for i, prompt_len in enumerate(prompt_lens):
            end_idx = start_idx + prompt_len
            sum_embeddings[i] = masked_hidden_states[start_idx:end_idx].sum(
                dim=0)
            start_idx = end_idx

        num_non_instruction_tokens = prompt_lens - instruction_lens
        mean_embeddings = sum_embeddings / num_non_instruction_tokens.unsqueeze(
            1)

        pooled_data = self.head(mean_embeddings,
                                pooling_metadata=pooling_metadata)

        pooled_outputs = [
            PoolingSequenceGroupOutput(data) for data in pooled_data
        ]

        return PoolerOutput(outputs=pooled_outputs)


class GritLM(LlamaForCausalLM, SupportsV0Only):
    """This class implements the embedding model for parasail-ai/GritLM-7B-aphrodite.

    The class inherits from LlamaForCausalLM and provides a custom pooling
    layer.

    The main difference between the pooling layer in GritLM and the one in
    LlamaForCausalLM is that GritLM ignores the query instruction in the prompt
    when pooling the hidden states.

    Embedding prompts should be in the following format:
    - With instruction: "<|user|>\nINSTRUCTION\n<|embed|>\nPROMPT".
    - Without instruction: "<|embed|>\nPROMPT".

    Generation prompts should be in the following format:
    - "<|user|>\nPROMPT\n<|assistant|>\n"
    """

    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(aphrodite_config=aphrodite_config, prefix=prefix, **kwargs)

        self.runner_type = aphrodite_config.model_config.runner_type

        self._pooler = GritLMPooler(aphrodite_config.model_config)

        for layer in self.model.layers:
            if self.runner_type == "pooling" and hasattr(layer, "self_attn"):
                assert isinstance(layer.self_attn.attn.impl, XFormersImpl), (
                    "GritLM embedding is only supported by XFormers backend, "
                    "which can be forced by APHRODITE_ATTENTION_BACKEND=XFORMERS")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        # Change attention to non-causal for pooling tasks.
        if self.runner_type == "pooling":
            attn_metadata = get_forward_context().attn_metadata
            assert attn_metadata.prefill_metadata.attn_bias is None
            attn_metadata.prefill_metadata.attn_bias = [
                BlockDiagonalMask.from_seqlens(attn_metadata.seq_lens)
            ]

        return super().forward(
            input_ids=input_ids,
            positions=positions,
            **kwargs,
        )

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)
