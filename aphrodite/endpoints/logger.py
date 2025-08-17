from collections.abc import Sequence
from typing import List, Optional, Union

import torch
from loguru import logger

from aphrodite.common.pooling_params import PoolingParams
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.lora.request import LoRARequest


class RequestLogger:

    def __init__(self, *, max_log_len: Optional[int]) -> None:

        self.max_log_len = max_log_len

    def log_inputs(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]],
        prompt_embeds: Optional[torch.Tensor],
        params: Optional[Union[SamplingParams, PoolingParams]],
        lora_request: Optional[LoRARequest],
    ) -> None:
        max_log_len = self.max_log_len
        if max_log_len is not None:
            if prompt is not None:
                prompt = prompt[:max_log_len]

            if prompt_token_ids is not None:
                prompt_token_ids = prompt_token_ids[:max_log_len]

        logger.info(f"Received request {request_id}: "
                    f"params: {params}, "
                    f"num_prompt_tokens: {len(prompt_token_ids)}, "
                    f"lora_request: {lora_request}, "
                    "prompt_embeds shape: {}",
                    prompt_embeds.shape if prompt_embeds is not None else None)

    def log_outputs(
        self,
        request_id: str,
        outputs: str,
        output_token_ids: Optional[Sequence[int]],
        finish_reason: Optional[str] = None,
        is_streaming: bool = False,
        delta: bool = False,
    ) -> None:
        max_log_len = self.max_log_len
        if max_log_len is not None:
            if outputs is not None:
                outputs = outputs[:max_log_len]

            if output_token_ids is not None:
                # Convert to list and apply truncation
                output_token_ids = list(output_token_ids)[:max_log_len]

        stream_info = ""
        if is_streaming:
            stream_info = (" (streaming delta)"
                           if delta else " (streaming complete)")

        logger.info(
            "Generated response {}: output: {}, "
            "output_token_ids: {}, finish_reason: {}",
            request_id,
            stream_info,
            outputs,
            output_token_ids,
            finish_reason,
        )
