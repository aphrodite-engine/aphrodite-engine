from typing import List, Optional, Union

import torch
from loguru import logger

from aphrodite.common.pooling_params import PoolingParams
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.lora.request import LoRARequest


class RequestLogger:

    def __init__(self, *, max_log_len: Optional[int]) -> None:
        super().__init__()

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
                    f"prompt: {prompt}, "
                    f"params: {params}, "
                    f"prompt_token_ids: {prompt_token_ids}, "
                    f"lora_request: {lora_request}, "
                    "prompt_embeds shape: {}",
                    prompt_embeds.shape if prompt_embeds is not None else None)
