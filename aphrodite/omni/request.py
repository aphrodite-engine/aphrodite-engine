# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from aphrodite.multimodal.inputs import MultiModalFeatureSpec
from aphrodite.sampling_params import SamplingParams
from aphrodite.v1.engine import EngineCoreRequest
from aphrodite.v1.request import Request

if TYPE_CHECKING:
    from aphrodite.v1.core.kv_cache_utils import BlockHash

from aphrodite.omni.engine import AdditionalInformationPayload, PromptEmbedsPayload


class OmniRequest(Request):
    """Request class for omni models, extending the base Request.

    This class extends the base Sonar Request with support for prompt
    embeddings and additional information payloads, enabling direct
    transfer of pre-computed embeddings between stages.

    Args:
        prompt_embeds: Optional serialized prompt embeddings payload.
            Used for direct transfer of embeddings between stages.
        additional_information: Optional additional information payload
            containing tensors or lists to be passed along with the request.
    """

    def __init__(
        self,
        *args,
        prompt_embeds: PromptEmbedsPayload | torch.Tensor | None = None,
        # Optional external request ID for tracking
        external_req_id: str | None = None,
        additional_information: AdditionalInformationPayload | dict[str, Any] | None = None,
        model_intermediate_buffer: dict | None = None,
        **kwargs,
    ):
        if prompt_embeds is not None:
            kwargs["prompt_embeds"] = self._maybe_decode_prompt_embeds(prompt_embeds)
        super().__init__(*args, **kwargs)
        # Sonar 0.27 owns this counter; accelerator images still based on 0.26
        # do not. Keep the Omni scheduler's stale-output drain compatible with
        # both request layouts until those images move to 0.27.
        self.num_stale_output_tokens = int(getattr(self, "num_stale_output_tokens", 0) or 0)
        # Preserve serialized prompt embeddings payload (optional)
        self.prompt_embeds_payload: PromptEmbedsPayload | None = (
            prompt_embeds if isinstance(prompt_embeds, PromptEmbedsPayload) else None
        )
        # Optional external request ID for tracking
        self.external_req_id: str | None = external_req_id
        # Connectors decode the transport payload before scheduler reuse.
        self.additional_information: AdditionalInformationPayload | dict[str, Any] | None = additional_information
        self._omni_segment_generation: int = 0
        # Runner-owned runtime payload.
        self.model_intermediate_buffer: dict | None = model_intermediate_buffer

    @staticmethod
    def _maybe_decode_prompt_embeds(
        prompt_embeds: PromptEmbedsPayload | torch.Tensor | None,
    ) -> torch.Tensor | None:
        if isinstance(prompt_embeds, PromptEmbedsPayload):
            dtype = getattr(np, prompt_embeds.dtype)
            arr = np.frombuffer(prompt_embeds.data, dtype=dtype)
            arr = arr.reshape(prompt_embeds.shape)
            return torch.from_numpy(arr)
        return prompt_embeds

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        """Create an OmniRequest from an OmniEngineCoreRequest.

        Args:
            request: The OmniEngineCoreRequest to convert
            block_hasher: Optional function to compute block hashes for
                prefix caching

        Returns:
            OmniRequest instance created from the engine core request
        """
        return cls(
            request_id=request.request_id,
            # Optional external request ID for tracking
            external_req_id=request.external_req_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            additional_information=getattr(request, "additional_information", None),
            model_intermediate_buffer=getattr(request, "model_intermediate_buffer", None),
            resumable=request.resumable,
            session_id=request.session_id,
            reasoning_ended=request.reasoning_ended,
            reasoning_parser_kwargs=request.reasoning_parser_kwargs,
            abort_immediately=request.abort_immediately,
        )


@dataclass
class OmniStreamingUpdate:
    """
    Override: add additional information
    Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None
    additional_information: AdditionalInformationPayload | dict[str, Any] | None = None
    model_intermediate_buffer: dict | None = None

    @classmethod
    def from_request(cls, request: "Request") -> "OmniStreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
            additional_information=getattr(request, "additional_information", None),
            model_intermediate_buffer=getattr(request, "model_intermediate_buffer", None),
        )
