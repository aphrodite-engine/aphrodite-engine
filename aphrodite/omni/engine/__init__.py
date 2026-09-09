# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Engine components for Sonar Omni.
"""

from typing import Any

import msgspec
import torch

from aphrodite.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
)


class PromptEmbedsPayload(msgspec.Struct):
    """Serialized prompt embeddings payload for direct transfer.

    data: raw bytes of the tensor in row-major order
    shape: [seq_len, hidden_size]
    dtype: torch dtype name (e.g., "float16", "float32")
    """

    data: bytes
    shape: list[int]
    dtype: str


class AdditionalInformationEntry(msgspec.Struct):
    """One entry of additional_information.

    Three supported forms are encoded:
      - tensor: data/shape/dtype
      - list: a Python list (msgspec-serializable)
      - scalar: a Python scalar (msgspec-serializable)
    Exactly one of (tensor_data, list_data, scalar_data) should be non-None.
    """

    # Tensor form
    tensor_data: bytes | None = None
    tensor_shape: list[int] | None = None
    tensor_dtype: str | None = None

    # List form
    list_data: list[Any] | None = None

    # Scalar form
    scalar_data: Any | None = None


class AdditionalInformationPayload(msgspec.Struct):
    """Serialized dictionary payload for additional_information.

    Keys are strings; values are encoded as AdditionalInformationEntry.
    """

    entries: dict[str, AdditionalInformationEntry]


class OmniEngineCoreRequest(EngineCoreRequest):
    """Engine core request for omni models with embeddings support.

    Extends the base EngineCoreRequest with support for additional
    information payloads, enabling direct transfer of pre-computed data
    between pipeline stages.

    Note: prompt_embeds is inherited from EngineCoreRequest
    (torch.Tensor | None). PromptEmbedsPayload should be decoded to
    torch.Tensor before constructing this request.

    Attributes:
        additional_information: Optional serialized additional information
            dictionary containing tensors or lists to pass along with the request
    """

    # Optional additional information dictionary (serialized)
    additional_information: AdditionalInformationPayload | None = None
    # Runner-owned runtime payload. This is materialized directly into
    # GPUModelRunner.model_intermediate_buffer instead of using the deprecated
    # additional_information request transport.
    model_intermediate_buffer: dict[str, Any] | None = None

    @classmethod
    def from_request(
        cls,
        request: EngineCoreRequest,
        *,
        prompt_embeds: torch.Tensor | None = None,
        additional_information: AdditionalInformationPayload | None = None,
        model_intermediate_buffer: dict[str, Any] | None = None,
    ) -> "OmniEngineCoreRequest":
        """Clone an EngineCoreRequest into an OmniEngineCoreRequest with optional payload overrides."""

        if prompt_embeds is None:
            prompt_embeds = request.prompt_embeds
        if additional_information is None:
            additional_information = getattr(request, "additional_information", None)
        if model_intermediate_buffer is None:
            model_intermediate_buffer = getattr(request, "model_intermediate_buffer", None)

        fields = {field.name: getattr(request, field.name) for field in msgspec.structs.fields(EngineCoreRequest)}
        fields.update(
            prompt_embeds=prompt_embeds,
            additional_information=additional_information,
            model_intermediate_buffer=model_intermediate_buffer,
        )
        return cls(**fields)


class OmniEngineCoreOutput(EngineCoreOutput):
    # Dedicated channel for multimodal outputs (image/audio/latent).
    # pooling_output is inherited from EngineCoreOutput as torch.Tensor | None
    # and retains its original Sonar semantics for pooling/embedding tasks.
    multimodal_output: dict[str, torch.Tensor] | None = None
    # Finished flag for streaming input segment
    is_segment_finished: bool | None = False
    # Streaming update prompt length
    new_prompt_len_snapshot: int | None = None


class OmniEngineCoreOutputs(EngineCoreOutputs[OmniEngineCoreOutput]):
    outputs: list[OmniEngineCoreOutput] = []
