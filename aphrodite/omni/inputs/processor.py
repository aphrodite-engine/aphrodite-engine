# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve Omni payloads across core input validation."""

from aphrodite.inputs import split_enc_dec_input
from aphrodite.omni.engine.serialization import serialize_additional_information
from aphrodite.omni.inputs.preprocess import OmniInputPreprocessor
from aphrodite.v1.engine.input_processor import InputProcessor


class OmniInputProcessor(InputProcessor):
    def __init__(self, aphrodite_config, **kwargs):
        super().__init__(aphrodite_config, **kwargs)
        self.input_preprocessor = OmniInputPreprocessor(aphrodite_config, self.renderer)

    def process_inputs(self, request_id, prompt, params, supported_tasks, **kwargs):
        if not isinstance(prompt, dict) or "type" not in prompt:
            prompt = self.input_preprocessor.preprocess(prompt, kwargs.get("tokenization_kwargs"))
        _, decoder = split_enc_dec_input(prompt)
        request = super().process_inputs(request_id, prompt, params, supported_tasks, **kwargs)
        request.additional_information = serialize_additional_information(decoder.get("additional_information"))
        request.model_intermediate_buffer = decoder.get("model_intermediate_buffer")
        if decoder.get("prompt_embeds") is not None:
            request.prompt_embeds = decoder["prompt_embeds"]
        return request
