# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt Omni's prompt hooks to Sonar's renderer-based preprocessing."""

from aphrodite.inputs import build_enc_dec_input
from aphrodite.renderers.inputs.preprocess import parse_dec_only_prompt, parse_enc_dec_prompt


class InputPreprocessor:
    def __init__(self, aphrodite_config, renderer):
        self.model_config = aphrodite_config.model_config
        self.renderer = renderer

    def _tokenize_prompt(self, prompt, tokenization_kwargs=None):
        return self._tokenize({"prompt": prompt}, tokenization_kwargs)

    def _truncate_inputs(self, tokens, tokenization_kwargs=None):
        return self._tokenize({"prompt_token_ids": tokens}, tokenization_kwargs)

    def _tokenize(self, prompt, kwargs):
        params = self.renderer.default_cmpl_tok_params.with_kwargs(**(kwargs or {}))
        return self.renderer._tokenize_singleton_prompt(prompt, params)["prompt_token_ids"]

    def _process_multimodal(
        self, prompt, mm_data, mm_processor_kwargs=None, tokenization_kwargs=None, *, mm_uuids=None
    ):
        return self.renderer._process_multimodal(
            prompt,
            mm_data,
            mm_uuids=mm_uuids,
            mm_processor_kwargs=mm_processor_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

    def _process_embeds(self, prompt):
        return self.renderer._process_embeds(prompt)

    def preprocess(self, prompt, tokenization_kwargs=None):
        if self.model_config.is_encoder_decoder:
            parsed = parse_enc_dec_prompt(prompt)
            return build_enc_dec_input(
                self._prompt_to_llm_inputs(parsed["encoder_prompt"], tokenization_kwargs),
                self._prompt_to_llm_inputs(parsed["decoder_prompt"], tokenization_kwargs)
                if parsed["decoder_prompt"] is not None
                else None,
                decoder_start_token_id=self.renderer.get_dec_start_token_id(),
                skip_decoder_start_token=self.renderer._get_skip_decoder_start_token(),
            )
        return self._prompt_to_llm_inputs(parse_dec_only_prompt(prompt), tokenization_kwargs)
