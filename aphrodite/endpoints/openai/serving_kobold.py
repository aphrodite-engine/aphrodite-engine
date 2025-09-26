import json
from collections.abc import AsyncGenerator
from typing import Optional

from fastapi import Request

from aphrodite.common.sampling_params import _SAMPLING_EPS, SamplingParams
from aphrodite.endpoints.logger import RequestLogger
from aphrodite.endpoints.openai.protocol import KAIGenerationInputSchema
from aphrodite.endpoints.openai.serving_engine import OpenAIServing
from aphrodite.endpoints.openai.serving_models import OpenAIServingModels
from aphrodite.engine.protocol import EngineClient
from aphrodite.transformers_utils.tokenizer import AnyTokenizer
from aphrodite.utils import random_uuid


class OpenAIServingKobold(OpenAIServing):
    """Serving class for KoboldAI API compatibility."""

    def __init__(
        self,
        engine_client: EngineClient,
        model_config,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
            **kwargs
        )

    async def create_kobold_stream(
        self,
        request: KAIGenerationInputSchema,
        raw_request: Optional[Request] = None,
    ) -> AsyncGenerator[str, None]:
        """Create streaming response for KoboldAI API."""

        tokenizer = await self.engine_client.get_tokenizer()
        sampling_params, input_tokens = self._prepare_engine_payload(
            request, tokenizer)

        results_generator = self.engine_client.generate(
            {
                "prompt": request.prompt,
                "prompt_token_ids": input_tokens,
            },
            sampling_params,
            request.genkey,
        )

        previous_output = ""
        async for res in results_generator:
            new_chunk = res.outputs[0].text[len(previous_output):]
            previous_output += new_chunk
            yield (
                f"event: message\ndata: {json.dumps({'token': new_chunk})}\n\n"
            )

    def _prepare_engine_payload(
        self,
        kai_payload: KAIGenerationInputSchema,
        tokenizer: AnyTokenizer,
    ) -> tuple[SamplingParams, list[int]]:
        """Create SamplingParams and truncated input tokens for AsyncEngine"""

        if not kai_payload.genkey:
            kai_payload.genkey = f"kai-{random_uuid()}"

        kai_payload.top_k = (
            kai_payload.top_k if kai_payload.top_k != 0.0 else -1)
        kai_payload.tfs = max(_SAMPLING_EPS, kai_payload.tfs or 0.0)
        if (kai_payload.temperature or 0.0) < _SAMPLING_EPS:
            kai_payload.n = 1
            kai_payload.top_p = 1.0
            kai_payload.top_k = -1

        sampling_params = SamplingParams(
            n=kai_payload.n or 1,
            best_of=kai_payload.n or 1,
            repetition_penalty=kai_payload.rep_pen or 1.0,
            temperature=kai_payload.temperature or 1.0,
            smoothing_factor=kai_payload.smoothing_factor or 0.0,
            smoothing_curve=kai_payload.smoothing_curve or 1.0,
            tfs=kai_payload.tfs or 1.0,
            top_p=kai_payload.top_p or 1.0,
            top_k=kai_payload.top_k or -1,
            top_a=kai_payload.top_a or 0.0,
            min_p=kai_payload.min_p or 0.0,
            typical_p=kai_payload.typical or 1.0,
            eta_cutoff=kai_payload.eta_cutoff or 0.0,
            epsilon_cutoff=kai_payload.eps_cutoff or 0.0,
            stop=kai_payload.stop_sequence,
            include_stop_str_in_output=
            kai_payload.include_stop_str_in_output or False,
            max_tokens=kai_payload.max_length,
            seed=kai_payload.sampler_seed,
            xtc_probability=kai_payload.xtc_probability or 0.0,
            xtc_threshold=kai_payload.xtc_threshold or 0.0
        )

        max_input_tokens = max(
            1, kai_payload.max_context_length - kai_payload.max_length)
        input_tokens = (
            tokenizer(kai_payload.prompt).input_ids[-max_input_tokens:])

        return sampling_params, input_tokens
