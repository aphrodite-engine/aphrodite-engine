# Adapted from openai/api_server.py and tgi-kai-bridge

import argparse
import asyncio
import json
import os
from http import HTTPStatus
from typing import AsyncGenerator, List, Tuple

import fastapi
import uvicorn
from fastapi import APIRouter, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from loguru import logger
from prometheus_client import make_asgi_app

from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import _SAMPLING_EPS, SamplingParams
from aphrodite.utils import random_uuid
from aphrodite.endpoints.kobold.protocol import KAIGenerationInputSchema
from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.transformers_utils.tokenizer import get_tokenizer

TIMEOUT_KEEP_ALIVE = 5  # seconds

served_model: str = "Read Only"
engine: AsyncAphrodite = None
gen_cache: dict = {}
app = fastapi.FastAPI()

badwordsids: List[int] = []

# Add prometheus asgi middleware to route /metrics/ requests
metrics_app = make_asgi_app()
app.mount("/metrics/", metrics_app)


def _set_badwords(tokenizer, hf_config):  # pylint: disable=redefined-outer-name
    global badwordsids
    if hf_config.bad_words_ids is not None:
        badwordsids = hf_config.bad_words_ids
        return

    badwordsids = [
        v for k, v in tokenizer.get_vocab().items()
        if any(c in str(k) for c in "[]")
    ]
    if tokenizer.pad_token_id in badwordsids:
        badwordsids.remove(tokenizer.pad_token_id)
    badwordsids.append(tokenizer.eos_token_id)


kai_api = APIRouter()
extra_api = APIRouter()
kobold_lite_ui = ""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse({
        "msg": message,
        "type": "invalid_request_error"
    },
                        status_code=status_code.value)


@app.exception_handler(ValueError)
def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return create_error_response(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))


def prepare_engine_payload(
        kai_payload: KAIGenerationInputSchema
) -> Tuple[SamplingParams, List[int]]:
    """Create SamplingParams and truncated input tokens for AsyncEngine"""

    if not kai_payload.genkey:
        kai_payload.genkey = f"kai-{random_uuid()}"

    if kai_payload.max_context_length > max_model_len:
        raise ValueError(
            f"max_context_length ({kai_payload.max_context_length}) "
            "must be less than or equal to "
            f"max_model_len ({max_model_len})")

    # KAIspec: top_k == 0 means disabled, aphrodite: top_k == -1 means disabled
    # https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings
    kai_payload.top_k = kai_payload.top_k if kai_payload.top_k != 0.0 else -1
    kai_payload.tfs = max(_SAMPLING_EPS, kai_payload.tfs)
    if kai_payload.temperature < _SAMPLING_EPS:
        # temp < _SAMPLING_EPS: greedy sampling
        kai_payload.n = 1
        kai_payload.top_p = 1.0
        kai_payload.top_k = -1

    if kai_payload.dynatemp_range is not None:
        dynatemp_min = kai_payload.temperature - kai_payload.dynatemp_range
        dynatemp_max = kai_payload.temperature + kai_payload.dynatemp_range

    sampling_params = SamplingParams(
        n=kai_payload.n,
        best_of=kai_payload.n,
        repetition_penalty=kai_payload.rep_pen,
        temperature=kai_payload.temperature,
        dynatemp_min=dynatemp_min if kai_payload.dynatemp_range > 0 else 0.0,
        dynatemp_max=dynatemp_max if kai_payload.dynatemp_range > 0 else 0.0,
        dynatemp_exponent=kai_payload.dynatemp_exponent,
        smoothing_factor=kai_payload.smoothing_factor,
        smoothing_curve=kai_payload.smoothing_curve,
        tfs=kai_payload.tfs,
        top_p=kai_payload.top_p,
        top_k=kai_payload.top_k,
        top_a=kai_payload.top_a,
        min_p=kai_payload.min_p,
        typical_p=kai_payload.typical,
        eta_cutoff=kai_payload.eta_cutoff,
        epsilon_cutoff=kai_payload.eps_cutoff,
        mirostat_mode=kai_payload.mirostat,
        mirostat_tau=kai_payload.mirostat_tau,
        mirostat_eta=kai_payload.mirostat_eta,
        stop=kai_payload.stop_sequence,
        include_stop_str_in_output=kai_payload.include_stop_str_in_output,
        custom_token_bans=badwordsids
        if kai_payload.use_default_badwordsids else [],
        max_tokens=kai_payload.max_length,
        seed=kai_payload.sampler_seed,
    )

    max_input_tokens = max(
        1, kai_payload.max_context_length - kai_payload.max_length)
    input_tokens = tokenizer(kai_payload.prompt).input_ids[-max_input_tokens:]

    return sampling_params, input_tokens


@kai_api.post("/generate")
async def generate(kai_payload: KAIGenerationInputSchema) -> JSONResponse:
    """Generate text"""

    sampling_params, input_tokens = prepare_engine_payload(kai_payload)
    result_generator = engine.generate(None, sampling_params,
                                       kai_payload.genkey, input_tokens)

    final_res: RequestOutput = None
    previous_output = ""
    async for res in result_generator:
        final_res = res
        new_chunk = res.outputs[0].text[len(previous_output):]
        previous_output += new_chunk
        gen_cache[kai_payload.genkey] = previous_output

    assert final_res is not None
    del gen_cache[kai_payload.genkey]

    return JSONResponse(
        {"results": [{
            "text": output.text
        } for output in final_res.outputs]})


@extra_api.post("/generate/stream")
async def generate_stream(
        kai_payload: KAIGenerationInputSchema) -> StreamingResponse:
    """Generate text SSE streaming"""

    sampling_params, input_tokens = prepare_engine_payload(kai_payload)
    results_generator = engine.generate(None, sampling_params,
                                        kai_payload.genkey, input_tokens)

    async def stream_kobold() -> AsyncGenerator[bytes, None]:
        previous_output = ""
        async for res in results_generator:
            new_chunk = res.outputs[0].text[len(previous_output):]
            previous_output += new_chunk
            yield b"event: message\n"
            yield f"data: {json.dumps({'token': new_chunk})}\n\n".encode()

    return StreamingResponse(stream_kobold(),
                             headers={
                                 "Cache-Control": "no-cache",
                                 "Connection": "keep-alive"
                             },
                             media_type="text/event-stream")


@extra_api.post("/generate/check")
@extra_api.get("/generate/check")
async def check_generation(request: Request):
    """Check outputs in progress (poll streaming)"""

    text = ""
    try:
        request_dict = await request.json()
        if "genkey" in request_dict and request_dict["genkey"] in gen_cache:
            text = gen_cache[request_dict["genkey"]]
    except json.JSONDecodeError:
        pass

    return JSONResponse({"results": [{"text": text}]})


@extra_api.post("/abort")
async def abort_generation(request: Request):
    """Abort running generation"""
    try:
        request_dict = await request.json()
        if "genkey" in request_dict:
            await engine.abort(request_dict["genkey"])
    except json.JSONDecodeError:
        pass

    return JSONResponse({})


@extra_api.post("/tokencount")
async def count_tokens(request: Request):
    """Tokenize string and return token count"""

    request_dict = await request.json()
    tokenizer_result = tokenizer(request_dict["prompt"])
    return JSONResponse({"value": len(tokenizer_result.input_ids)})


@kai_api.get("/info/version")
async def get_version():
    """Impersonate KAI"""
    return JSONResponse({"result": "1.2.4"})


@kai_api.get("/model")
async def get_model():
    """Get current model"""
    return JSONResponse({"result": f"aphrodite/{served_model}"})


@kai_api.get("/config/soft_prompts_list")
async def get_available_softprompts():
    """Stub for compatibility"""
    return JSONResponse({"values": []})


@kai_api.get("/config/soft_prompt")
async def get_current_softprompt():
    """Stub for compatibility"""
    return JSONResponse({"value": ""})


@kai_api.put("/config/soft_prompt")
async def set_current_softprompt():
    """Stub for compatibility"""
    return JSONResponse({})


@kai_api.get("/config/max_length")
async def get_max_length() -> JSONResponse:
    """Return the configured max output length"""
    max_length = args.max_length
    return JSONResponse({"value": max_length})


@kai_api.get("/config/max_context_length")
@extra_api.get("/true_max_context_length")
async def get_max_context_length() -> JSONResponse:
    """Return the max context length based on the EngineArgs configuration."""
    max_context_length = engine_model_config.max_model_len
    return JSONResponse({"value": max_context_length})


@extra_api.get("/preloadstory")
async def get_preloaded_story() -> JSONResponse:
    """Stub for compatibility"""
    return JSONResponse({})


@extra_api.get("/version")
async def get_extra_version():
    """Impersonate KoboldCpp"""
    return JSONResponse({"result": "KoboldCpp", "version": "1.55.1"})


@app.get("/")
async def get_kobold_lite_ui():
    """Serves a cached copy of the Kobold Lite UI, loading it from disk on
    demand if needed."""
    # read and return embedded kobold lite
    global kobold_lite_ui
    if kobold_lite_ui == "":
        scriptpath = os.path.dirname(os.path.abspath(__file__))
        klitepath = os.path.join(scriptpath, "klite.embd")
        if os.path.exists(klitepath):
            with open(klitepath, "r") as f:
                kobold_lite_ui = f.read()
        else:
            print("Embedded Kobold Lite not found")
    return HTMLResponse(content=kobold_lite_ui)


@app.get("/health")
async def health() -> Response:
    """Health check route for K8s"""
    return Response(status_code=200)


app.include_router(kai_api, prefix="/api/v1")
app.include_router(kai_api, prefix="/api/latest", include_in_schema=False)
app.include_router(extra_api, prefix="/api/extra")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aphrodite KoboldAI-Compatible RESTful API server.")
    parser.add_argument("--host",
                        type=str,
                        default="localhost",
                        help="host name")
    parser.add_argument("--port", type=int, default=2242, help="port number")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")
    parser.add_argument("--max-length",
                        type=int,
                        default=256,
                        help="The maximum length of the generated text. "
                        "For use with Kobold Horde.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    logger.debug(f"args: {args}")
    logger.warning("The standalone Kobold API is deprecated and will not "
                   "receive updates. Please use the OpenAI API with the "
                   "--launch-kobold-api flag instead.")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncAphrodite.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())
    max_model_len = engine_model_config.max_model_len

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(engine_args.tokenizer,
                              tokenizer_mode=engine_args.tokenizer_mode,
                              trust_remote_code=engine_args.trust_remote_code)

    _set_badwords(tokenizer, engine_model_config.hf_config)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
