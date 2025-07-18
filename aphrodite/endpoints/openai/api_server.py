import asyncio
import atexit
import gc
import importlib
import inspect
import json
import multiprocessing
import os
import pickle
import re
import signal
import socket
import tempfile
import uuid
from argparse import Namespace
from contextlib import asynccontextmanager
from distutils.util import strtobool
from functools import partial
from http import HTTPStatus
from typing import Annotated, AsyncGenerator, AsyncIterator, Optional, Union

import uvloop
import yaml
from fastapi import (APIRouter, Depends, FastAPI, Form, HTTPException, Request,
                     UploadFile)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (HTMLResponse, JSONResponse, Response,
                               StreamingResponse)
from loguru import logger
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import State
from starlette.routing import Mount
from typing_extensions import assert_never

import aphrodite.common.envs as envs
from aphrodite.common.config import AphroditeConfig
from aphrodite.common.logger import log_once
from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import _SAMPLING_EPS, SamplingParams
from aphrodite.common.utils import (Device, FlexibleArgumentParser,
                                    get_open_zmq_ipc_path,
                                    is_valid_ipv6_address, random_uuid,
                                    set_ulimit)
from aphrodite.endpoints.chat_utils import (load_chat_template,
                                            resolve_hf_chat_template,
                                            resolve_mistral_chat_template)
from aphrodite.endpoints.logger import RequestLogger
from aphrodite.endpoints.openai.args import (make_arg_parser,
                                             validate_parsed_serve_args)
from aphrodite.endpoints.openai.protocol import (ChatCompletionRequest,
                                                 ChatCompletionResponse,
                                                 CompletionRequest,
                                                 CompletionResponse,
                                                 DetokenizeRequest,
                                                 DetokenizeResponse,
                                                 EmbeddingChatRequest,
                                                 EmbeddingCompletionRequest,
                                                 EmbeddingRequest,
                                                 EmbeddingResponse,
                                                 EmbeddingResponseData,
                                                 ErrorResponse,
                                                 KAIGenerationInputSchema,
                                                 LoadLoRAAdapterRequest,
                                                 PoolingChatRequest,
                                                 PoolingCompletionRequest,
                                                 PoolingRequest,
                                                 PoolingResponse,
                                                 RerankRequest, RerankResponse,
                                                 ScoreRequest, ScoreResponse,
                                                 TokenizeRequest,
                                                 TokenizeResponse,
                                                 TranscriptionRequest,
                                                 TranscriptionResponse,
                                                 UnloadLoRAAdapterRequest)
from aphrodite.endpoints.openai.serving_chat import OpenAIServingChat
from aphrodite.endpoints.openai.serving_completions import (
    OpenAIServingCompletion)
from aphrodite.endpoints.openai.serving_embedding import OpenAIServingEmbedding
from aphrodite.endpoints.openai.serving_engine import OpenAIServing
from aphrodite.endpoints.openai.serving_models import (BaseModelPath,
                                                       OpenAIServingModels)
from aphrodite.endpoints.openai.serving_pooling import OpenAIServingPooling
from aphrodite.endpoints.openai.serving_score import ServingScores
from aphrodite.endpoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from aphrodite.endpoints.openai.serving_transcription import (
    OpenAIServingTranscription)
from aphrodite.endpoints.openai.tool_parsers import ToolParserManager
from aphrodite.endpoints.utils import (cli_env_setup, load_aware_call,
                                       with_cancellation)
from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.engine.multiprocessing import (APHRODITE_RPC_SUCCESS_STR,
                                              RPCShutdownRequest)
from aphrodite.engine.multiprocessing.client import MQAphroditeEngineClient
from aphrodite.engine.multiprocessing.engine import run_mp_engine
from aphrodite.engine.protocol import EngineClient
from aphrodite.modeling.model_loader.weight_utils import get_model_config_yaml
from aphrodite.reasoning import ReasoningParserManager
from aphrodite.server import serve_http
from aphrodite.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from aphrodite.transformers_utils.tokenizer import MistralTokenizer
from aphrodite.usage.usage_lib import UsageContext
from aphrodite.version import __version__ as APHRODITE_VERSION

TIMEOUT_KEEP_ALIVE = 5  # seconds
SERVE_KOBOLD_LITE_UI = strtobool(os.getenv("SERVE_KOBOLD_LITE_UI", "1"))

router = APIRouter()
kai_api = APIRouter()
extra_api = APIRouter()
kobold_lite_ui = ""
sampler_json = ""
gen_cache: dict = {}
prometheus_multiproc_dir: tempfile.TemporaryDirectory

_running_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if app.state.log_stats:
            engine_client: EngineClient = app.state.engine_client

            async def _force_log():
                while True:
                    await asyncio.sleep(10.)
                    await engine_client.do_log_stats()

            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove)
        else:
            task = None

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        gc.collect()
        gc.freeze()
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
    finally:
        # Ensure app state including engine ref is gc'd
        del app.state


@asynccontextmanager
async def build_async_engine_client(
        args: Namespace) -> AsyncIterator[EngineClient]:

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)

    async with build_async_engine_client_from_engine_args(
            engine_args, args.disable_frontend_multiprocessing) as engine:
        yield engine


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Create the EngineConfig (determines if we can use V1).
    usage_context = UsageContext.OPENAI_API_SERVER
    aphrodite_config = engine_args.create_engine_config(
        usage_context=usage_context)

    # V1 AsyncLLM.
    if envs.APHRODITE_USE_V1:
        if disable_frontend_multiprocessing:
            logger.warning(
                "V1 is enabled, but got --disable-frontend-multiprocessing. "
                "To disable frontend multiprocessing, set APHRODITE_USE_V1=0.")

        from aphrodite.v1.engine.async_llm import AsyncLLM
        async_llm: Optional[AsyncLLM] = None
        try:
            async_llm = AsyncLLM.from_aphrodite_config(
                aphrodite_config=aphrodite_config,
                usage_context=usage_context,
                disable_log_requests=engine_args.disable_log_requests,
                disable_log_stats=engine_args.disable_log_stats)
            yield async_llm
        finally:
            if async_llm:
                async_llm.shutdown()

    # V0 AsyncLLM.
    elif (MQAphroditeEngineClient.is_unsupported_config(aphrodite_config)
          or disable_frontend_multiprocessing):

        engine_client: Optional[EngineClient] = None
        try:
            engine_client = AsyncAphrodite.from_aphrodite_config(
                aphrodite_config=aphrodite_config,
                usage_context=usage_context,
                disable_log_requests=engine_args.disable_log_requests,
                disable_log_stats=engine_args.disable_log_stats)
            yield engine_client
        finally:
            if engine_client and hasattr(engine_client, "shutdown"):
                engine_client.shutdown()

    # V0MQLLMEngine.
    else:
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            # Make TemporaryDirectory for prometheus multiprocessing
            # Note: global TemporaryDirectory will be automatically
            #   cleaned up upon exit.
            global prometheus_multiproc_dir
            prometheus_multiproc_dir = tempfile.TemporaryDirectory()
            os.environ[
                "PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
        else:
            logger.warning(
                "Found PROMETHEUS_MULTIPROC_DIR was set by user. "
                "This directory must be wiped between vLLM runs or "
                "you will find inaccurate metrics. Unset the variable "
                "and vLLM will properly handle cleanup.")

        # Select random path for IPC.
        ipc_path = get_open_zmq_ipc_path()
        logger.debug("Multiprocessing frontend to use {} for IPC Path.",
                     ipc_path)

        # Start RPCServer in separate process (holds the LLMEngine).
        # the current process might have CUDA context,
        # so we need to spawn a new process
        context = multiprocessing.get_context("spawn")

        # Ensure we can serialize transformer config before spawning
        maybe_register_config_serialize_by_value()

        # The Process can raise an exception during startup, which may
        # not actually result in an exitcode being reported. As a result
        # we use a shared variable to communicate the information.
        engine_alive = multiprocessing.Value('b', True, lock=False)
        engine_process = context.Process(
            target=run_mp_engine,
            args=(aphrodite_config, UsageContext.OPENAI_API_SERVER, ipc_path,
                  engine_args.disable_log_stats,
                  engine_args.disable_log_requests, engine_alive))
        engine_process.start()
        engine_pid = engine_process.pid
        assert engine_pid is not None, "Engine process failed to start."
        logger.info("Started engine process with PID {}", engine_pid)

        def _cleanup_ipc_path():
            socket_path = ipc_path.replace("ipc://", "")
            if os.path.exists(socket_path):
                os.remove(socket_path)

        # Ensure we clean up the local IPC socket file on exit.
        atexit.register(_cleanup_ipc_path)

        # Build RPCClient, which conforms to EngineClient Protocol.
        build_client = partial(MQAphroditeEngineClient, ipc_path,
                               aphrodite_config, engine_pid)
        mq_engine_client = await asyncio.get_running_loop().run_in_executor(
            None, build_client)
        try:
            while True:
                try:
                    await mq_engine_client.setup()
                    break
                except TimeoutError:
                    if (not engine_process.is_alive()
                            or not engine_alive.value):
                        raise RuntimeError(
                            "Engine process failed to start. See stack "
                            "trace for the root cause.") from None

            yield mq_engine_client  # type: ignore[misc]
        finally:
            # Ensure rpc server process was terminated
            engine_process.terminate()

            # Close all open connections to the backend
            mq_engine_client.close()

            # Wait for engine process to join
            engine_process.join(4)
            if engine_process.exitcode is None:
                # Kill if taking longer than 5 seconds to stop
                engine_process.kill()

            # Lazy import for prometheus multiprocessing.
            # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
            # before prometheus_client is imported.
            # See https://prometheus.github.io/client_python/multiprocess/
            from prometheus_client import multiprocess
            multiprocess.mark_process_dead(engine_process.pid)


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise HTTPException(
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported Media Type: Only 'application/json' is allowed"
        )


def mount_metrics(app: FastAPI):
    # Lazy import for prometheus multiprocessing.
    # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
    # before prometheus_client is imported.
    # See https://prometheus.github.io/client_python/multiprocess/
    from prometheus_client import (REGISTRY, CollectorRegistry, make_asgi_app,
                                   multiprocess)
    from prometheus_fastapi_instrumentator import Instrumentator

    registry = REGISTRY

    prometheus_multiproc_dir_path = os.getenv("PROMETHEUS_MULTIPROC_DIR", None)
    if prometheus_multiproc_dir_path is not None:
        logger.debug("Aphrodite to use {} as PROMETHEUS_MULTIPROC_DIR",
                     prometheus_multiproc_dir_path)
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)

    Instrumentator(
        excluded_handlers=[
            "/metrics",
            "/health",
            "/load",
            "/ping",
            "/version",
            "/server_info",
        ],
        registry=registry,
    ).add().instrument(app).expose(app)

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


def base(request: Request) -> OpenAIServing:
    # Reuse the existing instance
    return tokenization(request)


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


def chat(request: Request) -> Optional[OpenAIServingChat]:
    return request.app.state.openai_serving_chat


def completion(request: Request) -> Optional[OpenAIServingCompletion]:
    return request.app.state.openai_serving_completion


def pooling(request: Request) -> Optional[OpenAIServingPooling]:
    return request.app.state.openai_serving_pooling


def embedding(request: Request) -> Optional[OpenAIServingEmbedding]:
    return request.app.state.openai_serving_embedding


def score(request: Request) -> Optional[ServingScores]:
    return request.app.state.openai_serving_scores


def rerank(request: Request) -> Optional[ServingScores]:
    return request.app.state.openai_serving_scores


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


def transcription(request: Request) -> OpenAIServingTranscription:
    return request.app.state.openai_serving_transcription


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health")
async def health(raw_request: Request) -> Response:
    """Health check."""
    await engine_client(raw_request).check_health()
    return Response(status_code=200)


@router.get("/load")
async def get_server_load_metrics(request: Request):
    # This endpoint returns the current server load metrics.
    # It tracks requests utilizing the GPU from the following routes:
    # - /v1/chat/completions
    # - /v1/completions
    # - /v1/audio/transcriptions
    # - /v1/embeddings
    # - /pooling
    # - /score
    # - /v1/score
    # - /rerank
    # - /v1/rerank
    # - /v2/rerank
    return JSONResponse(
        content={'server_load': request.app.state.server_load_metrics})


@router.api_route("/ping", methods=["GET", "POST"])
async def ping(raw_request: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    return await health(raw_request)


@router.post("/v1/tokenize", dependencies=[Depends(validate_json_request)])
@with_cancellation
async def tokenize(request: TokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)

    generator = await handler.create_tokenize(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, TokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/v1/detokenize", dependencies=[Depends(validate_json_request)])
@with_cancellation
async def detokenize(request: DetokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)

    generator = await handler.create_detokenize(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, DetokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    handler = models(raw_request)

    models_ = await handler.show_available_models()
    return JSONResponse(content=models_.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": APHRODITE_VERSION}
    return JSONResponse(content=ver)


@router.get("/.well-known/serviceinfo")
async def serviceinfo():
    """Return service information including version, API endpoints,
    and documentation URLs."""

    return JSONResponse(
        content={
            "version": 0.2,
            "software": {
                "name": "Aphrodite Engine",
                "version": APHRODITE_VERSION,
                "repository":
                "https://github.com/PygmalionAI/aphrodite-engine",
                "homepage": "https://aphrodite.pygmalion.chat",
                "logo": "https://pygmalion.chat/icons/favicon.ico",
            },
            "api": {
                "openai": {
                    "name": "OpenAI API",
                    "rel_url": "/v1",
                    "documentation": "/redoc",
                    "version": 1,
                },
                "koboldai": {
                    "name": "KoboldAI API",
                    "rel_url": "/api",
                    "documentation": "/redoc",
                    "version": 1,
                }
            }
        })


@router.post("/v1/chat/completions",
             dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    handler = chat(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Chat Completions API")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/completions", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    handler = completion(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Completions API")

    generator = await handler.create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/embeddings", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    handler = embedding(raw_request)
    if handler is None:
        fallback_handler = pooling(raw_request)
        if fallback_handler is None:
            return base(raw_request).create_error_response(
                message="The model does not support Embeddings API")

        logger.warning(
            "Embeddings API will become exclusive to embedding models "
            "in a future release. To return the hidden states directly, "
            "use the Pooling API (`/pooling`) instead.")

        res = await fallback_handler.create_pooling(request, raw_request)

        generator: Union[ErrorResponse, EmbeddingResponse]
        if isinstance(res, PoolingResponse):
            generator = EmbeddingResponse(
                id=res.id,
                object=res.object,
                created=res.created,
                model=res.model,
                data=[
                    EmbeddingResponseData(
                        index=d.index,
                        embedding=d.data,  # type: ignore
                    ) for d in res.data
                ],
                usage=res.usage,
            )
        else:
            generator = res
    else:
        generator = await handler.create_embedding(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, EmbeddingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/pooling", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_pooling(request: PoolingRequest, raw_request: Request):
    handler = pooling(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Pooling API")

    generator = await handler.create_pooling(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, PoolingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/score", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_score(request: ScoreRequest, raw_request: Request):
    handler = score(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Score API")

    generator = await handler.create_score(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, ScoreResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/v1/score", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_score_v1(request: ScoreRequest, raw_request: Request):
    logger.warning(
        "To indicate that Score API is not part of standard OpenAI API, we "
        "have moved it to `/score`. Please update your client accordingly.")

    return await create_score(request, raw_request)


@router.post("/v1/audio/transcriptions")
@with_cancellation
@load_aware_call
async def create_transcriptions(request: Annotated[TranscriptionRequest,
                                                   Form()],
                                raw_request: Request):
    handler = transcription(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Transcriptions API")

    audio_data = await request.file.read()
    generator = await handler.create_transcription(audio_data, request,
                                                   raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, TranscriptionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/rerank", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def do_rerank(request: RerankRequest, raw_request: Request):
    handler = rerank(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Rerank (Score) API")
    generator = await handler.do_rerank(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, RerankResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/v1/rerank", dependencies=[Depends(validate_json_request)])
@with_cancellation
async def do_rerank_v1(request: RerankRequest, raw_request: Request):
    log_once(
        "warning",
        "To indicate that the rerank API is not part of the standard OpenAI"
        " API, we have located it at `/rerank`. Please update your client "
        "accordingly. (Note: Conforms to JinaAI rerank API)",
    )

    return await do_rerank(request, raw_request)


@router.post("/v2/rerank", dependencies=[Depends(validate_json_request)])
@with_cancellation
async def do_rerank_v2(request: RerankRequest, raw_request: Request):
    return await do_rerank(request, raw_request)


TASK_HANDLERS: dict[str, dict[str, tuple]] = {
    "generate": {
        "messages": (ChatCompletionRequest, create_chat_completion),
        "default": (CompletionRequest, create_completion),
    },
    "embed": {
        "messages": (EmbeddingChatRequest, create_embedding),
        "default": (EmbeddingCompletionRequest, create_embedding),
    },
    "score": {
        "default": (RerankRequest, do_rerank)
    },
    "rerank": {
        "default": (RerankRequest, do_rerank)
    },
    "reward": {
        "messages": (PoolingChatRequest, create_pooling),
        "default": (PoolingCompletionRequest, create_pooling),
    },
    "classify": {
        "messages": (PoolingChatRequest, create_pooling),
        "default": (PoolingCompletionRequest, create_pooling),
    },
}

if envs.APHRODITE_SERVER_DEV_MODE:

    @router.get("/server_info")
    async def show_server_info(raw_request: Request):
        server_info = {
            "aphrodite_config": str(raw_request.app.state.aphrodite_config)
        }
        return JSONResponse(content=server_info)

    @router.post("/reset_prefix_cache")
    async def reset_prefix_cache(raw_request: Request):
        """
        Reset the prefix cache. Note that we currently do not check if the
        prefix cache is successfully reset in the API server.
        """
        device = None
        device_str = raw_request.query_params.get("device")
        if device_str is not None:
            device = Device[device_str.upper()]
        logger.info("Resetting prefix cache with specific {}...", str(device))
        await engine_client(raw_request).reset_prefix_cache(device)
        return Response(status_code=200)

    @router.post("/sleep")
    async def sleep(raw_request: Request):
        # get POST params
        level = raw_request.query_params.get("level", "1")
        await engine_client(raw_request).sleep(int(level))
        # FIXME: in v0 with frontend multiprocessing, the sleep command
        # is sent but does not finish yet when we return a response.
        return Response(status_code=200)

    @router.post("/wake_up")
    async def wake_up(raw_request: Request):
        tags = raw_request.query_params.getlist("tags")
        if tags == []:
            # set to None to wake up all tags if no tags are provided
            tags = None
        logger.info("wake up the engine with tags: {}", tags)
        await engine_client(raw_request).wake_up(tags)
        # FIXME: in v0 with frontend multiprocessing, the wake-up command
        # is sent but does not finish yet when we return a response.
        return Response(status_code=200)

    @router.get("/is_sleeping")
    async def is_sleeping(raw_request: Request):
        logger.info("check whether the engine is sleeping")
        is_sleeping = await engine_client(raw_request).is_sleeping()
        return JSONResponse(content={"is_sleeping": is_sleeping})


@router.post("/invocations", dependencies=[Depends(validate_json_request)])
async def invocations(raw_request: Request):
    """
    For SageMaker, routes requests to other handlers based on model `task`.
    """
    body = await raw_request.json()
    task = raw_request.app.state.task

    if task not in TASK_HANDLERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task: '{task}' for '/invocations'. "
            f"Expected one of {set(TASK_HANDLERS.keys())}")

    handler_config = TASK_HANDLERS[task]
    if "messages" in body:
        request_model, handler = handler_config["messages"]
    else:
        request_model, handler = handler_config["default"]

    # this is required since we lose the FastAPI automatic casting
    request = request_model.model_validate(body)
    return await handler(request, raw_request)


if envs.APHRODITE_TORCH_PROFILER_DIR:
    logger.warning(
        "Torch Profiler is enabled in the API server. This should ONLY be "
        "used for local development!")

    @router.post("/start_profile")
    async def start_profile(raw_request: Request):
        logger.info("Starting profiler...")
        await engine_client(raw_request).start_profile()
        logger.info("Profiler started.")
        return Response(status_code=200)

    @router.post("/stop_profile")
    async def stop_profile(raw_request: Request):
        logger.info("Stopping profiler...")
        await engine_client(raw_request).stop_profile()
        logger.info("Profiler stopped.")
        return Response(status_code=200)


if envs.APHRODITE_ALLOW_RUNTIME_LORA_UPDATING:
    logger.warning(
        "LoRA dynamic loading & unloading is enabled in the API server. "
        "This should ONLY be used for local development!")

    @router.post("/v1/load_lora_adapter",
                 dependencies=[Depends(validate_json_request)])
    async def load_lora_adapter(request: LoadLoRAAdapterRequest,
                                raw_request: Request):
        handler = models(raw_request)
        response = await handler.load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)

    @router.post("/v1/unload_lora_adapter",
                 dependencies=[Depends(validate_json_request)])
    async def unload_lora_adapter(request: UnloadLoRAAdapterRequest,
                                  raw_request: Request):
        handler = models(raw_request)
        response = await handler.unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)


# ============ KoboldAI API ============ #

badwordsids: list[int] = []


def prepare_engine_payload(
        kai_payload: KAIGenerationInputSchema,
        tokenizer
) -> tuple[SamplingParams, list[int]]:
    """Create SamplingParams and truncated input tokens for AsyncEngine"""

    if not kai_payload.genkey:
        kai_payload.genkey = f"kai-{random_uuid()}"

    kai_payload.top_k = kai_payload.top_k if kai_payload.top_k != 0.0 else -1
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
        include_stop_str_in_output=kai_payload.include_stop_str_in_output or False,
        custom_token_bans=badwordsids
        if kai_payload.use_default_badwordsids else [],
        max_tokens=kai_payload.max_length,
        seed=kai_payload.sampler_seed,
        xtc_probability=kai_payload.xtc_probability or 0.0,
        xtc_threshold=kai_payload.xtc_threshold or 0.0,
    )

    max_input_tokens = max(
        1, kai_payload.max_context_length - kai_payload.max_length)
    input_tokens = tokenizer(kai_payload.prompt).input_ids[-max_input_tokens:]

    return sampling_params, input_tokens


@kai_api.post("/generate")
async def generate(kai_payload: KAIGenerationInputSchema,
                   raw_request: Request) -> JSONResponse:
    tokenizer = await engine_client(raw_request).get_tokenizer()
    sampling_params, input_tokens = prepare_engine_payload(kai_payload, tokenizer)
    result_generator = engine_client(raw_request).generate(
        {
            "prompt": kai_payload.prompt,
            "prompt_token_ids": input_tokens,
        },
        sampling_params,
        kai_payload.genkey,
    )

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
async def generate_stream(kai_payload: KAIGenerationInputSchema,
                          raw_request: Request) -> StreamingResponse:

    tokenizer = await engine_client(raw_request).get_tokenizer()
    sampling_params, input_tokens = prepare_engine_payload(kai_payload, tokenizer)
    results_generator = engine_client(raw_request).generate(
        {
            "prompt": kai_payload.prompt,
            "prompt_token_ids": input_tokens,
        },
        sampling_params,
        kai_payload.genkey,
    )

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
                                 "Connection": "keep-alive",
                             },
                             media_type="text/event-stream")


@extra_api.post("/generate/check")
@extra_api.get("/generate/check")
async def check_generation(request: Request):
    text = ""
    try:
        request_dict = await request.json()
        if "genkey" in request_dict and request_dict["genkey"] in gen_cache:
            text = gen_cache[request_dict["genkey"]]
    except json.JSONDecodeError:
        pass

    return JSONResponse({"results": [{"text": text}]})


@extra_api.post("/abort")
async def abort_generation(raw_request: Request):
    try:
        request_dict = await raw_request.json()
        if "genkey" in request_dict:
            await engine_client(raw_request).abort(request_dict["genkey"])
    except json.JSONDecodeError:
        pass

    return JSONResponse({})


@extra_api.post("/tokencount")
async def count_tokens(request: TokenizeRequest, raw_request: Request):
    """Tokenize string and return token count"""

    generator = await tokenization(raw_request).create_tokenize(
        request, raw_request)
    return JSONResponse({"value": generator.model_dump()["tokens"]})


@kai_api.get("/info/version")
async def get_version():
    """Impersonate KAI"""
    return JSONResponse({"result": "1.2.4"})


@kai_api.get("/model")
async def get_model(raw_request: Request):
    return JSONResponse(
        {"result": f"aphrodite/{raw_request.app.state.served_model_names[0]}"})


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
async def get_max_length(raw_request: Request) -> JSONResponse:
    max_length = raw_request.app.state.aphrodite_config.max_model_len
    return JSONResponse({"value": max_length})


@kai_api.get("/config/max_context_length")
@extra_api.get("/true_max_context_length")
async def get_max_context_length(raw_request: Request) -> JSONResponse:
    max_context_length = raw_request.app.state.aphrodite_config.max_model_len
    return JSONResponse({"value": max_context_length})


@extra_api.get("/preloadstory")
async def get_preloaded_story() -> JSONResponse:
    """Stub for compatibility"""
    return JSONResponse({})


@extra_api.get("/version")
async def get_extra_version():
    """Impersonate KoboldCpp"""
    return JSONResponse({"result": "KoboldCpp", "version": "1.63"})


@router.get("/")
async def get_kobold_lite_ui():
    """Serves a cached copy of the Kobold Lite UI, loading it from disk
    on demand if needed. Can be disabled with SERVE_KOBOLD_LITE_UI=0."""
    if not SERVE_KOBOLD_LITE_UI:
        return JSONResponse(content={"error": "Kobold Lite UI is disabled"},
                            status_code=404)
    global kobold_lite_ui
    if kobold_lite_ui == "":
        scriptpath = os.path.dirname(os.path.abspath(__file__))
        klitepath = os.path.join(scriptpath, "../kobold/klite.embd")
        klitepath = os.path.normpath(klitepath)  # Normalize the path
        if os.path.exists(klitepath):
            with open(klitepath, "r", encoding="utf-8") as f:
                kobold_lite_ui = f.read()
        else:
            logger.error("Kobold Lite UI not found at " + klitepath)
    return HTMLResponse(content=kobold_lite_ui)


# ============ KoboldAI API ============ #


def build_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:
        app = FastAPI(openapi_url=None,
                      docs_url=None,
                      redoc_url=None,
                      lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    
    # Include KoboldAI API routes if enabled
    if envs.APHRODITE_KOBOLD_API:
        app.include_router(kai_api, prefix="/api/v1")
        app.include_router(extra_api, prefix="/api/extra")
        logger.info("KoboldAI API routes enabled")
    
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        err = ErrorResponse(message=str(exc),
                            type="BadRequestError",
                            code=HTTPStatus.BAD_REQUEST)
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    # Ensure --api-key option from CLI takes precedence over APHRODITE_API_KEY
    if token := args.api_keys or envs.APHRODITE_API_KEY:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if request.method == "OPTIONS":
                return await call_next(request)
            url_path = request.url.path
            if app.root_path and url_path.startswith(app.root_path):
                url_path = url_path[len(app.root_path):]
            if not url_path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    if args.enable_request_id_headers:
        logger.warning(
            "CAUTION: Enabling X-Request-Id headers in the API Server. "
            "This can harm performance at high QPS.")

        @app.middleware("http")
        async def add_request_id(request: Request, call_next):
            request_id = request.headers.get(
                "X-Request-Id") or uuid.uuid4().hex
            response = await call_next(request)
            response.headers["X-Request-Id"] = request_id
            return response

    if envs.APHRODITE_DEBUG_LOG_API_SERVER_RESPONSE:
        logger.warning("CAUTION: Enabling log response in the API Server. "
                       "This can include sensitive information and should be "
                       "avoided in production.")

        @app.middleware("http")
        async def log_response(request: Request, call_next):
            response = await call_next(request)
            response_body = [
                section async for section in response.body_iterator
            ]
            response.body_iterator = iterate_in_threadpool(iter(response_body))
            logger.info("response_body={{}}",
                        response_body[0].decode() if response_body else None)
            return response

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)  # type: ignore[arg-type]
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    return app


async def init_app_state(
    engine_client: EngineClient,
    aphrodite_config: AphroditeConfig,
    state: State,
    args: Namespace,
) -> None:
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]
    state.served_model_names = served_model_names

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model)
        for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.aphrodite_config = aphrodite_config
    model_config = aphrodite_config.model_config

    resolved_chat_template = load_chat_template(args.chat_template)
    if resolved_chat_template is not None:
        # Get the tokenizer to check official template
        tokenizer = await engine_client.get_tokenizer()

        if isinstance(tokenizer, MistralTokenizer):
            # The warning is logged in resolve_mistral_chat_template.
            resolved_chat_template = resolve_mistral_chat_template(
                chat_template=resolved_chat_template)
        else:
            hf_chat_template = resolve_hf_chat_template(
                tokenizer,
                chat_template=None,
                tools=None,
                trust_remote_code=model_config.trust_remote_code)

            if hf_chat_template != resolved_chat_template:
                logger.warning(
                    "Using supplied chat template: {}\n"
                    "It is different from official chat template '{}'. "
                    "This discrepancy may lead to performance degradation.",
                    resolved_chat_template, args.model)

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
    )
    await state.openai_serving_models.init_static_loras()
    state.openai_serving_chat = OpenAIServingChat(
        engine_client,
        model_config,
        state.openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
    ) if model_config.runner_type == "generate" else None
    state.openai_serving_completion = OpenAIServingCompletion(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    ) if model_config.runner_type == "generate" else None
    state.openai_serving_pooling = OpenAIServingPooling(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
    ) if model_config.runner_type == "pooling" else None
    state.openai_serving_embedding = OpenAIServingEmbedding(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
    ) if model_config.task == "embed" else None
    state.openai_serving_scores = ServingScores(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger) if model_config.task in (
            "score", "embed", "pooling") else None
    state.jinaai_serving_reranking = ServingScores(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger
    ) if model_config.task == "score" else None
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
    )
    state.openai_serving_transcription = OpenAIServingTranscription(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
    ) if model_config.runner_type == "transcription" else None
    state.task = model_config.task

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


def create_server_socket(addr: tuple[str, int]) -> socket.socket:
    family = socket.AF_INET
    if is_valid_ipv6_address(addr[0]):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(addr)

    return sock


async def run_server(args, **uvicorn_kwargs) -> None:
    logger.info("Aphrodite API server version {}", APHRODITE_VERSION)
    logger.debug("args: {}", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valid_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice \
        and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(f"invalid tool call parser: {args.tool_call_parser} "
                       f"(chose from {{ {','.join(valid_tool_parses)} }})")

    valid_reasoning_parses = ReasoningParserManager.reasoning_parsers.keys()
    if args.reasoning_parser \
        and args.reasoning_parser not in valid_reasoning_parses:
        raise KeyError(
            f"invalid reasoning parser: {args.reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parses)} }})")

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        aphrodite_config = await engine_client.get_aphrodite_config()
        await init_app_state(engine_client, aphrodite_config, app.state, args)

        def _listen_addr(a: str) -> str:
            if is_valid_ipv6_address(a):
                return '[' + a + ']'
            return a or "0.0.0.0"

        is_ssl = args.ssl_keyfile and args.ssl_certfile
        logger.info("Starting Aphrodite API server on http{}://{}:{}",
                    "s" if is_ssl else "", _listen_addr(sock_addr[0]),
                    sock_addr[1])

        protocol = "https" if args.ssl_certfile else "http"
        root_path = args.root_path.rstrip("/") if args.root_path else ""
        host_name = args.host if args.host else "localhost"
        port_str = str(args.port)

        if SERVE_KOBOLD_LITE_UI:
            ui_url = f"{protocol}://{host_name}:{port_str}{root_path}/"
            logger.info(f"Kobold Lite UI:    {ui_url}")

        if not args.disable_fastapi_docs:
            logger.info(
                f"Documentation:     {protocol}://{host_name}:{port_str}{root_path}/redoc"
            )  # noqa: E501
        logger.info(
            f"Completions API:   {protocol}://{host_name}:{port_str}{root_path}/v1/completions"
        )  # noqa: E501
        logger.info(
            f"Chat API:          {protocol}://{host_name}:{port_str}{root_path}/v1/chat/completions"
        )  # noqa: E501
        logger.info(
            f"Embeddings API:    {protocol}://{host_name}:{port_str}{root_path}/v1/embeddings"
        )  # noqa: E501
        logger.info(
            f"Pooling API:       {protocol}://{host_name}:{port_str}{root_path}/pooling"
        )  # noqa: E501
        logger.info(
            f"Score API:         {protocol}://{host_name}:{port_str}{root_path}/score"
        )  # noqa: E501
        logger.info(
            f"Rerank API:        {protocol}://{host_name}:{port_str}{root_path}/rerank"
        )  # noqa: E501
        logger.info(
            f"Rerank API v1:     {protocol}://{host_name}:{port_str}{root_path}/v1/rerank"
        )  # noqa: E501
        logger.info(
            f"Rerank API v2:     {protocol}://{host_name}:{port_str}{root_path}/v2/rerank"
        )  # noqa: E501
        logger.info(
            f"Transcription API: {protocol}://{host_name}:{port_str}{root_path}/v1/audio/transcriptions"
        )  # noqa: E501
        logger.info(
            f"Tokenization API:  {protocol}://{host_name}:{port_str}{root_path}/v1/tokenize"
        )  # noqa: E501

        if envs.APHRODITE_KOBOLD_API:
            logger.info(
                f"KoboldAI API:      {protocol}://{host_name}:{port_str}{root_path}/api/v1"
            )  # noqa: E501
            logger.info(
                f"KoboldAI Extra:    {protocol}://{host_name}:{port_str}{root_path}/api/extra"
            )  # noqa: E501

        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


if __name__ == "__main__":
    # NOTE:
    # This section should be in sync with aphrodite/endpoints/cli.py
    # for CLI entrypoints.
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="Aphrodite OpenAI-Compatible RESTful API Server")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
