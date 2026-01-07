"""Lightweight tokenizer-only HTTP server.

This module provides a standalone tokenizer server that exposes tokenization
endpoints without running the full Aphrodite engine.
"""

from __future__ import annotations

import signal
import socket
from argparse import Namespace
from http import HTTPStatus

import uvloop
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from typing_extensions import assert_never

from aphrodite.endpoints.logger import RequestLogger
from aphrodite.endpoints.openai.protocol import (
    DetokenizeRequest,
    DetokenizeResponse,
    ErrorInfo,
    ErrorResponse,
    TokenizeResponse,
)
from aphrodite.endpoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from aphrodite.endpoints.openai.serving_tokenization import OpenAIServingTokenization
from aphrodite.endpoints.utils import with_cancellation
from aphrodite.engine.protocol import EngineClient
from aphrodite.logger import init_logger
from aphrodite.lora.request import LoRARequest
from aphrodite.server import serve_http
from aphrodite.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer
from aphrodite.utils.argparse_utils import FlexibleArgumentParser
from aphrodite.utils.network_utils import is_valid_ipv6_address
from aphrodite.version import __version__ as APHRODITE_VERSION

router = APIRouter()

logger = init_logger(__name__)


class MinimalModelConfig:
    """Minimal model config for tokenizer-only server."""

    def __init__(
        self,
        model: str,
        tokenizer: str | None = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tokenizer_revision: str | None = None,
        revision: str | None = None,
        max_model_len: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer or model
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.tokenizer_revision = tokenizer_revision
        self.revision = revision
        # aphrodite's default max_model_len if not specified
        self.max_model_len = max_model_len or 2048
        self.runner_type = "generate"
        self.supported_tasks = ["generate"]
        self.encoder_config = None  # No encoder for tokenizer-only mode


class DummyEngineClient:
    """Minimal EngineClient implementation for tokenization-only server.

    This provides just enough implementation to use OpenAIServingTokenization
    without running the full Aphrodite engine.
    """

    def __init__(self, model_config: MinimalModelConfig):
        self.model_config = model_config
        self._tokenizer: AnyTokenizer | None = None
        self.processor = None  # Not needed for tokenizer-only server
        self.io_processor = None  # Not needed for tokenizer-only server

    def _ensure_tokenizer(self) -> AnyTokenizer:
        """Lazy-load the tokenizer on first use."""
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer(
                self.model_config.tokenizer,
                tokenizer_mode=self.model_config.tokenizer_mode,
                trust_remote_code=self.model_config.trust_remote_code,
                revision=self.model_config.tokenizer_revision,
            )
        return self._tokenizer

    async def get_tokenizer(self, lora_request: LoRARequest | None = None) -> AnyTokenizer:
        """Get the tokenizer (ignores LoRA for tokenizer-only server)."""
        return self._ensure_tokenizer()

    async def check_health(self) -> None:
        """Health check always passes for tokenizer server."""
        pass

    async def get_model_config(self) -> MinimalModelConfig:
        """Get the model configuration."""
        return self.model_config

    @property
    def is_running(self) -> bool:
        return True

    @property
    def is_stopped(self) -> bool:
        return False

    @property
    def errored(self) -> bool:
        return False

    @property
    def dead_error(self) -> BaseException:
        raise RuntimeError("Tokenizer server does not support this operation")


def tokenization(request: Request) -> OpenAIServingTokenization:
    """Dependency to get the tokenization handler from app state."""
    return request.app.state.openai_serving_tokenization


@router.get("/health", response_class=Response)
async def health() -> Response:
    """Health check endpoint."""
    return Response(status_code=200)


@router.get("/ping", response_class=Response)
@router.post("/ping", response_class=Response)
async def ping() -> Response:
    """Ping check endpoint."""
    return Response(status_code=200)


@router.post(
    "/v1/tokenize",
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def tokenize(raw_request: Request):
    """Tokenize a prompt."""
    try:
        body = await raw_request.json()
        if "prompt" in body:
            from aphrodite.endpoints.openai.protocol import TokenizeCompletionRequest

            request = TokenizeCompletionRequest.model_validate(body)
        elif "messages" in body:
            from aphrodite.endpoints.openai.protocol import TokenizeChatRequest

            request = TokenizeChatRequest.model_validate(body)
        else:
            raise ValueError("Request must contain either 'prompt' or 'messages' field")
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"Invalid request format: {str(e)}",
        ) from e

    handler = tokenization(raw_request)

    try:
        generator = await handler.create_tokenize(request, raw_request)
    except NotImplementedError as e:
        raise HTTPException(status_code=HTTPStatus.NOT_IMPLEMENTED.value, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.error.code)
    elif isinstance(generator, TokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post(
    "/v1/detokenize",
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def detokenize(raw_request: Request):
    """Detokenize token IDs."""
    try:
        body = await raw_request.json()
        request = DetokenizeRequest.model_validate(body)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"Invalid request format: {str(e)}",
        ) from e

    handler = tokenization(raw_request)

    try:
        generator = await handler.create_detokenize(request, raw_request)
    except OverflowError as e:
        raise RequestValidationError(errors=[str(e)]) from e
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.error.code)
    elif isinstance(generator, DetokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.get("/tokenizer_info")
async def get_tokenizer_info(raw_request: Request):
    """Get comprehensive tokenizer information."""
    result = await tokenization(raw_request).get_tokenizer_info()
    return JSONResponse(
        content=result.model_dump(),
        status_code=result.code if isinstance(result, ErrorResponse) else 200,
    )


@router.get("/version")
async def show_version():
    """Show the Aphrodite version."""
    ver = {"version": APHRODITE_VERSION}
    return JSONResponse(content=ver)


def build_app(args: Namespace) -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title="Aphrodite Tokenizer Server",
        description="Lightweight tokenization-only server",
    )
    app.include_router(router)

    if hasattr(args, "allowed_origins") and args.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=args.allowed_origins,
            allow_credentials=getattr(args, "allow_credentials", False),
            allow_methods=getattr(args, "allowed_methods", ["*"]),
            allow_headers=getattr(args, "allowed_headers", ["*"]),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException):
        err = ErrorResponse(
            error=ErrorInfo(
                message=exc.detail,
                type=HTTPStatus(exc.status_code).phrase,
                code=exc.status_code,
            )
        )
        return JSONResponse(err.model_dump(), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError):
        exc_str = str(exc)
        errors_str = str(exc.errors())

        if exc.errors() and errors_str and errors_str != exc_str:
            message = f"{exc_str} {errors_str}"
        else:
            message = exc_str

        err = ErrorResponse(
            error=ErrorInfo(
                message=message,
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST,
            )
        )
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)

    return app


async def init_app_state(
    engine_client: EngineClient,
    model_config: MinimalModelConfig,
    app: FastAPI,
    args: Namespace,
) -> None:
    """Initialize the application state."""
    app.state.engine_client = engine_client

    if hasattr(args, "served_model_name") and args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    app.state.served_model_names = served_model_names

    request_logger = None
    if hasattr(args, "enable_log_requests") and args.enable_log_requests:
        max_log_len = getattr(args, "max_log_len", 0)
        request_logger = RequestLogger(max_log_len=max_log_len)

    base_model_paths = [BaseModelPath(name=name, model_path=args.model) for name in served_model_names]

    app.state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
        lora_modules=None,
    )

    chat_template = getattr(args, "chat_template", None)
    chat_template_content_format = getattr(args, "chat_template_content_format", "auto")

    app.state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        app.state.openai_serving_models,
        request_logger=request_logger,
        chat_template=chat_template,
        chat_template_content_format=chat_template_content_format,
        log_error_stack=getattr(args, "log_error_stack", False),
    )


def create_server_socket(addr: tuple[str, int]) -> socket.socket:
    """Create a server socket."""
    family = socket.AF_INET
    if is_valid_ipv6_address(addr[0]):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(addr)

    return sock


def create_model_config_for_tokenizer(args: Namespace) -> MinimalModelConfig:
    """Create a minimal ModelConfig for tokenizer-only operation.

    This only populates fields needed for tokenization.
    """
    return MinimalModelConfig(
        model=args.model,
        tokenizer=getattr(args, "tokenizer", None),
        tokenizer_mode=getattr(args, "tokenizer_mode", "auto"),
        trust_remote_code=getattr(args, "trust_remote_code", False),
        tokenizer_revision=getattr(args, "tokenizer_revision", None),
        revision=getattr(args, "revision", None),
        max_model_len=getattr(args, "max_model_len", None),
    )


async def run_server(args: Namespace) -> None:
    """Run the tokenizer server."""
    logger.info("Aphrodite Tokenizer Server version %s", APHRODITE_VERSION)
    logger.info("Starting tokenizer server for model: %s", args.model)

    sock_addr = (args.host or "0.0.0.0", args.port)
    sock = create_server_socket(sock_addr)

    def signal_handler(*_) -> None:
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    app = build_app(args)

    model_config = create_model_config_for_tokenizer(args)
    engine_client = DummyEngineClient(model_config)

    await init_app_state(engine_client, model_config, app, args)

    # Log server info
    host_name = args.host if args.host else "localhost"
    port_str = str(args.port)
    base_url = f"http://{host_name}:{port_str}"

    logger.info("Tokenizer server listening on %s", base_url)
    logger.info("Health check:                    %s/health", base_url)
    logger.info("Tokenization API:                %s/v1/tokenize", base_url)
    logger.info("Detokenization API:              %s/v1/detokenize", base_url)
    logger.info("Tokenizer Info API:              %s/tokenizer_info", base_url)
    logger.info("Version API:                     %s/version", base_url)

    shutdown_task = await serve_http(
        app,
        sock=sock,
        host=args.host,
        port=args.port,
        log_level=getattr(args, "uvicorn_log_level", "info"),
        access_log=not getattr(args, "disable_uvicorn_access_log", False),
    )

    try:
        await shutdown_task
    finally:
        sock.close()


def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    # TODO: move this out of here
    """Add tokenizer server arguments to the parser."""
    parser.add_argument("model", type=str, help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=2242, help="Port to bind the server to")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name or path")
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        choices=["auto", "slow", "mistral"],
        help="Tokenizer mode",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when downloading model and tokenizer",
    )
    parser.add_argument(
        "--tokenizer-revision",
        type=str,
        default=None,
        help="Tokenizer revision",
    )
    parser.add_argument("--revision", type=str, default=None, help="Model revision")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length",
    )
    parser.add_argument("--chat-template", type=str, default=None, help="Chat template to use")
    parser.add_argument(
        "--chat-template-content-format",
        type=str,
        default="auto",
        choices=["auto", "string", "openai"],
        help="Chat template content format",
    )
    parser.add_argument(
        "--enable-log-requests",
        action="store_true",
        help="Enable logging of requests",
    )
    parser.add_argument(
        "--max-log-len",
        type=int,
        default=0,
        help="Maximum length of log messages",
    )
    parser.add_argument(
        "--log-error-stack",
        action="store_true",
        help="Log full error stack traces",
    )
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Uvicorn log level",
    )
    parser.add_argument(
        "--disable-uvicorn-access-log",
        action="store_true",
        help="Disable uvicorn access log",
    )
    parser.add_argument(
        "--served-model-name",
        nargs="+",
        type=str,
        default=None,
        help="The model name(s) used in the API",
    )
    parser.add_argument(
        "--allowed-origins",
        type=lambda s: s.split(","),
        default=["*"],
        help="Comma-separated list of allowed origins for CORS",
    )
    parser.add_argument(
        "--allow-credentials",
        action="store_true",
        help="Allow credentials for CORS",
    )
    parser.add_argument(
        "--allowed-methods",
        type=lambda s: s.split(","),
        default=["*"],
        help="Comma-separated list of allowed methods for CORS",
    )
    parser.add_argument(
        "--allowed-headers",
        type=lambda s: s.split(","),
        default=["*"],
        help="Comma-separated list of allowed headers for CORS",
    )

    return parser


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Aphrodite Tokenizer Server")
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    uvloop.run(run_server(args))
