# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

import aphrodite.envs as envs
from aphrodite.diffusion.runtime.entrypoints.openai import image_api, video_api
from aphrodite.diffusion.runtime.server_args import ServerArgs, prepare_server_args
from aphrodite.logging_utils.formatter import Colors, _supports_color


@asynccontextmanager
async def lifespan(app: FastAPI):
    from aphrodite.diffusion.runtime.scheduler_client import (
        run_zeromq_broker,
        scheduler_client,
    )

    # 1. Initialize the singleton client that connects to the backend Scheduler
    server_args = app.state.server_args
    scheduler_client.initialize(server_args)

    # 2. Start the ZMQ Broker in the background to handle offline requests
    broker_task = asyncio.create_task(run_zeromq_broker(server_args))

    yield

    # On shutdown
    print("FastAPI app is shutting down...")
    broker_task.cancel()
    scheduler_client.close()


class UvicornFormatter(logging.Formatter):
    """Custom formatter for uvicorn that matches Aphrodite's styling with colors."""

    def __init__(self, fmt, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)

        self.verbose_logging = envs.APHRODITE_LOGGING_VERBOSE
        self.use_color = _supports_color() and os.environ.get("APHRODITE_LOGGING_COLOR", "1") in ("1", "true", "True")
        self.level_colors = {
            "DEBUG": Colors.DEBUG,
            "INFO": Colors.INFO,
            "WARNING": Colors.WARNING,
            "ERROR": Colors.ERROR,
            "CRITICAL": Colors.CRITICAL,
        }
        self.path_color = Colors.PATH
        self.time_color = Colors.TIME
        self.reset_color = Colors.RESET

    def format(self, record):
        if not self.verbose_logging:
            original_datefmt = self.datefmt
            self.datefmt = "%H:%M:%S"

        msg = super().format(record)

        if not self.verbose_logging:
            self.datefmt = original_datefmt

        if "WARNING" in msg:
            msg = msg.replace("WARNING", "WARN", 1)

        if self.use_color:
            level_color = self.level_colors.get(record.levelname, "")
            level_str = "WARN" if record.levelname == "WARNING" else record.levelname

            if level_str in msg:
                msg = msg.replace(level_str, f"{level_color}{level_str}{self.reset_color}", 1)

            asctime = self.formatTime(record, self.datefmt if not self.verbose_logging else "%m-%d %H:%M:%S")
            if asctime in msg:
                msg = msg.replace(asctime, f"{self.time_color}{asctime}{self.reset_color}", 1)

            if self.verbose_logging:
                name_with_lineno = f"[{record.name:<15}:{record.lineno:>4}]"
                if name_with_lineno in msg:
                    msg = msg.replace(name_with_lineno, f"{self.path_color}{name_with_lineno}{self.reset_color}", 1)

        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])

        return msg


def create_uvicorn_log_config() -> dict:
    """Create uvicorn log config that matches Aphrodite's log format."""

    if envs.APHRODITE_LOGGING_VERBOSE:
        date_format = "%m-%d %H:%M:%S"
        default_format = (
            f"{envs.APHRODITE_LOGGING_PREFIX}%(levelname)s %(asctime)s [%(name)-15s:%(lineno)4d] %(message)s"
        )
        access_format = (
            f"{envs.APHRODITE_LOGGING_PREFIX}%(levelname)s %(asctime)s [%(name)-15s:%(lineno)4d] %(message)s"
        )
    else:
        date_format = "%H:%M:%S"
        default_format = f"{envs.APHRODITE_LOGGING_PREFIX}%(levelname)s %(asctime)s %(message)s"
        access_format = f"{envs.APHRODITE_LOGGING_PREFIX}%(levelname)s %(asctime)s %(message)s"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "aphrodite.diffusion.runtime.entrypoints.http_server.UvicornFormatter",
                "datefmt": date_format,
                "format": default_format,
            },
            "access": {
                "()": "aphrodite.diffusion.runtime.entrypoints.http_server.UvicornFormatter",
                "datefmt": date_format,
                "format": access_format,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": envs.APHRODITE_LOGGING_STREAM,
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": envs.APHRODITE_LOGGING_STREAM,
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }


def create_app(server_args: ServerArgs):
    """
    Create and configure the FastAPI application instance.
    """
    app = FastAPI(lifespan=lifespan)
    app.include_router(image_api.router)
    app.include_router(video_api.router)
    app.state.server_args = server_args
    return app


if __name__ == "__main__":
    import uvicorn

    server_args = prepare_server_args([])
    app = create_app(server_args)
    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port,
        log_config=None,
        reload=False,  # Set to True during development for auto-reloading
    )
