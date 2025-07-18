"""
Internal logging utility.
"""

import datetime
import logging
import os
import sys
from functools import partial
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.markup import escape
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           TaskProgressColumn, TextColumn, TimeRemainingColumn)

import aphrodite.common.envs as envs

RICH_CONSOLE = Console()
LOG_LEVEL = os.getenv("APHRODITE_LOG_LEVEL", "INFO").upper()

APHRODITE_CONFIGURE_LOGGING = envs.APHRODITE_CONFIGURE_LOGGING


def unwrap(wrapped, default=None):
    """Unwrap function for Optionals."""
    if wrapped is None:
        return default
    return wrapped


def get_loading_progress_bar():
    """Gets a pre-made progress bar for loading tasks."""

    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=RICH_CONSOLE,
    )


def _log_formatter(record: dict):
    """Log message formatter."""

    color_map = {
        "TRACE": "dim blue",
        "DEBUG": "cyan",
        "INFO": "green",
        "SUCCESS": "bold green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold white on red",
    }
    level = record.get("level")
    level_color = color_map.get(level.name, "cyan")
    colored_level = f"[{level_color}]{level.name}[/{level_color}]:"

    separator = " " * (9 - len(level.name))

    message = unwrap(record.get("message"), "")

    # Replace once loguru allows for turning off str.format
    message = message.replace("{", "{{").replace("}", "}}").replace("<", r"\<")

    # Escape markup tags from Rich
    message = escape(message)
    lines = message.splitlines()

    fmt = ""
    if len(lines) > 1:
        fmt = "\n".join(
            [f"{colored_level}{separator}{line}" for line in lines])
    else:
        fmt = f"{colored_level}{separator}{message}"

    return fmt


_logged_messages = set()


def log_once(level, message, *args, **kwargs):
    if message not in _logged_messages:
        _logged_messages.add(message)
        logger.log(level, message, *args, **kwargs)


# Uvicorn log handler
# Uvicorn log portions inspired from https://github.com/encode/uvicorn/discussions/2027#discussioncomment-6432362
class UvicornLoggingHandler(logging.Handler):

    def emit(self, record: logging.LogRecord) -> None:
        logger.opt(exception=record.exc_info).log(record.levelname,
                                                  self.format(record).rstrip())


# Uvicorn config for logging. Passed into run when creating all loggers in
# server
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "uvicorn": {
            "class":
            f"{UvicornLoggingHandler.__module__}.{UvicornLoggingHandler.__qualname__}",  # noqa
        },
    },
    "root": {
        "handlers": ["uvicorn"],
        "propagate": False,
        "level": LOG_LEVEL
    },
}


def setup_logger():
    """Bootstrap the logger."""

    logger.remove()

    logger.add(
        RICH_CONSOLE.print,
        level=LOG_LEVEL,
        format=_log_formatter,
        colorize=True,
    )

    logger.log_once = log_once


setup_logger()


def _trace_calls(log_path, root_dir, frame, event, arg=None):
    if event in ['call', 'return']:
        # Extract the filename, line number, function name, and the code object
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        if not filename.startswith(root_dir):
            # only log the functions in the aphrodite root_dir
            return
        # Log every function call or return
        try:
            last_frame = frame.f_back
            if last_frame is not None:
                last_filename = last_frame.f_code.co_filename
                last_lineno = last_frame.f_lineno
                last_func_name = last_frame.f_code.co_name
            else:
                # initial frame
                last_filename = ""
                last_lineno = 0
                last_func_name = ""
            with open(log_path, 'a') as f:
                if event == 'call':
                    f.write(f"{datetime.datetime.now()} Call to"
                            f" {func_name} in {filename}:{lineno}"
                            f" from {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
                else:
                    f.write(f"{datetime.datetime.now()} Return from"
                            f" {func_name} in {filename}:{lineno}"
                            f" to {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
        except NameError:
            # modules are deleted during shutdown
            pass
    return partial(_trace_calls, log_path, root_dir)


def enable_trace_function_call(log_file_path: str,
                               root_dir: Optional[str] = None):
    """
    Enable tracing of every function call in code under `root_dir`.
    This is useful for debugging hangs or crashes.
    `log_file_path` is the path to the log file.
    `root_dir` is the root directory of the code to trace. If None, it is the
    aphrodite root directory.

    Note that this call is thread-level, any threads calling this function
    will have the trace enabled. Other threads will not be affected.
    """
    logger.warning(
        "APHRODITE_TRACE_FUNCTION is enabled. It will record every"
        " function executed by Python. This will slow down the code. It "
        "is suggested to be used for debugging hang or crashes only.")
    logger.info(f"Trace frame log is saved to {log_file_path}")
    if root_dir is None:
        # by default, this is the aphrodite root directory
        root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.settrace(partial(_trace_calls, log_file_path, root_dir))
