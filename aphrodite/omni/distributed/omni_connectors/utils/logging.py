# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Callable

_aphrodite_init_logger: Callable[[str], logging.Logger] | None

try:
    from aphrodite.logger import init_logger

    _aphrodite_init_logger = init_logger
except Exception:  # pragma: no cover - optional dependency
    _aphrodite_init_logger = None


def get_connector_logger(name: str) -> logging.Logger:
    """Return a logger preferring Sonar's init_logger when available."""
    return _aphrodite_init_logger(name) if _aphrodite_init_logger is not None else logging.getLogger(name)
