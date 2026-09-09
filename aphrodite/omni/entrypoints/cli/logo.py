# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import regex as re

from aphrodite.logger import init_logger

logger = init_logger(__name__)

ORANGE = "\033[38;5;208m"
BLUE = "\033[34m"
WHITE = "\033[97m"
PURPLE = "\033[35m"
RESET = "\033[0m"

LOGO = f"{WHITE}Sonar {BLUE}Omni{RESET}"


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def log_logo() -> None:
    # Bypass current_formatter_type() which has fragile handler-count/name
    # checks that fail in some runtime configurations. Instead, directly
    # check the env vars that control colored output — matching Sonar's
    # _use_color() logic. main.py sets APHRODITE_LOGGING_COLOR=1 before any
    # Sonar import, so this env var is the authoritative source of truth.
    use_color = "NO_COLOR" not in os.environ and os.environ.get("APHRODITE_LOGGING_COLOR") != "0"
    logo = LOGO if use_color else _ANSI_RE.sub("", LOGO)
    logger.info(logo)
