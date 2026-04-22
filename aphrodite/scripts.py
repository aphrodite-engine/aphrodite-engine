# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from aphrodite.entrypoints.cli.main import main as aphrodite_main
from aphrodite.logger import init_logger

logger = init_logger(__name__)


# Backwards compatibility for the move from aphrodite.scripts to
# aphrodite.entrypoints.cli.main
def main():
    logger.warning(
        "aphrodite.scripts.main() is deprecated. Please re-install "
        "aphrodite or use aphrodite.entrypoints.cli.main.main() instead."
    )
    aphrodite_main()
