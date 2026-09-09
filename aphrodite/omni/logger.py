# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

from aphrodite.logger import init_logger


def _configure_aphrodite_omni_root_logger():
    """
    Configure the root logger for aphrodite.omni to propagate to aphrodite's root logger.
    """
    aphrodite_root = logging.getLogger("aphrodite")
    aphrodite_omni_root = logging.getLogger("aphrodite.omni")
    aphrodite_omni_root.handlers = []

    aphrodite_omni_root.parent = aphrodite_root

    aphrodite_omni_root.propagate = True

    aphrodite_omni_root.setLevel(logging.NOTSET)


_configure_aphrodite_omni_root_logger()
init_logger(__name__)
