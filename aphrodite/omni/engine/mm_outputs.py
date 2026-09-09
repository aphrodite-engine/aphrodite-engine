# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

from aphrodite.omni.outputs.mm_outputs import *  # noqa: F401,F403

warnings.warn(
    "Importing from 'aphrodite.omni.engine.mm_outputs' is deprecated. Use 'aphrodite.omni.outputs.mm_outputs' instead.",
    DeprecationWarning,
    stacklevel=2,
)
