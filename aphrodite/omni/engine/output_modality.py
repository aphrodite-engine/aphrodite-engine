# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

from aphrodite.omni.outputs.output_modality import *  # noqa: F401,F403

warnings.warn(
    "Importing from 'aphrodite.omni.engine.output_modality' is deprecated. Use 'aphrodite.omni.outputs.output_modality' instead.",
    DeprecationWarning,
    stacklevel=2,
)
