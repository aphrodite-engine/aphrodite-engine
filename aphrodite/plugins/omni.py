# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register Omni models only in explicitly enabled engine processes."""

import os


def register() -> None:
    if os.environ.get("APHRODITE_OMNI_ENABLED") != "1":
        return
    from aphrodite.omni.engine.arg_utils import register_omni_models_to_aphrodite

    register_omni_models_to_aphrodite()
