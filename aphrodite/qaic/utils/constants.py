# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
QEFF_DIR = os.path.dirname(UTILS_DIR)
ROOT_DIR = os.path.dirname(QEFF_DIR)
QEFF_CACHE_DIR_NAME = "qeff_cache"


# Store the qeff_models inside the ~/.cache directory or over-ride with an env
# variable.
def get_models_dir():
    """
    Determine the directory for storing QEFF models.
    Priority:
    1. Use $XDG_CACHE_HOME/qeff_models if XDG_CACHE_HOME is set.
    2. Use QEFF_HOME if set in environment.
    3. Default to ~/.cache/qeff_models.
    Sets QEFF_MODELS_DIR environment variable if not already set.
    Returns:
        str: Path to the QEFF models directory.
    """
    qeff_cache_home = os.environ.get("QEFF_HOME")
    # Check if XDG_CACHE_HOME is set
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if qeff_cache_home:
        qeff_models_dir = os.path.join(qeff_cache_home, QEFF_CACHE_DIR_NAME)
    # Check if QEFF_MODELS_DIR is set
    elif xdg_cache_home:
        qeff_models_dir = os.path.join(xdg_cache_home, QEFF_CACHE_DIR_NAME)
    else:
        # Use ~/.cache/qeff_models as the default
        qeff_models_dir = os.path.join(
            os.path.expanduser("~"), ".cache", QEFF_CACHE_DIR_NAME
        )

    # Set QEFF_MODELS_DIR environment variable
    return qeff_models_dir


QEFF_MODELS_DIR = get_models_dir()

ONNX_EXPORT_EXAMPLE_BATCH_SIZE = 1
ONNX_EXPORT_EXAMPLE_SEQ_LEN = 32


class Constants:
    # Export Constants.
    SEQ_LEN = 32
    CTX_LEN = 32
    PROMPT_LEN = 8
    INPUT_STR = ["My name is"]
    GB = 2**30
    MAX_QPC_LIMIT = 30
