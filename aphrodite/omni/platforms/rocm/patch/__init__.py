# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def apply_patches():
    """Apply all ROCm-specific patches."""
    from aphrodite.omni.platforms.rocm.patch import worker  # noqa: F401
