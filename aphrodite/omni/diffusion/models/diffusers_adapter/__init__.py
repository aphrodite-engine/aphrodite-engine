# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Diffusers backend adapter for Sonar Omni."""

from aphrodite.omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import (
    DiffusersAdapterPipeline,
)

__all__ = [
    "DiffusersAdapterPipeline",
]
