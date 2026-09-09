# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CLI helpers for Sonar Omni entrypoints."""

# To ensure patch imports work properly, disable unused import checks
# ruff: noqa: E402, F401
# isort: off
from aphrodite.omni.benchmarks.patch import patch
# isort: on

from aphrodite.omni.entrypoints.cli.benchmark.omni_duplex_eval import OmniDuplexEvalSubcommand
from aphrodite.omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand

from .serve import OmniServeCommand

__all__ = ["OmniServeCommand", "OmniBenchmarkServingSubcommand", "OmniDuplexEvalSubcommand"]
