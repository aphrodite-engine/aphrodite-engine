# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
from .abstract import Executor
from .uniproc_executor import UniProcExecutor

__all__ = ["Executor", "UniProcExecutor"]
