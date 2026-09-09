# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client-side libraries for Sonar Omni serving APIs.

Modules under this package speak to a running Sonar Omni server over the
network. They are for applications and tests; server runtime code
(``aphrodite.omni.engine``, ``aphrodite.omni.entrypoints``, ``aphrodite.omni.model_executor``)
must never import from here.
"""
