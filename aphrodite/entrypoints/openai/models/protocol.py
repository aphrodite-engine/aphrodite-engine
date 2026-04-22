# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project


from dataclasses import dataclass


@dataclass
class BaseModelPath:
    name: str
    model_path: str


@dataclass
class LoRAModulePath:
    name: str
    path: str
    base_model_name: str | None = None
