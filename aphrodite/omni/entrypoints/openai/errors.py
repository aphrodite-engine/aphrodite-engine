# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


class InvalidInputReferenceError(ValueError):
    def __init__(self, message: str = "Invalid input reference.") -> None:
        super().__init__(message)
