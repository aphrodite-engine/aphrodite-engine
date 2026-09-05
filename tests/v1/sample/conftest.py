# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


@pytest.fixture(autouse=True)
def restore_default_device():
    original_device = torch.get_default_device()
    try:
        yield
    finally:
        torch.set_default_device(original_device)
