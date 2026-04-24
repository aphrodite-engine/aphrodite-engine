# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
import pytest
from transformers import AutoTokenizer

from aphrodite.transformers_utils.tokenizer import AnyTokenizer


@pytest.fixture(scope="function")
def default_tokenizer() -> AnyTokenizer:
    return AutoTokenizer.from_pretrained("gpt2")
