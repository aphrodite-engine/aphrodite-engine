import pytest
from transformers import AutoTokenizer

from aphrodite.transformers_utils.tokenizer import AnyTokenizer


@pytest.fixture(scope="function")
def default_tokenizer() -> AnyTokenizer:
    return AutoTokenizer.from_pretrained("gpt2")
