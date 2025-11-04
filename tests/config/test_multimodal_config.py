import pytest

from aphrodite.attention.backends.registry import _Backend
from aphrodite.config.multimodal import MultiModalConfig


def test_mm_encoder_attn_backend_str_conversion():
    config = MultiModalConfig(mm_encoder_attn_backend="FLASH_ATTN")
    assert config.mm_encoder_attn_backend == _Backend.FLASH_ATTN


def test_mm_encoder_attn_backend_invalid():
    with pytest.raises(ValueError):
        MultiModalConfig(mm_encoder_attn_backend="not_a_backend")


def test_mm_encoder_attn_backend_hash_updates():
    base_hash = MultiModalConfig().compute_hash()
    overridden_hash = MultiModalConfig(mm_encoder_attn_backend=_Backend.FLASH_ATTN).compute_hash()
    assert base_hash != overridden_hash
