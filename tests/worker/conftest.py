import pytest


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    This module tests V0 internals, so set APHRODITE_USE_V1=0.
    """
    monkeypatch.setenv('APHRODITE_USE_V1', '0')