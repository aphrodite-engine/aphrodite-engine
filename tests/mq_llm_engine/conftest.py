import pytest


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    Since this module is V0 only, set APHRODITE_USE_V1=0 for
    all tests in the module.
    """
    monkeypatch.setenv('APHRODITE_USE_V1', '0')
