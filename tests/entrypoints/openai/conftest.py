import pytest

from aphrodite.assets.audio import AudioAsset


@pytest.fixture
def mary_had_lamb():
    path = AudioAsset("mary_had_lamb").get_local_path()
    with open(str(path), "rb") as f:
        yield f


@pytest.fixture
def winning_call():
    path = AudioAsset("winning_call").get_local_path()
    with open(str(path), "rb") as f:
        yield f


@pytest.fixture
def foscolo():
    # Test translation it->en
    path = AudioAsset("azacinto_foscolo").get_local_path()
    with open(str(path), "rb") as f:
        yield f
