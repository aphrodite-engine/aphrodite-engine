import aphrodite


def test_embedded_commit_defined():
    assert hasattr(aphrodite, "__version__")
    assert hasattr(aphrodite, "__version_tuple__")
    assert aphrodite.__version__ != "dev"
    assert aphrodite.__version_tuple__ != (0, 0, "dev")
