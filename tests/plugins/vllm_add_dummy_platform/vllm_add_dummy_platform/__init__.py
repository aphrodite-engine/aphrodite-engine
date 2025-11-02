def dummy_platform_plugin() -> str | None:
    return "aphrodite_add_dummy_platform.dummy_platform.DummyPlatform"


def register_ops():
    import aphrodite_add_dummy_platform.dummy_custom_ops  # noqa
