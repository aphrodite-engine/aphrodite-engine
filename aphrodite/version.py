import warnings

try:
    import aphrodite.commit_id
    __commit__ = aphrodite.commit_id.__commit__
    __short_commit__ = aphrodite.commit_id.__short_commit__
except Exception as e:
    warnings.warn(f"Failed to read commit hash:\n{e}",
                  RuntimeWarning,
                  stacklevel=2)
    __commit__ = "COMMIT_HASH_PLACEHOLDER"
    __short_commit__ = "SHORT_COMMIT_HASH_PLACEHOLDER"

__version__ = "0.6.7"
