#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prevent direct imports of Hugging Face Hub repository APIs."""

import sys

import regex as re

HF_NAMES = (
    r"HfApi|HfFileSystem|hf_hub_download|snapshot_download"
    r"|list_repo_files|file_exists|try_to_load_from_cache"
    r"|list_repo_refs|repo_exists"
)
HF_IMPORT_RE = re.compile(
    r"^\s*from\s+huggingface_hub\s+import\s*\([^)]*\b(?:" + HF_NAMES + r")\b"
    r"|"
    r"^\s*from\s+huggingface_hub\s+import\b[^\n]*\b(?:" + HF_NAMES + r")\b",
    re.MULTILINE,
)

ALLOWED_FILES = {"aphrodite/transformers_utils/repo_utils.py"}
ALLOWED_DIRS = ("examples/",)


def scan_file(path: str) -> int:
    if path in ALLOWED_FILES or path.startswith(ALLOWED_DIRS):
        return 0

    with open(path, encoding="utf-8") as file:
        content = file.read()

    returncode = 0
    for match in HF_IMPORT_RE.finditer(content):
        line_start = content.rfind("\n", 0, match.start()) + 1
        if "#" in content[line_start : match.start()]:
            continue
        line_num = content[: match.start() + 1].count("\n") + 1
        print(
            f"{path}:{line_num}: \033[91merror:\033[0m "
            "Found direct huggingface_hub repository API import. "
            "Use the shared, Aphrodite-tagged helpers from "
            "aphrodite.transformers_utils.repo_utils instead."
        )
        returncode = 1
    return returncode


def main() -> int:
    returncode = 0
    for filename in sys.argv[1:]:
        returncode |= scan_file(filename)
    return returncode


def test_regex() -> None:
    cases = [
        ("from huggingface_hub import snapshot_download", True),
        ("from huggingface_hub import hf_hub_download", True),
        ("from huggingface_hub import HfApi", True),
        ("from huggingface_hub import HfFileSystem", True),
        ("from huggingface_hub import list_repo_files", True),
        ("from huggingface_hub import try_to_load_from_cache", True),
        ("    from huggingface_hub import snapshot_download", True),
        ("from huggingface_hub import PyTorchModelHubMixin, hf_hub_download", True),
        ("from huggingface_hub import (snapshot_download)", True),
        ("from huggingface_hub import (\n    snapshot_download,\n)", True),
        (
            "from huggingface_hub import (\n    PyTorchModelHubMixin,\n    HfApi,\n)",
            True,
        ),
        ("import huggingface_hub", False),
        ("import huggingface_hub as hf", False),
        ("from huggingface_hub import PyTorchModelHubMixin", False),
        ("from huggingface_hub.constants import HF_HUB_CACHE", False),
        ("from huggingface_hub.utils import EntryNotFoundError", False),
        ("from aphrodite.transformers_utils.repo_utils import hf_api", False),
        ("from huggingface_hub import (\n    PyTorchModelHubMixin,\n)", False),
        ("# from huggingface_hub import snapshot_download", False),
    ]
    for index, (content, should_match) in enumerate(cases):
        result = bool(HF_IMPORT_RE.search(content))
        assert result == should_match, f"case {index} failed: {content!r} (expected {should_match}, got {result})"
    print("All regex tests passed.")


if __name__ == "__main__":
    if "--test-regex" in sys.argv:
        test_regex()
    else:
        sys.exit(main())
