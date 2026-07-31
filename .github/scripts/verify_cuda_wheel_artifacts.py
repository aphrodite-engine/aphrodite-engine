#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify native artifacts required by a CUDA wheel."""

from __future__ import annotations

import argparse
import stat
import sys
import zipfile
from pathlib import Path

import tomllib


def supported_python_minors(pyproject: Path) -> list[int]:
    with pyproject.open("rb") as file:
        spec = tomllib.load(file)["project"]["requires-python"]
    bounds = spec.removeprefix(">=3.").split(",<3.")
    if len(bounds) != 2 or not all(bound.isdigit() for bound in bounds):
        raise ValueError(f"Cannot parse requires-python from {pyproject}")
    lower, upper = map(int, bounds)
    return list(range(lower, upper))


def verify(wheel: Path, pyproject: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        entries = {entry.filename: entry for entry in archive.infolist()}

    missing: list[str] = []
    rust_binary = "aphrodite/aphrodite-rs"
    if rust_binary not in entries:
        missing.append(rust_binary)
    else:
        mode = entries[rust_binary].external_attr >> 16
        if not mode & stat.S_IXUSR:
            missing.append(f"{rust_binary} (not executable)")

    if not any(
        name.startswith("aphrodite/_rust_tool_parser")
        and name.endswith(".so")
        and "/" not in name.removeprefix("aphrodite/")
        for name in entries
    ):
        missing.append("aphrodite/_rust_tool_parser*.so")

    for minor in supported_python_minors(pyproject):
        prefix = f"aphrodite/third_party/deep_gemm/_C.cpython-3{minor}-"
        if not any(name.startswith(prefix) and name.endswith(".so") for name in entries):
            missing.append(f"{prefix}*.so")

    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise RuntimeError(f"{wheel} is missing required native artifacts:\n{formatted}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()
    verify(args.wheel, args.pyproject)
    print(f"Verified CUDA wheel native artifacts: {args.wheel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
