# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import zipfile
from email.parser import Parser
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("expected")
    args = parser.parse_args()

    with zipfile.ZipFile(args.wheel) as archive:
        metadata_path = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_path).decode())

    actual = metadata["Version"]
    if actual != args.expected:
        raise SystemExit(f"Wheel version {actual!r} does not match {args.expected!r}")
    print(f"Verified {args.wheel.name}: version {actual}")


if __name__ == "__main__":
    main()
