# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import html
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    digest = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    name = args.wheel.name
    url = f"{args.url}#sha256={digest}"
    document = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Aphrodite Engine nightly wheels</title>
  </head>
  <body>
    <a href="{html.escape(url, quote=True)}" data-requires-python="&gt;=3.10,&lt;3.15">{html.escape(name)}</a>
    <!-- Built from {html.escape(args.commit)} -->
  </body>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)


if __name__ == "__main__":
    main()
