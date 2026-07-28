# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import html
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry-file",
        type=Path,
        required=True,
        help="Tab-separated wheel name, URL, and optional SHA-256 digest",
    )
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    links = []
    for line_number, line in enumerate(args.entry_file.read_text().splitlines(), start=1):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) not in (2, 3):
            raise ValueError(f"{args.entry_file}:{line_number}: expected 2 or 3 tab-separated fields")
        name, url = fields[:2]
        digest = fields[2] if len(fields) == 3 else ""
        if not name.endswith(".whl") or Path(name).name != name:
            raise ValueError(f"Invalid wheel name: {name!r}")
        if digest:
            url = f"{url}#sha256={digest}"
        links.append(
            f'    <a href="{html.escape(url, quote=True)}" '
            f'data-requires-python="&gt;=3.10,&lt;3.15">'
            f"{html.escape(name)}</a><br>"
        )

    if not links:
        raise ValueError(f"No wheel entries found in {args.entry_file}")
    document = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Aphrodite Engine nightly wheels</title>
  </head>
  <body>
{chr(10).join(links)}
    <!-- Latest build from {html.escape(args.commit)} -->
  </body>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)


if __name__ == "__main__":
    main()
