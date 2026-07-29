# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite Engine project

import argparse
import subprocess
from pathlib import Path

REPOSITORY_URL = "https://github.com/dphnAI/sonar"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def github_username(email: str) -> str | None:
    if not email.endswith("@users.noreply.github.com"):
        return None
    local_part = email.split("@", 1)[0]
    return local_part.split("+", 1)[-1] or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tag_commit = git("rev-parse", f"{args.tag}^{{commit}}")
    previous_tag = git(
        "describe",
        "--tags",
        "--abbrev=0",
        "--match",
        "v[0-9]*",
        f"{tag_commit}^",
    )
    log = git(
        "log",
        "--no-merges",
        "--format=%H%x00%s%x00%an%x00%ae",
        f"{previous_tag}..{tag_commit}",
    )

    lines = ["## What's Changed", ""]
    for entry in log.splitlines():
        commit, subject, author, email = entry.split("\0")
        username = github_username(email)
        attribution = f"@{username}" if username else author
        lines.append(f"- {subject} ([`{commit[:9]}`]({REPOSITORY_URL}/commit/{commit})) by {attribution}")

    lines.extend(
        [
            "",
            f"**Full Changelog**: {REPOSITORY_URL}/compare/{previous_tag}...{args.tag}",
            "",
        ]
    )
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
