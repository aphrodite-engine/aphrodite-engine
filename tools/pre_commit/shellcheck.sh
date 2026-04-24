#!/bin/bash
set -euo pipefail

scversion="stable"

if [ -d "shellcheck-${scversion}" ]; then
    shellcheck_dir="$(pwd)/shellcheck-${scversion}"
    export PATH="$PATH:${shellcheck_dir}"
fi

if ! [ -x "$(command -v shellcheck)" ]; then
    if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
        echo "Please install shellcheck: https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
        exit 1
    fi

    wget -qO- "https://github.com/koalaman/shellcheck/releases/download/${scversion?}/shellcheck-${scversion?}.linux.x86_64.tar.xz" | tar -xJv
    shellcheck_dir="$(pwd)/shellcheck-${scversion}"
    export PATH="$PATH:${shellcheck_dir}"
fi

find . -path ./.git -prune -o -name "*.sh" -print0 | \
  xargs -0 sh -c "for f in \"\$@\"; do case \"\$f\" in ./reference/*|./shellcheck-stable/*) continue ;; esac; git check-ignore -q \"\$f\" || shellcheck -s bash \"\$f\"; done" --
