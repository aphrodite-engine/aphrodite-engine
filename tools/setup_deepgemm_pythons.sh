#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Provision one bare Python environment per supported CPython version and
# print their interpreter paths as a colon-separated list for CMake.
#
# Usage:
#   export DEEPGEMM_PYTHON_INTERPRETERS="$(
#     tools/setup_deepgemm_pythons.sh
#   )"
#
# Optional: DEEPGEMM_VENV_PREFIX (default: /tmp/dgenv).
set -euo pipefail

if [ "$#" -eq 0 ]; then
  pyproject="$(dirname "$0")/../pyproject.toml"
  spec="$(
    grep -E '^requires-python' "$pyproject" |
      grep -oE '>=3\.[0-9]+,<3\.[0-9]+'
  )"
  lo="${spec#>=3.}"
  lo="${lo%%,*}"
  hi="${spec##*<3.}"
  set -- $(seq "$lo" $((hi - 1)) | sed 's/^/3./')
fi

prefix="${DEEPGEMM_VENV_PREFIX:-/tmp/dgenv}"
mkdir -p "$prefix"

paths=""
for version in "$@"; do
  venv="${prefix}/${version}"
  if [ ! -x "${venv}/bin/python" ]; then
    uv venv \
      --python "$version" \
      "$venv" \
      --python-preference only-managed \
      --seed \
      >/dev/null
  fi
  paths="${paths}:${venv}/bin/python"
done

echo "${paths#:}"
