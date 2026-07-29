#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

output_dir="${1:?usage: build_metal_wheel.sh OUTPUT_DIR}"

mkdir -p "$output_dir"
rm -f "$output_dir"/*.whl

# uv creates a new isolated build environment for every wheel. CMake otherwise
# keeps Torch paths from the previous environment, which uv removes after the
# build. Refresh only CMake's path cache and retain incremental build outputs.
find build -type f -name CMakeCache.txt -delete 2>/dev/null || true

APHRODITE_TARGET_DEVICE=metal \
APHRODITE_REQUIRE_RUST_FRONTEND=1 \
MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}" \
  uv build --wheel --out-dir "$output_dir"

wheel="$(find "$output_dir" -maxdepth 1 -type f -name '*.whl' -print -quit)"
test -n "$wheel"

rm -rf .wheel-test
uv venv --python 3.13 .wheel-test
uv pip install --python .wheel-test/bin/python "$wheel"
.wheel-test/bin/python -c \
  "import aphrodite; from aphrodite.metal.metal import get_ops; get_ops()"
test -x .wheel-test/lib/python3.13/site-packages/aphrodite/aphrodite-rs
