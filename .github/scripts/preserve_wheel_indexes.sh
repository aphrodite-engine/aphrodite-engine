#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

public_dir="${1:?usage: preserve_wheel_indexes.sh DOCS_PUBLIC_DIR}"
site="${SONAR_SITE_URL:-https://sonar.dphn.ai}"

preserve() {
  local relative_path=$1
  local output="${public_dir}/${relative_path}/index.html"
  mkdir -p "$(dirname "$output")"
  curl --fail --location --silent --show-error \
    "${site%/}/${relative_path}/" \
    --output "$output" ||
    rm -f "$output"
}

preserve "nightly/aphrodite-engine"
preserve "whl"
preserve "simple/aphrodite-engine"

for channel in release nightly; do
  for platform in \
    cuda/x86_64 \
    cpu/x86_64 \
    cpu/aarch64 \
    rocm/x86_64 \
    xpu/x86_64 \
    metal/aarch64; do
    preserve "whl/${channel}/${platform}"
    preserve "whl/${channel}/${platform}/simple/aphrodite-engine"
    if [[ "$channel" == "nightly" ]]; then
      preserve "nightly/${platform}"
    fi
  done
done
