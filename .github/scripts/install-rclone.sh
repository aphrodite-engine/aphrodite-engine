#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

install_dir="${1:?usage: install-rclone.sh INSTALL_DIR}"
case "$(uname -m)" in
    x86_64|amd64) architecture="amd64" ;;
    aarch64|arm64) architecture="arm64" ;;
    *)
        echo "Unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac
archive="rclone-current-linux-${architecture}.zip"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

curl --fail --location --retry 5 --retry-all-errors \
    "https://downloads.rclone.org/${archive}" \
    --output "${tmp_dir}/${archive}"
unzip -q "${tmp_dir}/${archive}" -d "$tmp_dir"
rclone_path="$(find "$tmp_dir" -type f -path '*/rclone-*-linux-*/rclone' -print -quit)"
version_dir="$(basename "$(dirname "$rclone_path")")"
version="${version_dir#rclone-}"
version="${version%-linux-${architecture}}"

curl --fail --location --retry 5 --retry-all-errors \
    "https://downloads.rclone.org/${version}/SHA256SUMS" \
    --output "${tmp_dir}/SHA256SUMS"
versioned_archive="${version_dir}.zip"
expected="$(awk -v file="$versioned_archive" '$2 == file { print $1 }' "${tmp_dir}/SHA256SUMS")"
actual="$(sha256sum "${tmp_dir}/${archive}" | cut -d " " -f 1)"
if [[ -z "$expected" ]]; then
    echo "Checksum for ${versioned_archive} was not found in rclone SHA256SUMS" >&2
    exit 1
fi
if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch for ${archive}: expected ${expected}, got ${actual}" >&2
    exit 1
fi

mkdir -p "$install_dir"
cp "$rclone_path" "${install_dir}/rclone"
chmod +x "${install_dir}/rclone"
"${install_dir}/rclone" version
