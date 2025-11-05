#!/bin/bash
# Wrapper script for aphrodite-router that sets up the library path

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${SCRIPT_DIR}/../candle_binding/target/release"

export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH}"

exec "${SCRIPT_DIR}/aphrodite-router" "$@"

