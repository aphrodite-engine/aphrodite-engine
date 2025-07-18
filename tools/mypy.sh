#!/bin/bash

CI=${1:-0}
PYTHON_VERSION=${2:-local}

if [ "$CI" -eq 1 ]; then
    set -e
fi

if [ $PYTHON_VERSION == "local" ]; then
    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
fi

run_mypy() {
    echo "Running mypy on $1"
    if [ "$CI" -eq 1 ] && [ -z "$1" ]; then
        mypy --python-version "${PYTHON_VERSION}" "$@"
        return
    fi
    mypy --follow-imports skip --python-version "${PYTHON_VERSION}" "$@"
}

run_mypy # Note that this is less strict than CI
run_mypy tests
run_mypy aphrodite/attention
run_mypy aphrodite/compilation
run_mypy aphrodite/distributed
run_mypy aphrodite/engine
run_mypy aphrodite/executor
run_mypy aphrodite/inputs
run_mypy aphrodite/lora
run_mypy aphrodite/modeling
run_mypy aphrodite/plugins
run_mypy aphrodite/prompt_adapter
run_mypy aphrodite/spec_decode
run_mypy aphrodite/worker
run_mypy aphrodite/v1
run_mypy aphrodite/common