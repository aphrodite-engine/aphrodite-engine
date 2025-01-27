#!/bin/bash

set -e

# NOTE(alpin): These are the default values for my own machine.
MAX_JOBS=96
NVCC_THREADS=96

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --max_jobs) MAX_JOBS="$2"; shift ;;
        --nvcc_threads) NVCC_THREADS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

DOCKER_BUILDKIT=1 docker build -f Dockerfile . --target aphrodite-openai --tag alpindale/aphrodite-openai \
    --build-arg max_jobs=${MAX_JOBS} --build-arg nvcc_threads=${NVCC_THREADS}

commit=$(git rev-parse --short HEAD)
docker tag alpindale/aphrodite-openai alpindale/aphrodite-openai:${commit}
docker push alpindale/aphrodite-openai:${commit}
docker tag alpindale/aphrodite-openai alpindale/aphrodite-openai:latest
docker push alpindale/aphrodite-openai:latest