#!/bin/bash
set -e

# Script to export wheels using BuildKit cache
# This ensures that the build stages are cached and reused when exporting
# 
# Usage:
#   ./docker/export_wheels.sh                    # Export both wheels
#   ./docker/export_wheels.sh kernels           # Export only kernels wheel
#   ./docker/export_wheels.sh main               # Export only main wheel
#   ./docker/export_wheels.sh kernels main      # Export both (explicit)
#
# Environment variables:
#   CUDA_VERSION      - CUDA version (default: 12.9.1)
#   TARGETPLATFORM    - Target platform (default: linux/amd64)
#   MAX_JOBS           - Number of parallel jobs for Ninja (default: 2)
#   NVCC_THREADS       - Number of threads for nvcc (default: 8)
#
# Example:
#   MAX_JOBS=4 NVCC_THREADS=16 ./docker/export_wheels.sh kernels

CUDA_VERSION="${CUDA_VERSION:-12.9.1}"
TARGETPLATFORM="${TARGETPLATFORM:-linux/amd64}"
MAX_JOBS="${MAX_JOBS:-2}"
NVCC_THREADS="${NVCC_THREADS:-8}"
EXPORT_KERNELS="${1:-true}"
EXPORT_MAIN="${2:-true}"

# If no args provided, export both
if [ "$1" = "kernels" ]; then
    EXPORT_KERNELS=true
    EXPORT_MAIN=false
elif [ "$1" = "main" ]; then
    EXPORT_KERNELS=false
    EXPORT_MAIN=true
fi

if [ "$EXPORT_KERNELS" = "true" ]; then
    echo "Building kernels stage for caching (if not already cached)..."
    DOCKER_BUILDKIT=1 docker build \
        --target build-kernels \
        -t aphrodite-kernels:cache \
        --build-arg CUDA_VERSION="${CUDA_VERSION}" \
        --build-arg TARGETPLATFORM="${TARGETPLATFORM}" \
        --build-arg max_jobs="${MAX_JOBS}" \
        --build-arg nvcc_threads="${NVCC_THREADS}" \
        -f docker/Dockerfile . || true
    
    echo "Exporting kernels wheel..."
    DOCKER_BUILDKIT=1 docker build \
        --target kernels-wheel-export \
        --cache-from aphrodite-kernels:cache \
        --output ./wheels/kernels \
        --build-arg CUDA_VERSION="${CUDA_VERSION}" \
        --build-arg TARGETPLATFORM="${TARGETPLATFORM}" \
        --build-arg max_jobs="${MAX_JOBS}" \
        --build-arg nvcc_threads="${NVCC_THREADS}" \
        -f docker/Dockerfile .
    echo "✓ Kernels wheel exported to ./wheels/kernels"
fi

if [ "$EXPORT_MAIN" = "true" ]; then
    echo "Building main wheel stage for caching (if not already cached)..."
    DOCKER_BUILDKIT=1 docker build \
        --target build \
        -t aphrodite-build:cache \
        --build-arg CUDA_VERSION="${CUDA_VERSION}" \
        --build-arg TARGETPLATFORM="${TARGETPLATFORM}" \
        --build-arg max_jobs="${MAX_JOBS}" \
        --build-arg nvcc_threads="${NVCC_THREADS}" \
        -f docker/Dockerfile . || true
    
    echo "Exporting main wheel..."
    DOCKER_BUILDKIT=1 docker build \
        --target main-wheel-export \
        --cache-from aphrodite-build:cache \
        --output ./wheels/main \
        --build-arg CUDA_VERSION="${CUDA_VERSION}" \
        --build-arg TARGETPLATFORM="${TARGETPLATFORM}" \
        --build-arg max_jobs="${MAX_JOBS}" \
        --build-arg nvcc_threads="${NVCC_THREADS}" \
        -f docker/Dockerfile .
    echo "✓ Main wheel exported to ./wheels/main"
fi

echo "Done!"

