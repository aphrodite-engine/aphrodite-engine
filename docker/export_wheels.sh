#!/bin/bash
set -e

# Script to export wheels using BuildKit cache
# This ensures that the build stages are cached and reused when exporting
# 
# Usage:
#   ./docker/export_wheels.sh                   # Export the main wheel
#   CUDA_VERSION=12.8.1 ./docker/export_wheels.sh
#
# Environment variables:
#   CUDA_VERSION      - CUDA version (default: 13.0.0)
#   TARGETPLATFORM    - Target platform (default: linux/amd64)
#   TORCH_CUDA_ARCH_LIST - CUDA arch list to compile into the wheel
#   MAX_JOBS           - Number of parallel jobs for Ninja (default: 2)
#   NVCC_THREADS       - Number of threads for nvcc (default: 8)
#   PYTHON_VERSION     - Python version in the build image (default: 3.12)

CUDA_VERSION="${CUDA_VERSION:-13.0.0}"
TARGETPLATFORM="${TARGETPLATFORM:-linux/amd64}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0 7.5 8.0 8.9 9.0 10.0 12.0}"
MAX_JOBS="${MAX_JOBS:-2}"
NVCC_THREADS="${NVCC_THREADS:-8}"

echo "Building main wheel stage for caching (if not already cached)..."
DOCKER_BUILDKIT=1 docker build \
    --target build \
    -t aphrodite-build:cache \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg TARGETPLATFORM="${TARGETPLATFORM}" \
    --build-arg torch_cuda_arch_list="${TORCH_CUDA_ARCH_LIST}" \
    --build-arg max_jobs="${MAX_JOBS}" \
    --build-arg nvcc_threads="${NVCC_THREADS}" \
    -f docker/Dockerfile . || true

echo "Exporting main wheel..."
mkdir -p ./wheels/main
DOCKER_BUILDKIT=1 docker build \
    --target main-wheel-export \
    --cache-from aphrodite-build:cache \
    --output ./wheels/main \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg TARGETPLATFORM="${TARGETPLATFORM}" \
    --build-arg torch_cuda_arch_list="${TORCH_CUDA_ARCH_LIST}" \
    --build-arg max_jobs="${MAX_JOBS}" \
    --build-arg nvcc_threads="${NVCC_THREADS}" \
    -f docker/Dockerfile .
echo "✓ Main wheel exported to ./wheels/main"

echo "Done!"
