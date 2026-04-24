#!/bin/bash
set -euo pipefail

FILES=("$@")

if printf '%s\n' "${FILES[@]}" | grep -q "^docker/Dockerfile$"; then
  echo "docker/Dockerfile has changed, attempting to update dependency graph..."

  if ! command -v docker &> /dev/null; then
    echo "Warning: Docker command not found. Skipping Dockerfile graph update."
    exit 0
  fi
  if ! docker info &> /dev/null; then
    echo "Warning: Docker daemon is not running. Skipping Dockerfile graph update."
    exit 0
  fi

  TARGET_GRAPH_FILE="docs/assets/contributing/dockerfile-stages-dependency.png"
  mkdir -p "$(dirname "$TARGET_GRAPH_FILE")"

  OLD_HASH=""
  if [ -f "$TARGET_GRAPH_FILE" ]; then
    OLD_HASH=$(sha256sum "$TARGET_GRAPH_FILE")
  fi

  docker run \
    --rm \
    --user "$(id -u):$(id -g)" \
    --workdir /workspace \
    --volume "$(pwd -P)":/workspace \
    ghcr.io/patrickhoefler/dockerfilegraph:alpine \
    --output png \
    --dpi 200 \
    --max-label-length 50 \
    --filename docker/Dockerfile \
    --legend

  if [ -f "./Dockerfile.png" ]; then
    mv "./Dockerfile.png" "$TARGET_GRAPH_FILE"
  else
    DOCKERFILE_PNG=$(find . -name "Dockerfile.png" -type f | head -1)
    if [ -n "$DOCKERFILE_PNG" ]; then
      mv "$DOCKERFILE_PNG" "$TARGET_GRAPH_FILE"
    else
      echo "Error: Could not find the generated PNG file"
      find . -name "*.png" -type f -mmin -5
      exit 1
    fi
  fi

  NEW_HASH=$(sha256sum "$TARGET_GRAPH_FILE")
  if [ "$NEW_HASH" != "$OLD_HASH" ]; then
    echo "Graph has changed. Please stage the updated file: $TARGET_GRAPH_FILE"
    exit 1
  else
    echo "No changes in graph detected."
  fi
fi

exit 0
