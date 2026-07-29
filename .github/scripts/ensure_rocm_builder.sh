#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

control_host="${DOCKER_HOST:-tcp://127.0.0.1:23750}"
host_control="${ROCM_HOST_DOCKER_HOST:-unix:///var/run/docker.sock}"
dind_container="${ROCM_DIND_CONTAINER:-aphrodite-nightly-docker}"
builder_name="${ROCM_BUILDER_NAME:-sonar-rocm}"
container_name="${ROCM_BUILDKIT_CONTAINER:-sonar-rocm-buildkit}"
cache_size="${ROCM_BUILDKIT_CACHE_SIZE:-300g}"
builder_port="${ROCM_BUILDKIT_PORT:-12351}"
dind_address="$(
  docker --host "$host_control" inspect --format \
    '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' \
    "$dind_container"
)"
test -n "$dind_address"
builder_addr="tcp://${dind_address}:${builder_port}"

container_args="$(
  docker --host "$control_host" inspect \
    --format '{{json .Config.Cmd}}' "$container_name" 2>/dev/null || true
)"
container_mounts="$(
  docker --host "$control_host" inspect \
    --format '{{json .HostConfig.Tmpfs}}' "$container_name" 2>/dev/null || true
)"
container_ports="$(
  docker --host "$control_host" inspect \
    --format '{{json .HostConfig.PortBindings}}' "$container_name" \
    2>/dev/null || true
)"
if [[ -n "$container_args" ]] &&
  { [[ "$container_args" != *"tcp://0.0.0.0:1234"* ]] ||
    [[ "$container_mounts" != *'"/var/lib/buildkit"'* ]] ||
    [[ "$container_ports" != *"${builder_port}"* ]]; }; then
  docker --host "$control_host" rm --force "$container_name" >/dev/null
  container_args=""
fi

if [[ -z "$container_args" ]]; then
  docker --host "$control_host" run --detach \
    --name "$container_name" \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --publish "${builder_port}:1234" \
    --tmpfs "/var/lib/buildkit:rw,size=${cache_size},mode=1777" \
    moby/buildkit:rootless \
    --oci-worker-no-process-sandbox \
    --addr tcp://0.0.0.0:1234
elif [ "$(docker --host "$control_host" inspect -f '{{.State.Running}}' "$container_name")" != "true" ]; then
  docker --host "$control_host" start "$container_name" >/dev/null
fi

configured_endpoint="$(
  docker buildx inspect "$builder_name" 2>/dev/null |
    sed -n 's/^Endpoint:[[:space:]]*//p'
)"
if [[ -n "$configured_endpoint" && "$configured_endpoint" != "$builder_addr" ]]; then
  docker buildx rm "$builder_name" >/dev/null
  configured_endpoint=""
fi

if [[ -z "$configured_endpoint" ]]; then
  docker buildx create \
    --name "$builder_name" \
    --driver remote \
    "$builder_addr"
fi
docker buildx use "$builder_name"

for _ in $(seq 1 60); do
  if docker buildx inspect --bootstrap >/dev/null 2>&1; then
    exit 0
  fi
  sleep 1
done

docker --host "$control_host" logs --tail 100 "$container_name" >&2 || true
echo "ROCm BuildKit builder did not become ready at $builder_addr." >&2
exit 1
