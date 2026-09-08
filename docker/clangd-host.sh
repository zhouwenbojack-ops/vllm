#!/usr/bin/env bash
# "clangd.path": "/Users/bytedance/Codes/vllm/docker/clangd-host.sh",
# "clangd.arguments": [
#     "--compile-commands-dir=/Users/bytedance/Codes/vllm",
#     "--background-index"
# ],


set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE_DIR="${DEV_WORKSPACE_DIR:-${SCRIPT_DIR}}"
PROJECT_NAME="${SCRIPT_DIR##*/}"
PROJECT_NAME="$(printf '%s' "${PROJECT_NAME}" |
  tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_.-' '-')"
PROJECT_KEY="$(printf '%s' "${WORKSPACE_DIR}" | cksum | awk '{print $1}')"
CONTAINER_NAME="${DEV_CONTAINER_NAME:-${PROJECT_NAME}-read-${PROJECT_KEY}}"

if [[ "$(docker inspect --format '{{.State.Running}}' \
  "${CONTAINER_NAME}" 2>/dev/null || true)" != "true" ]]; then
  printf 'clangd container is not running; run ./env.sh up first\n' >&2
  exit 1
fi

exec docker exec \
  --interactive \
  --workdir "${WORKSPACE_DIR}" \
  "${CONTAINER_NAME}" \
  clangd --query-driver=/usr/bin/c++ "$@"
