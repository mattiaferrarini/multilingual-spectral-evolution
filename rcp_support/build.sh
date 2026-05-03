#!/bin/bash
# Build & push a CS-552 custom image. Run from this directory.
#
# Most teams should use the default course image directly and do not need
# this script. It is provided as a starting point only if your project
# genuinely needs a custom image with extra system packages or libraries.
#
# Prereq: `docker login registry.rcp.epfl.ch` with your EPFL credentials.
#
# Usage:
#   ./build.sh           # build & push :v1
#   ./build.sh v2        # build & push :v2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REGISTRY="registry.rcp.epfl.ch"
PROJECT="course-cs-552"        # confirm with RCP
IMAGE="base-vllm"
TAG="${1:-v1}"

# Pin to a specific upstream tag, never `:latest`.
VLLM_TAG="${VLLM_TAG:-v0.11.0}"

FULL="${REGISTRY}/${PROJECT}/${IMAGE}:${TAG}"

echo ">>> Building ${FULL} on top of vllm/vllm-openai:${VLLM_TAG}"
docker build \
  --pull \
  --platform linux/amd64 \
  -f "${SCRIPT_DIR}/Dockerfile" \
  --build-arg "VLLM_TAG=${VLLM_TAG}" \
  -t "${FULL}" \
  "${SCRIPT_DIR}"

echo ">>> Pushing ${FULL}"
docker push "${FULL}"

echo ">>> Done. Students pull: ${FULL}"
