#!/bin/bash
#------------------------------------------------------------
#  * Copyright (c) 2026 Christopher Gray
#  * All rights reserved. 
# Version: 0.0.1
# Updated: 6/14/2026
# Install: wget https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_container_qdrant.sh && chmod u+x install_container_qdrant.sh
#
# Installs the latest STABLE (non-beta) Qdrant vector DB in a
# Docker container on Ubuntu 22.04+, with persistent storage in
# the directory defined by $QDRANT_DATA_DIR below.
#
# Docs: https://qdrant.tech/documentation/guides/installation/
#------------------------------------------------------------

set -euo pipefail

#------------------------------------------------------------
# Configuration  (edit these)
#------------------------------------------------------------
QDRANT_DATA_DIR="/opt/qdrant/storage"          # host dir for collections/segments
QDRANT_SNAPSHOTS_DIR="/opt/qdrant/snapshots"   # host dir for snapshots
QDRANT_CONTAINER_NAME="qdrant"
QDRANT_HTTP_PORT="6333"                          # REST + web dashboard
QDRANT_GRPC_PORT="6334"                          # gRPC
QDRANT_IMAGE="qdrant/qdrant"
QDRANT_RESTART_POLICY="unless-stopped"
#------------------------------------------------------------

echo "

Installing Qdrant (Docker)

"

#------------------------------------------------------------
# Must run as root (apt + docker)
#------------------------------------------------------------
if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: please run as root (sudo)." >&2
  exit 1
fi

#------------------------------------------------------------
# Install Docker if it is not already present
#------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo ">> Docker not found. Installing Docker Engine..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y ca-certificates curl gnupg jq

  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg

  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list

  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
else
  echo ">> Docker already installed: $(docker --version)"
  # jq is used below to resolve the latest stable release tag.
  if ! command -v jq >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y jq curl
  fi
fi

#------------------------------------------------------------
# Resolve the latest STABLE (non-beta) Qdrant version from GitHub.
# Falls back to the "latest" Docker tag (which is also stable) if
# the API is unreachable.
#------------------------------------------------------------
echo ">> Resolving latest stable Qdrant release..."
QDRANT_VERSION="$(
  curl -fsSL https://api.github.com/repos/qdrant/qdrant/releases/latest \
    | jq -r '.tag_name' 2>/dev/null | sed 's/^v//'
)"

# Guard against beta/rc/alpha or an empty result.
if [ -z "${QDRANT_VERSION:-}" ] || echo "$QDRANT_VERSION" | grep -qiE 'beta|rc|alpha|dev'; then
  echo ">> Could not resolve a clean stable tag; using 'latest'."
  QDRANT_TAG="latest"
else
  QDRANT_TAG="v${QDRANT_VERSION}"
fi

echo ">> Using image: ${QDRANT_IMAGE}:${QDRANT_TAG}"

#------------------------------------------------------------
# Prepare persistent storage directories
#------------------------------------------------------------
echo ">> Creating data directories..."
mkdir -p "$QDRANT_DATA_DIR" "$QDRANT_SNAPSHOTS_DIR"

#------------------------------------------------------------
# Pull image and (re)create the container
#------------------------------------------------------------
echo ">> Pulling image..."
docker pull "${QDRANT_IMAGE}:${QDRANT_TAG}"

if docker ps -a --format '{{.Names}}' | grep -qx "$QDRANT_CONTAINER_NAME"; then
  echo ">> Removing existing container '$QDRANT_CONTAINER_NAME'..."
  docker rm -f "$QDRANT_CONTAINER_NAME"
fi

echo ">> Starting container..."
docker run -d \
  --name "$QDRANT_CONTAINER_NAME" \
  --restart "$QDRANT_RESTART_POLICY" \
  -p "${QDRANT_HTTP_PORT}:6333" \
  -p "${QDRANT_GRPC_PORT}:6334" \
  -v "${QDRANT_DATA_DIR}:/qdrant/storage" \
  -v "${QDRANT_SNAPSHOTS_DIR}:/qdrant/snapshots" \
  "${QDRANT_IMAGE}:${QDRANT_TAG}"

#------------------------------------------------------------
# Done
#------------------------------------------------------------
echo "

Qdrant is running.

  Container : $QDRANT_CONTAINER_NAME
  Image     : ${QDRANT_IMAGE}:${QDRANT_TAG}
  Data dir  : $QDRANT_DATA_DIR
  Snapshots : $QDRANT_SNAPSHOTS_DIR
  REST API  : http://localhost:${QDRANT_HTTP_PORT}
  Dashboard : http://localhost:${QDRANT_HTTP_PORT}/dashboard
  gRPC      : localhost:${QDRANT_GRPC_PORT}

  Logs      : docker logs -f $QDRANT_CONTAINER_NAME
  Stop      : docker stop $QDRANT_CONTAINER_NAME
  Start     : docker start $QDRANT_CONTAINER_NAME

"
