#!/bin/bash
#------------------------------------------------------------
#  * Copyright (c) 2026 Christopher Gray
#  * All rights reserved.
# Version: 0.0.5
# Updated: 6/14/2026
# Install: wget https://github.com/c2theg/ai/edit/main/install_container_mongodb.sh && chmod u+x install_container_mongodb.sh
#
# Installs the latest STABLE (non-rc/beta) MongoDB Community
# server in a Docker container on Ubuntu 22.04+, with persistent
# storage in the directory defined by $MONGO_DATA_DIR below.
#
# Docs: https://www.mongodb.com/docs/manual/installation/
#
# mkdir -p "/media/data/sync/configs/containers/mongodb-cluster/"
# mkdir -p "/media/data/containers/mongodb/ai_c0"
# mkdir -p "/var/log/mongodb"
#------------------------------------------------------------

set -euo pipefail

#------------------------------------------------------------
# Configuration  (edit these)
#------------------------------------------------------------
MONGO_CONTAINER_NAME="Mongo_DB_AI"
MONGO_DATA_DIR="/media/data/containers/mongodb/ai_c0"   # host dir for /data/db
MONGO_CONFIG_FILE="/media/data/sync/configs/containers/mongodb-cluster/ai_mongo.conf"
MONGO_LOG_DIR="/var/log/mongodb"
MONGO_PORT="27020"                       # host:container port (must match mongo.conf)
MONGO_IMAGE="mongo"
MONGO_RESTART_POLICY="always"
MONGO_MEMORY="64g"                       # hard memory limit
MONGO_MEMORY_RESERVATION="512m"          # soft memory reservation
MONGO_ULIMIT_NOFILE="64000:64000"
# WiredTiger cache cap (GB). Passed as a mongod CLI flag, which OVERRIDES the
# cacheSizeGB in the .conf file. Export MONGO_WT_CACHE_GB to override; else 4.
MONGO_WT_CACHE_GB="${MONGO_WT_CACHE_GB:-4}"
MONGO_USE_HOST_NETWORK="false"           # "true" => --network=host (ignores MONGO_PORT)
#------------------------------------------------------------

echo "

Installing MongoDB (Docker)

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
# Resolve the latest STABLE (non-rc/beta) MongoDB version from
# Docker Hub. Keeps only plain X.Y.Z tags (drops -rc, -beta, etc).
# Falls back to the "latest" Docker tag if the API is unreachable.
#------------------------------------------------------------
echo ">> Resolving latest stable MongoDB release..."
MONGO_VERSION="$(
  curl -fsSL "https://hub.docker.com/v2/repositories/library/${MONGO_IMAGE}/tags?page_size=100&ordering=last_updated" \
    | jq -r '.results[].name' 2>/dev/null \
    | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' \
    | sort -V \
    | tail -1
)"

if [ -z "${MONGO_VERSION:-}" ]; then
  echo ">> Could not resolve a clean stable tag; using 'latest'."
  MONGO_TAG="latest"
else
  MONGO_TAG="${MONGO_VERSION}"
fi

echo ">> Using image: ${MONGO_IMAGE}:${MONGO_TAG}"

#------------------------------------------------------------
# Prepare persistent storage / log directories
#------------------------------------------------------------
echo ">> Creating data and log directories..."
mkdir -p "$MONGO_DATA_DIR" "$MONGO_LOG_DIR"

if [ ! -f "$MONGO_CONFIG_FILE" ]; then
  echo "WARNING: config file not found: $MONGO_CONFIG_FILE" >&2
  echo "         Create it before the container will start cleanly." >&2
fi

#------------------------------------------------------------
# Pull image and (re)create the container
#------------------------------------------------------------
echo ">> Pulling image..."
docker pull "${MONGO_IMAGE}:${MONGO_TAG}"

if docker ps -a --format '{{.Names}}' | grep -qx "$MONGO_CONTAINER_NAME"; then
  echo ">> Removing existing container '$MONGO_CONTAINER_NAME'..."
  docker rm -f "$MONGO_CONTAINER_NAME"
fi

# Choose networking mode.
if [ "$MONGO_USE_HOST_NETWORK" = "true" ]; then
  NET_ARGS=(--network=host)
else
  NET_ARGS=(-p "${MONGO_PORT}:${MONGO_PORT}")
fi

echo ">> Starting container..."
docker run -d \
  --name "$MONGO_CONTAINER_NAME" \
  "${NET_ARGS[@]}" \
  --restart "$MONGO_RESTART_POLICY" \
  --ulimit "nofile=${MONGO_ULIMIT_NOFILE}" \
  --memory "$MONGO_MEMORY" \
  --memory-reservation "$MONGO_MEMORY_RESERVATION" \
  -v "${MONGO_CONFIG_FILE}:/etc/mongo.conf" \
  -v "${MONGO_LOG_DIR}/:/var/log/mongodb/" \
  -v "${MONGO_DATA_DIR}/:/data/db" \
  "${MONGO_IMAGE}:${MONGO_TAG}" --config /etc/mongo.conf \
  --wiredTigerCacheSizeGB "${MONGO_WT_CACHE_GB}"

#------------------------------------------------------------
# Done
#------------------------------------------------------------
echo "

MongoDB is running.

  Container : $MONGO_CONTAINER_NAME
  Image     : ${MONGO_IMAGE}:${MONGO_TAG}
  Data dir  : $MONGO_DATA_DIR
  Config    : $MONGO_CONFIG_FILE
  WT cache  : ${MONGO_WT_CACHE_GB} GB (CLI override of conf file)
  Log dir   : $MONGO_LOG_DIR
  Port      : ${MONGO_PORT} (or host network if enabled)

  Connect   : mongosh --port ${MONGO_PORT}
  Logs      : docker logs -f $MONGO_CONTAINER_NAME
  Stop      : docker stop $MONGO_CONTAINER_NAME
  Start     : docker start $MONGO_CONTAINER_NAME

"
