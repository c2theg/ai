#!/usr/bin/env bash
#------------------------------------------------------------
#  * Copyright (c) 2026 Christopher Gray
#  * All rights reserved.
# Version: 1.0.0
# Updated: 9/30/2026
# Install: wget https://github.com/c2theg/ai/edit/main/install_container_qdrant.sh && chmod u+x install_container_qdrant.sh
#
# Installs (or rebuilds) the latest STABLE Qdrant vector DB in Docker.
#   - storage in <storage path>/storage, snapshots in <storage path>/snapshots
#   - config in <storage path>/config/production.yaml (API key, on-disk logging)
#   - logs in /var/log/qdrant/qdrant.log
#   - iptables guard: only RFC 1918, CGNAT/Tailscale and loopback may connect
#
# Usage:  sudo ./install_qdrant.sh [options]
#   -n, --name NAME          container name   (default: DB_Qdrant0)
#   -p, --http-port PORT     REST + dashboard (default: 6333)
#   -g, --grpc-port PORT     gRPC             (default: 6334)
#   -s, --storage-dir DIR    storage path     (default: /opt/qdrant/)
#   -l, --log-dir DIR        log path         (default: /var/log/qdrant/)
#   -a, --api-key [KEY]      require an API key (random if KEY omitted; or set QDRANT_API_KEY)
#       --version VER        image tag, e.g. v1.15.4 (default: latest stable release)
#       --no-firewall        skip the iptables guard
#   -y, --yes                accept defaults / flags without prompting
#   -h, --help
#
# With no flags on a terminal, each option is prompted with its default.
#
# Docs: https://qdrant.tech/documentation/guides/installation/
#------------------------------------------------------------
set -euo pipefail

QDRANT_IMAGE="qdrant/qdrant"

NAME="DB_Qdrant0"
HTTP_PORT="6333"
GRPC_PORT="6334"
DATA_ROOT="/opt/qdrant/"
LOG_DIR="/var/log/qdrant/"
API_KEY="${QDRANT_API_KEY:-}"
AUTH="no"
TAG=""
FIREWALL="yes"
ASSUME_YES="no"

if [[ -t 1 ]]; then
  G=$'\e[32m' R=$'\e[31m' Y=$'\e[33m' B=$'\e[1m' D=$'\e[2m' X=$'\e[0m'
else
  G='' R='' Y='' B='' D='' X=''
fi
info() { echo "${B}==>${X} $*"; }
ok()   { echo "  ${G}✔${X} $*"; }
warn() { echo "${Y}!${X} $*" >&2; }
die()  { echo "${R}✖${X} $*" >&2; exit 1; }
usage() { sed -n '/^# Usage:/,/^# With no flags/p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

#------------------------------------------------------------
# Arguments
#------------------------------------------------------------
[[ -n "$API_KEY" ]] && AUTH="yes"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--name)         NAME="${2:?}"; shift 2 ;;
    -p|--http-port)    HTTP_PORT="${2:?}"; shift 2 ;;
    -g|--grpc-port)    GRPC_PORT="${2:?}"; shift 2 ;;
    -s|--storage-dir)  DATA_ROOT="${2:?}"; shift 2 ;;
    -l|--log-dir)      LOG_DIR="${2:?}"; shift 2 ;;
    -a|--api-key)      AUTH="yes"
                       if [[ $# -gt 1 && "$2" != -* ]]; then API_KEY="$2"; shift; fi
                       shift ;;
    --version)         TAG="v${2#v}"; shift 2 ;;
    --no-firewall)     FIREWALL="no"; shift ;;
    -y|--yes)          ASSUME_YES="yes"; shift ;;
    -h|--help)         usage 0 ;;
    *)                 warn "unknown option: $1"; usage 1 ;;
  esac
done

# ask VAR "Prompt" — shows the current value as the default
ask() {
  local reply
  read -r -p "  $2 [${!1}]: " reply
  [[ -n "$reply" ]] && printf -v "$1" '%s' "$reply"
  return 0
}

if [[ "$ASSUME_YES" == "no" && -t 0 ]]; then
  info "Qdrant install options (Enter keeps the default)"
  ask NAME      "Container name"
  ask HTTP_PORT "HTTP port (REST + dashboard)"
  ask GRPC_PORT "gRPC port"
  ask DATA_ROOT "Storage path"
  ask LOG_DIR   "Log path"
  echo
fi

#------------------------------------------------------------
# Validation
#------------------------------------------------------------
[[ "$(id -u)" -eq 0 ]] || die "please run as root (sudo)"

[[ "$NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || die "invalid container name: $NAME"
for p in "$HTTP_PORT" "$GRPC_PORT"; do
  [[ "$p" =~ ^[0-9]+$ ]] && (( p >= 1 && p <= 65535 )) || die "invalid port: $p"
done
[[ "$HTTP_PORT" != "$GRPC_PORT" ]] || die "HTTP and gRPC ports must differ"
[[ "$DATA_ROOT" == /* ]] || die "storage path must be absolute: $DATA_ROOT"
[[ "$LOG_DIR" == /* ]]   || die "log path must be absolute: $LOG_DIR"
DATA_ROOT="${DATA_ROOT%/}"
LOG_DIR="${LOG_DIR%/}"

if [[ "$AUTH" == "yes" ]]; then
  if [[ -z "$API_KEY" ]]; then
    API_KEY="$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 48)"
    GENERATED="yes"
  fi
  [[ "$API_KEY" =~ ^[A-Za-z0-9._~+/=-]+$ ]] || die "API key may only contain letters, digits and . _ ~ + / = -"
  (( ${#API_KEY} >= 24 )) || warn "API key is shorter than 24 characters"
fi

#------------------------------------------------------------
# Docker (installed on Ubuntu/Debian if missing)
#------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  command -v apt-get >/dev/null || die "docker is not installed (auto-install supports Ubuntu/Debian only)"
  info "Docker not found — installing Docker Engine"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y ca-certificates curl gnupg

  . /etc/os-release
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/${ID}/gpg" | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list

  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
fi
docker info >/dev/null 2>&1 || die "docker daemon is not running"
command -v curl >/dev/null || die "curl is not installed"
ok "$(docker --version)"

#------------------------------------------------------------
# Resolve the latest STABLE (non-beta) release; fall back to "latest".
#------------------------------------------------------------
if [[ -z "$TAG" ]]; then
  info "Resolving latest stable Qdrant release"
  version="$(curl -fsSL --connect-timeout 10 https://api.github.com/repos/qdrant/qdrant/releases/latest 2>/dev/null \
    | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"v\{0,1\}\([^"]*\)".*/\1/p' | head -1 || true)"
  if [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    TAG="v$version"
  else
    warn "could not resolve a clean stable tag — using 'latest'"
    TAG="latest"
  fi
fi
IMAGE="$QDRANT_IMAGE:$TAG"

echo "  ${D}name=$NAME  http=$HTTP_PORT  grpc=$GRPC_PORT  storage=$DATA_ROOT  logs=$LOG_DIR  api-key=$AUTH  image=$IMAGE${X}"
echo

#------------------------------------------------------------
# Directories
#------------------------------------------------------------
info "Creating $DATA_ROOT/{storage,snapshots,config} and $LOG_DIR"
install -d -m 0750 "$DATA_ROOT" "$DATA_ROOT/storage" "$DATA_ROOT/snapshots" "$DATA_ROOT/config" "$LOG_DIR"
ok "directories ready"

#------------------------------------------------------------
# Config — the image loads /qdrant/config/production.yaml over its defaults.
# The API key lives here (mode 600) rather than in `docker inspect` env vars.
#------------------------------------------------------------
CONF="$DATA_ROOT/config/production.yaml"
[[ -f "$CONF" ]] && cp -p "$CONF" "$CONF.bak.$(date +%Y%m%d-%H%M%S)"

info "Writing $CONF"
{
  echo "# Generated by install_qdrant.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "log_level: INFO"
  echo
  echo "logger:"
  echo "  on_disk:"
  echo "    enabled: true"
  echo "    log_file: /qdrant/logs/qdrant.log"
  echo "    log_level: INFO"
  echo "    format: text"
  echo
  echo "service:"
  echo "  host: 0.0.0.0"
  echo "  http_port: 6333"
  echo "  grpc_port: 6334"
  echo "  enable_cors: false"
  if [[ "$AUTH" == "yes" ]]; then
    echo "  api_key: \"$API_KEY\""
  fi
  echo
  echo "telemetry_disabled: true"
} > "$CONF"
chmod 0600 "$CONF" "$CONF".bak.* 2>/dev/null || true
ok "api-key=$AUTH  telemetry=off  on-disk logging=$LOG_DIR/qdrant.log"

if [[ -d /etc/logrotate.d ]]; then
  cat > /etc/logrotate.d/qdrant <<EOF
$LOG_DIR/*.log {
    weekly
    rotate 8
    maxsize 100M
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
  ok "logrotate: /etc/logrotate.d/qdrant"
fi

#------------------------------------------------------------
# Firewall — only private / Tailscale / loopback sources may reach the ports
#------------------------------------------------------------
if [[ "$FIREWALL" == "yes" ]]; then
  if command -v iptables >/dev/null && iptables -L DOCKER-USER -n >/dev/null 2>&1; then
    info "Refreshing iptables guard for ports $HTTP_PORT, $GRPC_PORT"
    CHAIN="$(tr -c 'A-Za-z0-9_\n' '_' <<<"MAG_QDRANT_$NAME" | cut -c1-28)"
    # Match on the original (pre-DNAT) host port, per Docker's DOCKER-USER guidance.
    for p in "$HTTP_PORT" "$GRPC_PORT"; do
      while iptables -D DOCKER-USER -p tcp -m conntrack --ctorigdstport "$p" --ctdir ORIGINAL -j "$CHAIN" 2>/dev/null; do :; done
    done
    iptables -F "$CHAIN" 2>/dev/null || true
    iptables -X "$CHAIN" 2>/dev/null || true

    iptables -N "$CHAIN"
    for src in 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 100.64.0.0/10 127.0.0.0/8; do
      iptables -A "$CHAIN" -s "$src" -j RETURN
    done
    iptables -A "$CHAIN" -j DROP
    for p in "$HTTP_PORT" "$GRPC_PORT"; do
      iptables -I DOCKER-USER -p tcp -m conntrack --ctorigdstport "$p" --ctdir ORIGINAL -j "$CHAIN"
    done
    ok "chain $CHAIN (not persisted — save with netfilter-persistent if desired)"
  else
    warn "iptables / DOCKER-USER chain not available — skipping firewall guard"
  fi
fi

#------------------------------------------------------------
# Container
#------------------------------------------------------------
info "Pulling $IMAGE"
docker pull -q "$IMAGE" >/dev/null
ok "$(docker image inspect -f '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null || echo "$IMAGE")"

if docker container inspect "$NAME" >/dev/null 2>&1; then
  info "Removing existing container $NAME"
  docker rm -f "$NAME" >/dev/null
  ok "removed"
fi

info "Starting $NAME"
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  -p "$HTTP_PORT:6333" \
  -p "$GRPC_PORT:6334" \
  --ulimit nofile=65535:65535 \
  --log-opt max-size=10m --log-opt max-file=3 \
  -v "$DATA_ROOT/storage:/qdrant/storage" \
  -v "$DATA_ROOT/snapshots:/qdrant/snapshots" \
  -v "$CONF:/qdrant/config/production.yaml:ro" \
  -v "$LOG_DIR:/qdrant/logs" \
  "$IMAGE" >/dev/null

# /readyz is exempt from API-key auth.
ready="no"
for _ in $(seq 1 30); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$HTTP_PORT/readyz" 2>/dev/null; then ready="yes"; break; fi
  [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" == "true" ]] || break
  sleep 1
done
if [[ "$ready" == "yes" ]]; then
  ok "$NAME is ready"
else
  warn "$NAME is not ready — check: docker logs $NAME ; tail $LOG_DIR/qdrant.log"
fi

#------------------------------------------------------------
# Done
#------------------------------------------------------------
echo
docker ps -a --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
echo
echo "  Image     : $IMAGE"
echo "  Storage   : $DATA_ROOT/storage"
echo "  Snapshots : $DATA_ROOT/snapshots"
echo "  Config    : $CONF"
echo "  Logs      : $LOG_DIR/qdrant.log   (also: docker logs -f $NAME)"
echo "  REST API  : http://<this host's IP>:$HTTP_PORT"
echo "  Dashboard : http://<this host's IP>:$HTTP_PORT/dashboard"
echo "  gRPC      : <this host's IP>:$GRPC_PORT"
echo
info "Done. Add to the gateway .env:"
echo "  QDRANT_HOST=<this host's IP>"
echo "  QDRANT_PORT=$HTTP_PORT"
if [[ "$AUTH" == "yes" ]]; then
  if [[ "${GENERATED:-no}" == "yes" ]]; then
    echo "  QDRANT_API_KEY=$API_KEY    ${Y}# generated — save it now; it is also in $CONF${X}"
  else
    echo "  QDRANT_API_KEY=<the key you provided>"
  fi
fi
