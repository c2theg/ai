#!/usr/bin/env bash
#
# install_searxng.sh — Install & run SearXNG via Docker Compose using a
#                      local settings.yml.
#
# By: generated for Christopher Gray
# Usage:
#   sudo ./install_searxng.sh                 # install + start
#   sudo ./install_searxng.sh --settings /path/to/searxng_settings.yml
#   sudo ./install_searxng.sh --port 8080 --host 0.0.0.0
#   sudo ./install_searxng.sh --uninstall     # stop & remove the stack
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via flags)
# ---------------------------------------------------------------------------
INSTALL_DIR="/opt/searxng"
SETTINGS_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/searxng_settings.yml"
PORT="8080"
HOST="0.0.0.0"
IMAGE="searxng/searxng:latest"
UNINSTALL=0

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --settings)   SETTINGS_SRC="$2"; shift 2 ;;
    --dir)        INSTALL_DIR="$2";  shift 2 ;;
    --port)       PORT="$2";         shift 2 ;;
    --host)       HOST="$2";         shift 2 ;;
    --image)      IMAGE="$2";        shift 2 ;;
    --uninstall)  UNINSTALL=1;       shift   ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -n 14
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

log()  { printf '\033[1;32m[+]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Root check
# ---------------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
  die "Please run as root (sudo $0 ...)."
fi

# ---------------------------------------------------------------------------
# Uninstall path
# ---------------------------------------------------------------------------
if [[ $UNINSTALL -eq 1 ]]; then
  if [[ -f "$INSTALL_DIR/docker-compose.yml" ]]; then
    log "Stopping and removing SearXNG stack in $INSTALL_DIR ..."
    ( cd "$INSTALL_DIR" && docker compose down -v ) || true
  fi
  warn "Config left at $INSTALL_DIR (delete manually if desired: rm -rf $INSTALL_DIR)"
  log "Done."
  exit 0
fi

# ---------------------------------------------------------------------------
# Validate settings source
# ---------------------------------------------------------------------------
[[ -f "$SETTINGS_SRC" ]] || die "Settings file not found: $SETTINGS_SRC"
log "Using settings file: $SETTINGS_SRC"

# ---------------------------------------------------------------------------
# Install Docker + Compose plugin if missing
# ---------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  log "Docker not found — installing via get.docker.com ..."
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker 2>/dev/null || true
else
  log "Docker present: $(docker --version)"
fi

if ! docker compose version >/dev/null 2>&1; then
  die "Docker Compose plugin not available. Install 'docker-compose-plugin' and retry."
fi

# ---------------------------------------------------------------------------
# Lay down the install directory
# ---------------------------------------------------------------------------
log "Creating $INSTALL_DIR ..."
mkdir -p "$INSTALL_DIR"

# Copy settings.yml in place
install -m 0644 "$SETTINGS_SRC" "$INSTALL_DIR/settings.yml"

# ---------------------------------------------------------------------------
# Inject a fresh secret_key (never ship the committed placeholder to prod)
# ---------------------------------------------------------------------------
NEW_SECRET="$(openssl rand -hex 32)"
if grep -qE '^\s*secret_key:' "$INSTALL_DIR/settings.yml"; then
  sed -i.bak -E "s|^(\s*secret_key:).*|\1 \"${NEW_SECRET}\"|" "$INSTALL_DIR/settings.yml"
  rm -f "$INSTALL_DIR/settings.yml.bak"
  log "Generated a fresh secret_key."
else
  warn "No secret_key line found — leaving settings.yml untouched."
fi

# ---------------------------------------------------------------------------
# docker-compose.yml (SearXNG + Valkey cache)
# ---------------------------------------------------------------------------
log "Writing $INSTALL_DIR/docker-compose.yml ..."
cat > "$INSTALL_DIR/docker-compose.yml" <<COMPOSE
services:
  redis:
    container_name: searxng-valkey
    image: valkey/valkey:8-alpine
    command: valkey-server --save 30 1 --loglevel warning
    restart: unless-stopped
    networks: [searxng]
    volumes:
      - valkey-data:/data
    cap_drop: [ALL]
    cap_add: [SETGID, SETUID, DAC_OVERRIDE]

  searxng:
    container_name: searxng
    image: ${IMAGE}
    restart: unless-stopped
    depends_on: [redis]
    networks: [searxng]
    ports:
      - "${HOST}:${PORT}:8080"
    volumes:
      - ./settings.yml:/etc/searxng/settings.yml:rw
    environment:
      - SEARXNG_BASE_URL=http://localhost:${PORT}/
      - SEARXNG_REDIS_URL=redis://redis:6379/0
    cap_drop: [ALL]
    cap_add: [CHOWN, SETGID, SETUID]
    logging:
      driver: json-file
      options:
        max-size: "1m"
        max-file: "1"

networks:
  searxng:

volumes:
  valkey-data:
COMPOSE

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
log "Pulling images and starting the stack ..."
( cd "$INSTALL_DIR" && docker compose pull && docker compose up -d )

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
log "Waiting for SearXNG to answer ..."
ok=0
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then ok=1; break; fi
  sleep 2
done

echo
if [[ $ok -eq 1 ]]; then
  log "SearXNG is up: http://${HOST}:${PORT}/  (JSON API: http://${HOST}:${PORT}/search?q=test&format=json)"
else
  warn "SearXNG did not respond yet. Check logs:  cd $INSTALL_DIR && docker compose logs -f searxng"
fi

cat <<EOF

Manage the stack:
  cd $INSTALL_DIR
  docker compose ps            # status
  docker compose logs -f       # tail logs
  docker compose restart       # after editing settings.yml
  docker compose down          # stop
  sudo $0 --uninstall          # remove
EOF
