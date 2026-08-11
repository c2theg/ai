#!/usr/bin/env bash
#
# Install & run SearXNG via Docker Compose.
# By: Christopher Gray / https://github.com/c2theg |  Version: 0.0.27  | Updated: 8/11/2026
#
# Install:
#   wget https://raw.githubusercontent.com/c2theg/ai/main/install_searxng.sh && chmod u+x install_searxng.sh && sudo ./install_searxng.sh
#
# Companion files pulled from the same repo (c2theg/ai, flat layout):
#   searxng_settings.yml   -> /opt/searxng/settings.yml   (required)
#   searxng_limiter.toml   -> /opt/searxng/limiter.toml   (optional, has a built-in default)
# A copy of either sitting next to this script always wins over the download.
#
# Usage:
#   sudo ./install_searxng.sh                  # fetch config from GitHub, install, start
#   sudo ./install_searxng.sh --ref v1.0.0     # pin to a tag or commit SHA
#   sudo ./install_searxng.sh --repo you/fork  # pull from a different repo
#   sudo ./install_searxng.sh --settings ./searxng_settings.yml  # use a local file
#   sudo ./install_searxng.sh --url <raw-url>  # settings from an arbitrary URL
#   sudo ./install_searxng.sh --offline        # local files only, never touch the network
#   sudo ./install_searxng.sh --port 8080 --host 0.0.0.0
#   sudo ./install_searxng.sh --no-validate    # skip the engine preflight check
#   sudo ./install_searxng.sh --uninstall      # stop & remove the stack
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via flags)
# ---------------------------------------------------------------------------
INSTALL_DIR="/opt/searxng"

# --- Where the companion files live -----------------------------------------
# The repo layout is flat, so every asset is <RAW_BASE>/<filename>.
# REF may be a branch, a tag, or a full commit SHA. Pin it to a SHA for
# reproducible installs — a branch name means you get whatever is on it today.
REPO="c2theg/ai"
REF="main"
RAW_BASE=""            # derived from REPO/REF below unless --raw-base is given

# Filenames in the repo (left) -> filenames in $INSTALL_DIR (right).
SETTINGS_ASSET="searxng_settings.yml"
LIMITER_ASSET="searxng_limiter.toml"

SETTINGS_URL=""        # full override for just the settings file (--url)
SETTINGS_SRC=""        # local settings file (--settings); wins over downloading
PORT="7042"
HOST="0.0.0.0"
IMAGE="searxng/searxng:latest"
UNINSTALL=0
VALIDATE=1
OFFLINE=0              # --offline: never reach for GitHub, local files only

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --settings)   SETTINGS_SRC="$2"; shift 2 ;;
    --url)        SETTINGS_URL="$2"; shift 2 ;;
    --repo)       REPO="$2";         shift 2 ;;
    --ref)        REF="$2";          shift 2 ;;
    --raw-base)   RAW_BASE="$2";     shift 2 ;;
    --dir)        INSTALL_DIR="$2";  shift 2 ;;
    --port)       PORT="$2";         shift 2 ;;
    --host)       HOST="$2";         shift 2 ;;
    --image)      IMAGE="$2";        shift 2 ;;
    --uninstall)  UNINSTALL=1;       shift   ;;
    --no-validate) VALIDATE=0;       shift   ;;
    --offline)    OFFLINE=1;         shift   ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -n 24
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

log()  { printf '\033[1;32m[+]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -n "$RAW_BASE" ]] || RAW_BASE="https://raw.githubusercontent.com/${REPO}/${REF}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# fetch_asset <repo-filename> <destination> <required|optional>
#
# Resolution order for every companion file:
#   1. a copy sitting next to this script (git clone / scp workflow)
#   2. <RAW_BASE>/<repo-filename>
# raw.githubusercontent.com serves through a CDN that caches for a few minutes,
# so the no-cache headers matter right after you push an update.
# Returns 1 (without dying) when an optional asset is absent.
# ---------------------------------------------------------------------------
fetch_asset() {
  local asset="$1" dest="$2" mode="${3:-required}" url tmp

  if [[ -f "$SCRIPT_DIR/$asset" ]]; then
    log "Using $asset from $SCRIPT_DIR"
    install -m 0644 "$SCRIPT_DIR/$asset" "$dest"
    return 0
  fi

  if [[ $OFFLINE -eq 1 ]]; then
    [[ "$mode" == "required" ]] && die "--offline given but $asset is not next to the script ($SCRIPT_DIR)."
    return 1
  fi

  url="$RAW_BASE/$asset"
  tmp="$(mktemp)"
  log "Downloading $asset from $url ..."
  # Deliberately not using -f: we want the HTTP status rather than curl's own
  # error text, so an optional 404 reads as a plain fallback and not a failure.
  local code
  code="$(curl -sSL -o "$tmp" -w '%{http_code}' \
            -H 'Cache-Control: no-cache' -H 'Pragma: no-cache' "$url" 2>/dev/null || true)"
  if [[ "$code" != "200" ]]; then
    rm -f "$tmp"
    if [[ "$mode" == "required" ]]; then
      die "Could not fetch $asset (HTTP ${code:-no response}) from $url
    Check that '$asset' exists on ref '$REF' of $REPO, or pass --settings/--offline."
    fi
    warn "$asset is not in $REPO@$REF (HTTP $code) — using the built-in default."
    return 1
  fi
  install -m 0644 "$tmp" "$dest"
  rm -f "$tmp"
  return 0
}

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
# Report where the config is coming from before doing any work
# ---------------------------------------------------------------------------
if [[ -n "$SETTINGS_SRC" ]]; then
  [[ -f "$SETTINGS_SRC" ]] || die "Settings file not found: $SETTINGS_SRC"
  log "Config source: local file $SETTINGS_SRC"
elif [[ -n "$SETTINGS_URL" ]]; then
  log "Config source: $SETTINGS_URL"
elif [[ -f "$SCRIPT_DIR/$SETTINGS_ASSET" ]]; then
  log "Config source: $SCRIPT_DIR/$SETTINGS_ASSET"
else
  log "Config source: $REPO @ $REF ($RAW_BASE)"
fi

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

# --- settings.yml -----------------------------------------------------------
if [[ -n "$SETTINGS_SRC" ]]; then
  log "Copying local settings into $INSTALL_DIR/settings.yml ..."
  install -m 0644 "$SETTINGS_SRC" "$INSTALL_DIR/settings.yml"
elif [[ -n "$SETTINGS_URL" ]]; then
  # --url bypasses the repo layout entirely.
  tmp="$(mktemp)"
  log "Downloading settings from $SETTINGS_URL ..."
  curl -fsSL -H 'Cache-Control: no-cache' "$SETTINGS_URL" -o "$tmp" \
    || die "Failed to download settings from $SETTINGS_URL"
  install -m 0644 "$tmp" "$INSTALL_DIR/settings.yml"
  rm -f "$tmp"
else
  fetch_asset "$SETTINGS_ASSET" "$INSTALL_DIR/settings.yml" required
fi

# Sanity-check it actually looks like a SearXNG config before using it.
grep -qE '^use_default_settings:' "$INSTALL_DIR/settings.yml" \
  || die "settings.yml does not look like a SearXNG config (no top-level use_default_settings)."

# Guard against installing a config written for an older SearXNG. These four
# engine names are the ones that broke this stack: `xml_feed`, `yahoo_finance`,
# `tradingview` and `phind` do not exist in SearXNG at all, and `reddit` was
# removed upstream in 2026-07. SearXNG logs them and starts anyway, so without
# this check a stale config looks like a successful install.
stale_engines="$(grep -oE '^[[:space:]]*engine:[[:space:]]*(xml_feed|yahoo_finance|tradingview|phind|reddit)[[:space:]]*$' \
  "$INSTALL_DIR/settings.yml" | awk '{print $2}' | sort -u | tr '\n' ' ' || true)"
if [[ -n "${stale_engines// /}" ]]; then
  die "settings.yml references engines that no longer exist: ${stale_engines}
    You are installing the old revision. Push the current searxng_settings.yml to
    $REPO (ref: $REF), or pass --settings /path/to/searxng_settings.yml."
fi

# --- limiter.toml -----------------------------------------------------------
# Optional in the repo; if it is not there, the built-in default below is used.
if ! fetch_asset "$LIMITER_ASSET" "$INSTALL_DIR/limiter.toml" optional; then
  WRITE_DEFAULT_LIMITER=1
fi

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
# limiter.toml — fallback only; used when the repo has no searxng_limiter.toml
#
# Without this file SearXNG warns "missing config file: /etc/searxng/limiter.toml"
# on every boot. trusted_proxies is emptied because this stack is published
# directly on ${HOST}:${PORT} with no reverse proxy in front — leaving the
# default 127.0.0.0/8 in the list makes botdetection expect an X-Forwarded-For
# header that nothing sets, which is the "X-Forwarded-For nor X-Real-IP header
# is set!" error. If you later front this with nginx/Traefik, put that proxy's
# network back in the list.
# ---------------------------------------------------------------------------
if [[ "${WRITE_DEFAULT_LIMITER:-0}" -eq 1 ]]; then
  log "Writing default $INSTALL_DIR/limiter.toml ..."
  cat > "$INSTALL_DIR/limiter.toml" <<'LIMITER'
[botdetection]
ipv4_prefix = 32
ipv6_prefix = 48
trusted_proxies = []

[botdetection.ip_limit]
filter_link_local = false
link_token = false

[botdetection.ip_lists]
block_ip = []
pass_ip = []
pass_searxng_org = true
LIMITER
  chmod 0644 "$INSTALL_DIR/limiter.toml"
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
      - ./limiter.toml:/etc/searxng/limiter.toml:ro
    environment:
      - SEARXNG_BASE_URL=http://localhost:${PORT}/
      # SEARXNG_REDIS_URL is deprecated upstream (logs a DeprecationWarning from
      # valkeydb.py); SearXNG reads valkey.url now.
      - SEARXNG_VALKEY_URL=valkey://redis:6379/0
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
log "Pulling images ..."
( cd "$INSTALL_DIR" && docker compose pull )

# ---------------------------------------------------------------------------
# Validate settings.yml against the engines the pulled image actually ships.
#
# SearXNG does NOT fail on a bad engine reference — it logs "can't register
# engine" per entry and starts anyway with those engines missing. Engines also
# get added and removed upstream between releases (`reddit` was dropped in
# 2026-07), so a config that worked last month can quietly lose engines after a
# `docker compose pull`. This catches that at install time instead.
# ---------------------------------------------------------------------------
if [[ $VALIDATE -eq 1 ]]; then
  log "Validating engine references against $IMAGE ..."
  if ! docker run --rm -i --entrypoint python3 \
        -v "$INSTALL_DIR/settings.yml:/tmp/settings.yml:ro" \
        "$IMAGE" - <<'PYCHECK'
import os, sys, yaml

ENGINE_DIR = "/usr/local/searxng/searx/engines"
DEFAULTS   = "/usr/local/searxng/searx/settings.yml"

modules = {f[:-3] for f in os.listdir(ENGINE_DIR) if f.endswith(".py")}
with open(DEFAULTS, encoding="utf-8") as fh:
    default_names = {e["name"] for e in (yaml.safe_load(fh).get("engines") or [])}
with open("/tmp/settings.yml", encoding="utf-8") as fh:
    user = yaml.safe_load(fh) or {}

problems = []

uds = user.get("use_default_settings")
if isinstance(uds, dict):
    for key in ("keep_only", "remove"):
        for name in ((uds.get("engines") or {}).get(key) or []):
            if name not in default_names:
                problems.append(
                    f"use_default_settings.engines.{key}: {name!r} is not an engine in the shipped defaults"
                )

user_engines = user.get("engines") or []
for entry in user_engines:
    name = entry.get("name", "<unnamed>")
    module = entry.get("engine")
    if module is None:
        # No `engine:` key means this is meant to override a default by name.
        # If no default matches, SearXNG appends it as a brand-new engine with
        # no implementation and it fails to load.
        if name not in default_names:
            problems.append(f"engine {name!r}: no 'engine:' key and no default engine by that name")
    elif module not in modules:
        problems.append(f"engine {name!r}: engine module {module!r} does not exist in this image")

if problems:
    print(f"{len(problems)} problem(s) found in settings.yml:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)

print(f"validated {len(user_engines)} engine override(s) against {len(modules)} engine modules")
PYCHECK
  then
    die "settings.yml references engines this image does not have (see above). Fix it, or re-run with --no-validate."
  fi
fi

log "Starting the stack ..."
( cd "$INSTALL_DIR" && docker compose up -d )

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
