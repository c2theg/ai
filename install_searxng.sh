#!/usr/bin/env bash
#
# Install & run SearXNG via Docker Compose.
# By: Christopher Gray / https://github.com/c2theg |  Version: 1.0.38  | Updated: 8/17/2026
#
# ONE-COMMAND INSTALL — installs Docker if missing, fetches every config file
# from GitHub, generates the compose stack, starts it and verifies the JSON API:
#
#   curl -fsSL https://raw.githubusercontent.com/c2theg/ai/main/install_searxng.sh | sudo bash
#
#
# Everything ends up in /opt/searxng, pulled from c2theg/ai (flat layout):
#   searxng_settings.yml -> /opt/searxng/settings.yml         (required)
#   searxng_limiter.toml -> /opt/searxng/limiter.toml         (optional, built-in default)
#   install_searxng.sh   -> /opt/searxng/install_searxng.sh   (copy of this script)
#   generated            -> docker-compose.yml, .secret_key, .install-manifest
#   ca-certificates/     -> extra CAs trusted by the container (see --ca-cert)
#   .env                 -> API keys (BRAVE_API_KEY=...), mode 0600, never committed
#
# When run from a file, a copy of an asset sitting next to this script wins over
# the download. When piped, sibling lookup is disabled and everything comes from
# the repo — re-running is safe and idempotent (the secret_key is preserved).
#
# Usage:
#   sudo ./install_searxng.sh                  # fetch config from GitHub, install, start
#   sudo ./install_searxng.sh --ref v1.0.0     # pin to a tag or commit SHA
#   sudo ./install_searxng.sh --repo you/fork  # pull from a different repo
#   sudo ./install_searxng.sh --settings ./searxng_settings.yml  # use a local file
#   sudo ./install_searxng.sh --url <raw-url>  # settings from an arbitrary URL
#   sudo ./install_searxng.sh --offline        # local files only, never touch the network
#   sudo ./install_searxng.sh --ca-cert /path/corp-ca.pem   # trust an extra CA (repeatable)
#   sudo ./install_searxng.sh --brave-key BSA...            # or put it in /opt/searxng/.env
#
# /opt/searxng/.env also carries SEARXNG_PROXIES — a comma-separated list of
# outbound proxies that SearXNG round-robins per request, to spread load across
# multiple egress points and survive per-IP rate limits and CAPTCHAs.
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

CA_DIR=""              # set from INSTALL_DIR after arg parsing
SETTINGS_URL=""        # full override for just the settings file (--url)
SETTINGS_SRC=""        # local settings file (--settings); wins over downloading
PORT="7042"
HOST="0.0.0.0"
IMAGE="searxng/searxng:latest"
UNINSTALL=0
VALIDATE=1
OFFLINE=0              # --offline: never reach for GitHub, local files only
CA_CERTS=()            # --ca-cert FILE (repeatable): extra CAs to trust
ENV_FILE=""            # $INSTALL_DIR/.env — secrets, never committed to the repo
BRAVE_API_KEY=""       # --brave-key, or BRAVE_API_KEY in .env / the environment
SEARXNG_PROXIES=""     # .env: comma/semicolon separated outbound proxy URLs
SEARXNG_LOG_SEARCHES=""  # .env: "true" logs every query to the container log

# ---------------------------------------------------------------------------
# Engines that need an API key: "<engine name>|<.env variable>|<where to get one>"
#
# The engine names match `- name:` in settings.yml. Only engines actually
# present in the deployed settings.yml are ever reported, so this table can
# safely list more than the config currently uses. Every entry here is an
# engine that ships `api_key: ""` in SearXNG's own default settings.
# ---------------------------------------------------------------------------
API_KEY_REGISTRY=(
  "braveapi|BRAVE_API_KEY|api_key||https://api.search.brave.com  (free tier ~2k queries/month)"
  "sec edgar|SEC_USER_AGENT|User-Agent|Jane Doe jane.doe@example.com|required by SEC: your real name + email, see https://www.sec.gov/os/webmaster-faq"
  "exaapi|EXA_API_KEY|api_key||https://exa.ai"
  "jina|JINA_API_KEY|api_key||https://jina.ai"
  "springer nature|SPRINGER_API_KEY|api_key||https://dev.springernature.com"
  "core.ac.uk|CORE_API_KEY|api_key||https://core.ac.uk/services/api"
  "astrophysics data system|ADS_API_KEY|api_key||https://ui.adsabs.harvard.edu/user/settings/token"
)

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
    --ca-cert)    CA_CERTS+=("$2");  shift 2 ;;
    --brave-key)  BRAVE_API_KEY="$2"; BRAVE_KEY_SOURCE="--brave-key"; shift 2 ;;
    -h|--help)
      # Print the comment header verbatim, up to the first line of code, so the
      # help text can never drift out of sync with a hard-coded line count.
      if [[ -f "${BASH_SOURCE[0]:-}" ]]; then
        awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
      else
        echo "Usage: curl -fsSL <raw-url>/install_searxng.sh | sudo bash"
        echo "Flags: --repo --ref --settings --url --dir --port --host --image"
        echo "       --offline --no-validate --uninstall"
      fi
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

log()  { printf '\033[1;32m[+]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -n "$RAW_BASE" ]] || RAW_BASE="https://raw.githubusercontent.com/${REPO}/${REF}"
CA_DIR="$INSTALL_DIR/ca-certificates"
ENV_FILE="$INSTALL_DIR/.env"

# When the script is piped (curl ... | sudo bash) there is no directory to look
# in: $0 is "bash" and dirname resolves to $PWD, which is wherever the user
# happened to be standing. Picking up a stale searxng_settings.yml from there
# would silently override the repo copy, so sibling lookup is disabled entirely
# in that mode and everything comes from GitHub.
if [[ -f "${BASH_SOURCE[0]:-}" ]]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
else
  SCRIPT_DIR=""
  SCRIPT_PATH=""
fi

# fetch_asset reports where it got a file via ASSET_ORIGIN; callers copy that
# into their own variable so errors and the closing summary can name the real
# source instead of always blaming the repo.
ASSET_ORIGIN=""
SETTINGS_ORIGIN=""
LIMITER_ORIGIN=""

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

  if [[ -n "$SCRIPT_DIR" && -f "$SCRIPT_DIR/$asset" ]]; then
    log "Using $asset from $SCRIPT_DIR"
    install -m 0644 "$SCRIPT_DIR/$asset" "$dest"
    ASSET_ORIGIN="$SCRIPT_DIR/$asset"
    return 0
  fi

  if [[ $OFFLINE -eq 1 ]]; then
    if [[ "$mode" == "required" ]]; then
      die "--offline was given but $asset is not available locally${SCRIPT_DIR:+ (looked in $SCRIPT_DIR)}."
    fi
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
  ASSET_ORIGIN="$url"
  return 0
}

# ---------------------------------------------------------------------------
# install_ca_cert <file>
#
# Normalises a CA certificate into $CA_DIR so the container's
# `update-ca-certificates` will actually pick it up. Two traps make this fail
# silently if you just copy a file in:
#
#   1. update-ca-certificates ONLY reads files matching *.crt. A perfectly good
#      "corp-ca.pem" is ignored without a word.
#   2. It expects ONE certificate per file. Hand it a bundle and the extra
#      certificates after the first are dropped.
#
# So: convert DER->PEM if needed, split bundles, and write one .crt per cert.
# ---------------------------------------------------------------------------
install_ca_cert() {
  local src="$1" stem pem tmpd count=0 out

  [[ -f "$src" ]] || die "CA certificate not found: $src"
  stem="$(basename -- "$src")"; stem="${stem%.*}"
  stem="$(printf '%s' "$stem" | tr -c 'A-Za-z0-9._-' '_')"

  tmpd="$(mktemp -d)"
  pem="$tmpd/in.pem"

  # Detect the encoding by looking for the PEM armour, NOT by letting openssl
  # guess: OpenSSL 3.x auto-detects DER even without -inform, so `openssl x509
  # -in cert.der -noout` succeeds and a naive PEM-first check misroutes every
  # DER file into the PEM branch.
  if LC_ALL=C grep -qa -- '-----BEGIN CERTIFICATE-----' "$src"; then
    cp "$src" "$pem"                                        # already PEM
  elif openssl x509 -in "$src" -inform DER -noout 2>/dev/null; then
    log "  $stem: DER encoded, converting to PEM"
    openssl x509 -in "$src" -inform DER -outform PEM -out "$pem"
  else
    rm -rf "$tmpd"
    die "Not a readable X.509 certificate: $src
    Expected PEM ('-----BEGIN CERTIFICATE-----') or DER."
  fi

  # Split into one file per certificate. LC_ALL=C keeps awk from choking on
  # multibyte sequences if anything non-UTF8 sneaks through.
  LC_ALL=C awk -v dir="$tmpd" '
    /-----BEGIN CERTIFICATE-----/ { n++ }
    n > 0 { print > sprintf("%s/cert-%02d.pem", dir, n) }
  ' "$pem"

  for f in "$tmpd"/cert-*.pem; do
    [[ -e "$f" ]] || break
    if ! openssl x509 -in "$f" -noout 2>/dev/null; then
      warn "  skipping an unparseable certificate block in $src"
      continue
    fi
    count=$((count + 1))
    if [[ $count -eq 1 ]]; then out="$CA_DIR/${stem}.crt"
    else                        out="$CA_DIR/${stem}-${count}.crt"; fi
    install -m 0644 "$f" "$out"
    log "  + $(basename "$out"): $(openssl x509 -in "$out" -noout -subject 2>/dev/null | sed 's/^subject=//' | cut -c1-70)"
  done

  rm -rf "$tmpd"
  [[ $count -gt 0 ]] || die "No certificates found in $src"
}

# ---------------------------------------------------------------------------
# load_env_file — read KEY=value pairs from $INSTALL_DIR/.env
#
# Parsed line by line rather than `source`d: this file holds secrets, and
# sourcing it would execute whatever is in it as shell. Only names we know
# about are picked up, and a value already given on the command line wins.
# ---------------------------------------------------------------------------
load_env_file() {
  local k v entry var known
  [[ -f "$ENV_FILE" ]] || return 0
  while IFS='=' read -r k v; do
    k="${k#"${k%%[![:space:]]*}"}"; k="${k%"${k##*[![:space:]]}"}"   # trim
    [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    v="${v%$'\r'}"                                                   # strip CRLF
    v="${v#\"}"; v="${v%\"}"; v="${v#\'}"; v="${v%\'}"               # unquote

    # Only variables named in the registry — plus the global settings below —
    # are honoured.
    known=0
    [[ "$k" == "SEARXNG_PROXIES" || "$k" == "SEARXNG_LOG_SEARCHES" ]] && known=1
    for entry in "${API_KEY_REGISTRY[@]}"; do
      IFS='|' read -r _ var _ _ _ <<< "$entry"
      [[ "$k" == "$var" ]] && known=1 && break
    done
    [[ $known -eq 1 ]] || continue

    # A value already set (e.g. via --brave-key) wins over the file.
    [[ -n "${!k:-}" ]] || printf -v "$k" '%s' "$v"
  done < <(grep -vE '^[[:space:]]*(#|$)' "$ENV_FILE" || true)
}

# ---------------------------------------------------------------------------
# apply_api_key <settings-file> <engine-name> <value> [yaml-field]
#
# Rewrites ONLY that engine's block: fills in the key and flips both `inactive`
# and `disabled` to false. Both matter — braveapi.init() raises "No API key
# provided" when the key is empty, and init runs even for a disabled engine, so
# flipping inactive without supplying a key breaks the boot.
# ---------------------------------------------------------------------------
apply_api_key() {
  local file="$1" engine="$2" tmp   # $3 = value, $4 = field (default api_key)
  tmp="$(mktemp)"
  # Key and engine name travel via the environment, not `awk -v`, because -v
  # applies backslash-escape processing to the value.
  API_KEY_FOR_AWK="$3" API_ENGINE_FOR_AWK="$engine" API_FIELD_FOR_AWK="${4:-api_key}" awk '
    BEGIN {
      inblk = 0; key = ENVIRON["API_KEY_FOR_AWK"]; field = ENVIRON["API_FIELD_FOR_AWK"]
      want = "- name: " ENVIRON["API_ENGINE_FOR_AWK"]; touched = 0
    }
    {
      line = $0
      sub(/^[[:space:]]+/, "", line); sub(/[[:space:]]+$/, "", line)
    }
    line == want { inblk = 1; touched = 1; print; next }
    inblk && /^[[:space:]]*-[[:space:]]*name:/ { inblk = 0 }   # next engine
    inblk && /^[^[:space:]]/                   { inblk = 0 }   # back to top level
    inblk && /^[[:space:]]+inactive:/ { print "    inactive: false"; next }
    inblk && /^[[:space:]]+disabled:/ { print "    disabled: false"; next }
    inblk && $0 ~ ("^[[:space:]]+" field ":") {
      # Preserve the original indentation: api_key sits at 4 spaces, but
      # User-Agent is nested under `headers:` at 6.
      match($0, /^[[:space:]]+/)
      print substr($0, 1, RLENGTH) field ": \"" key "\""
      next
    }
    { print }
    END { if (!touched) exit 3 }
  ' "$file" > "$tmp" || { rm -f "$tmp"; return 3; }
  cat "$tmp" > "$file"
  rm -f "$tmp"
}


# ---------------------------------------------------------------------------
# apply_proxy_pool <settings-file> <proxy-list>
#
# Rewrites the MANAGED BLOCK inside `outgoing:` with a round-robin proxy pool.
# Accepts comma- or semicolon-separated URLs. An empty list restores the block
# to its inert commented form, so removing SEARXNG_PROXIES cleanly reverts to
# direct egress on the next run.
# ---------------------------------------------------------------------------
apply_proxy_pool() {
  local file="$1" raw="$2" tmp body="" url
  local NL=$'\n'
  tmp="$(mktemp)"

  if [[ -n "$raw" ]]; then
    body="  proxies:${NL}    all://:${NL}"
    while IFS= read -r url || [[ -n "$url" ]]; do
      url="${url#"${url%%[![:space:]]*}"}"; url="${url%"${url##*[![:space:]]}"}"
      [[ -n "$url" ]] || continue
      case "$url" in
        http://*|https://*|socks4://*|socks5://*|socks5h://*) ;;
        *) rm -f "$tmp"
           die "SEARXNG_PROXIES entry is not a proxy URL: '$url'
    Expected http://, https://, socks4://, socks5:// or socks5h://" ;;
      esac
      body+="      - ${url}${NL}"
    done < <(printf '%s' "$raw" | tr ',;' '\n\n')
  fi

  PROXY_BODY_FOR_AWK="$body" awk '
    BEGIN { skip = 0; body = ENVIRON["PROXY_BODY_FOR_AWK"] }
    /^[[:space:]]*# >>> MANAGED BLOCK: outbound proxy pool/ {
      print
      if (length(body) > 0) printf "%s", body
      skip = 1; next
    }
    /^[[:space:]]*# <<< MANAGED BLOCK: outbound proxy pool/ { skip = 0; print; next }
    skip == 1 { next }
    { print }
  ' "$file" > "$tmp" || { rm -f "$tmp"; return 1; }
  cat "$tmp" > "$file"
  rm -f "$tmp"
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
# settings.yml has to stay readable by the container's non-root searxng user, so
# the secret is protected by the directory mode rather than the file mode. Bind
# mounts are resolved by the daemon as root, so 0750 here does not block Docker.
chmod 0750 "$INSTALL_DIR"

# --- settings.yml -----------------------------------------------------------
if [[ -n "$SETTINGS_SRC" ]]; then
  log "Copying local settings into $INSTALL_DIR/settings.yml ..."
  install -m 0644 "$SETTINGS_SRC" "$INSTALL_DIR/settings.yml"
  SETTINGS_ORIGIN="$SETTINGS_SRC"
elif [[ -n "$SETTINGS_URL" ]]; then
  # --url bypasses the repo layout entirely.
  tmp="$(mktemp)"
  log "Downloading settings from $SETTINGS_URL ..."
  curl -fsSL -H 'Cache-Control: no-cache' "$SETTINGS_URL" -o "$tmp" \
    || die "Failed to download settings from $SETTINGS_URL"
  install -m 0644 "$tmp" "$INSTALL_DIR/settings.yml"
  rm -f "$tmp"
  SETTINGS_ORIGIN="$SETTINGS_URL"
else
  fetch_asset "$SETTINGS_ASSET" "$INSTALL_DIR/settings.yml" required
  SETTINGS_ORIGIN="$ASSET_ORIGIN"
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
  rm -f "$INSTALL_DIR/settings.yml"
  die "This is an outdated settings.yml — it references engines that no longer exist:
      ${stale_engines}
    Source was: ${SETTINGS_ORIGIN:-unknown}
    If that is a local file, delete it (or pass --settings with the current one)
    so the copy in $REPO@$REF is used instead."
fi

# --- limiter.toml -----------------------------------------------------------
# Optional in the repo; if it is not there, the built-in default below is used.
if fetch_asset "$LIMITER_ASSET" "$INSTALL_DIR/limiter.toml" optional; then
  LIMITER_ORIGIN="$ASSET_ORIGIN"
else
  WRITE_DEFAULT_LIMITER=1
  LIMITER_ORIGIN="built-in default (no $LIMITER_ASSET in $REPO@$REF)"
fi

# ---------------------------------------------------------------------------
# Inject a fresh secret_key (never ship the committed placeholder to prod)
# ---------------------------------------------------------------------------
# The secret is kept out of band in $INSTALL_DIR/.secret_key and re-applied on
# every run. settings.yml is overwritten from the repo each time, so without
# this a re-run would mint a new key and invalidate every existing session.
SECRET_FILE="$INSTALL_DIR/.secret_key"
if [[ -s "$SECRET_FILE" ]]; then
  SECRET="$(< "$SECRET_FILE")"
  log "Reusing the existing secret_key from $SECRET_FILE"
else
  SECRET="$(openssl rand -hex 32)"
  ( umask 077; printf '%s\n' "$SECRET" > "$SECRET_FILE" )
  log "Generated a fresh secret_key."
fi
chmod 0600 "$SECRET_FILE"

if grep -qE '^[[:space:]]*secret_key:' "$INSTALL_DIR/settings.yml"; then
  sed -i.bak -E "s|^([[:space:]]*secret_key:).*|\1 \"${SECRET}\"|" "$INSTALL_DIR/settings.yml"
  rm -f "$INSTALL_DIR/settings.yml.bak"
else
  die "settings.yml has no secret_key line to replace — refusing to start with the shipped placeholder."
fi

# Verify the substitution actually landed — a silently-unreplaced placeholder
# would ship a known secret_key to production.
if grep -q 'CHANGE_ME_AT_INSTALL_TIME' "$INSTALL_DIR/settings.yml"; then
  die "Failed to inject the secret_key into settings.yml — refusing to start."
fi

# ---------------------------------------------------------------------------
# API keys from .env
#
# settings.yml is public (it lives in the git repo), so engine API keys are held
# in $INSTALL_DIR/.env and injected here at install time. SearXNG has no generic
# ${ENV_VAR} interpolation in settings.yml — only a handful of settings carry an
# environ_name, and engine api_key fields are not among them — so substitution
# is the only route.
# ---------------------------------------------------------------------------
# --- outbound proxy pool ---------------------------------------------------
# Applied before the engine keys so a failure here aborts before anything else.
PROXY_STATE="direct (no SEARXNG_PROXIES set)"

# Which key-requiring engines are actually present in the deployed config?
KEYED_ENGINES=()        # "engine|VAR|url" for engines present in settings.yml
KEYS_CONFIGURED=()      # "engine|VAR" that got a key applied
KEYS_MISSING=()         # "engine|VAR|url" still waiting for a key

for _entry in "${API_KEY_REGISTRY[@]}"; do
  IFS='|' read -r _eng _var _field _placeholder _url <<< "$_entry"
  if grep -qE "^[[:space:]]*-[[:space:]]*name:[[:space:]]*${_eng}[[:space:]]*$" "$INSTALL_DIR/settings.yml"; then
    KEYED_ENGINES+=("$_entry")
  fi
done

# Seed .env with an entry for every keyed engine in this config.
if [[ ! -f "$ENV_FILE" ]]; then
  ( umask 077
    {
      echo "# SearXNG API keys. This file is NOT in the git repo — keep it that way."
      echo "# Re-run install_searxng.sh after editing to apply changes."
      echo "#"
      echo "# Leave a value empty to keep that engine switched off."
      echo
      for _entry in "${KEYED_ENGINES[@]:-}"; do
        [[ -n "$_entry" ]] || continue
        IFS='|' read -r _eng _var _field _placeholder _url <<< "$_entry"
        echo "# ${_eng} — ${_url}"
        echo "${_var}=${_placeholder}"
        echo
      done
      echo "# Outbound proxy pool — comma or semicolon separated."
      echo "# SearXNG round-robins these per request, spreading load across"
      echo "# egress points. This is what makes rate-limited engines (yahoo"
      echo "# finance, sec edgar) and CAPTCHA-blocked ones workable again."
      echo "# Example: SEARXNG_PROXIES=http://10.0.0.5:8080,socks5://10.0.0.6:1080"
      echo "SEARXNG_PROXIES="
      echo
      echo "# Log every search query to the container log (docker logs searxng)."
      echo "# SearXNG has no query-logging setting of its own — it is privacy-first"
      echo "# and deliberately never records queries. This turns on the granian web"
      echo "# server's access log with the query string included, which is the only"
      echo "# supported way to capture them without writing a plugin."
      echo "# PRIVACY: this records what every user searches for, in plain text."
      echo "SEARXNG_LOG_SEARCHES=false"
      echo
    } > "$ENV_FILE"
  )
  log "Created $ENV_FILE"
fi
chmod 0600 "$ENV_FILE"

load_env_file

if grep -q "MANAGED BLOCK: outbound proxy pool" "$INSTALL_DIR/settings.yml"; then
  apply_proxy_pool "$INSTALL_DIR/settings.yml" "$SEARXNG_PROXIES"
  if [[ -n "$SEARXNG_PROXIES" ]]; then
    _n="$(printf '%s' "$SEARXNG_PROXIES" | tr ',;' '\n\n' | grep -c '[^[:space:]]' || true)"
    log "Outbound proxy pool: $_n endpoint(s), round-robined per request."
    PROXY_STATE="$_n proxy endpoint(s)"
  fi
elif [[ -n "$SEARXNG_PROXIES" ]]; then
  warn "SEARXNG_PROXIES is set but settings.yml has no managed proxy block — ignored."
fi

for _entry in "${KEYED_ENGINES[@]:-}"; do
  [[ -n "$_entry" ]] || continue
  IFS='|' read -r _eng _var _field _placeholder _url <<< "$_entry"
  _key="${!_var:-}"

  # A value left at the shipped example counts as NOT set. This matters most
  # for SEC_USER_AGENT: sending their example contact string to the SEC would
  # be worse than not querying them at all, so the engine stays off until the
  # operator puts their own name and address in.
  if [[ -n "$_placeholder" && "$_key" == "$_placeholder" ]]; then
    warn "$_var is still the example value — '$_eng' left disabled."
    warn "  edit $ENV_FILE and re-run to enable it."
    KEYS_MISSING+=("$_entry")
    continue
  fi

  if [[ -z "$_key" ]]; then
    KEYS_MISSING+=("$_entry")
    continue
  fi
  if apply_api_key "$INSTALL_DIR/settings.yml" "$_eng" "$_key" "$_field"; then
    grep -qF "$_key" "$INSTALL_DIR/settings.yml" \
      || die "Failed to inject $_var into settings.yml for engine '$_eng'."
    log "$_var applied — '$_eng' engine enabled."
    KEYS_CONFIGURED+=("${_eng}|${_var}")
  else
    warn "No '$_eng' entry in settings.yml — $_var not applied."
  fi
done

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
# ---------------------------------------------------------------------------
# Extra CA certificates
#
# $CA_DIR is bind-mounted at /usr/local/share/ca-certificates, which the image's
# entrypoint feeds to `update-ca-certificates` on every boot (that is the
# "Updating certificates in /etc/ssl/certs ... N added" line in the logs). This
# fixes TLS for the whole container — httpx, curl, everything — rather than just
# SearXNG, and it keeps certificate verification ON, unlike `outgoing.verify:
# false`. The directory persists across runs: drop a cert in and re-run.
# ---------------------------------------------------------------------------
mkdir -p "$CA_DIR"
chmod 0755 "$CA_DIR"

for _cert in "${CA_CERTS[@]:-}"; do
  [[ -n "$_cert" ]] || continue
  log "Installing CA certificate from $_cert ..."
  install_ca_cert "$_cert"
done

# Rescue anything dropped into the directory by hand with the wrong extension —
# update-ca-certificates would ignore it silently otherwise.
shopt -s nullglob
for _stray in "$CA_DIR"/*; do
  case "$_stray" in
    *.crt) continue ;;
  esac
  warn "$(basename "$_stray") will be ignored by update-ca-certificates (needs a .crt extension) — converting"
  install_ca_cert "$_stray" && rm -f "$_stray"
done
shopt -u nullglob

CA_COUNT="$(find "$CA_DIR" -maxdepth 1 -name '*.crt' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$CA_COUNT" -gt 0 ]]; then
  log "$CA_COUNT extra CA certificate(s) will be trusted by the container."
fi

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
# Query logging is off unless SEARXNG_LOG_SEARCHES is truthy. The default
# granian access-log format omits the query string, so the format is overridden
# to include %(query_string)s — that is what actually records the search terms.
ACCESS_LOG_ENV=""
LOG_STATE="off (searches not logged)"
# ${VAR,,} needs bash 4+; tr keeps this working on older shells too.
case "$(printf '%s' "$SEARXNG_LOG_SEARCHES" | tr '[:upper:]' '[:lower:]')" in
  true|yes|1|on)
    ACCESS_LOG_ENV=$'      - GRANIAN_LOG_ACCESS_ENABLED=true\n      - GRANIAN_LOG_ACCESS_FMT=[%(time)s] %(addr)s "%(method)s %(path)s?%(query_string)s" %(status)d %(dt_ms).1fms'
    LOG_STATE="ON — every query is written to the container log"
    warn "Search logging is ON: every query will be recorded in plain text."
    ;;
esac

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
      # Extra trusted CAs; the entrypoint runs update-ca-certificates over this
      # directory at boot. Empty is fine (reports "0 added").
      - ./ca-certificates:/usr/local/share/ca-certificates:ro
    environment:
      - SEARXNG_BASE_URL=http://localhost:${PORT}/
      # SEARXNG_REDIS_URL is deprecated upstream (logs a DeprecationWarning from
      # valkeydb.py); SearXNG reads valkey.url now.
      - SEARXNG_VALKEY_URL=valkey://redis:6379/0
${ACCESS_LOG_ENV}
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

  # The script is written to a host file and mounted in, rather than piped to
  # stdin, so the container can choose its own interpreter. PyYAML lives only in
  # SearXNG's venv (/usr/local/searxng/.venv) — the image's bare `python3` has
  # no site-packages, so `--entrypoint python3` fails with ModuleNotFoundError.
  PYCHECK_FILE="$(mktemp)"
  # mktemp gives 0600 root-only; the container runs as uid 977 and would not be
  # able to read the mounted script. It holds no secrets.
  chmod 0644 "$PYCHECK_FILE"
  cat > "$PYCHECK_FILE" <<'PYCHECK'
import os, sys

# Exit codes are meaningful to the caller:
#   0 = clean, 4 = real problems found, anything else = could not validate
try:
    import yaml
except ImportError:
    print("PyYAML not importable in this interpreter", file=sys.stderr)
    sys.exit(3)

ENGINE_DIR = "/usr/local/searxng/searx/engines"
DEFAULTS   = "/usr/local/searxng/searx/settings.yml"

if not os.path.isdir(ENGINE_DIR) or not os.path.isfile(DEFAULTS):
    print(f"unexpected image layout: {ENGINE_DIR} / {DEFAULTS} missing", file=sys.stderr)
    sys.exit(3)

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
    sys.exit(4)

print(f"validated {len(user_engines)} engine override(s) against {len(modules)} engine modules")
PYCHECK

  set +e
  docker run --rm \
    -v "$INSTALL_DIR/settings.yml:/tmp/settings.yml:ro" \
    -v "$PYCHECK_FILE:/tmp/enginecheck.py:ro" \
    --entrypoint sh "$IMAGE" -c '
      for py in /usr/local/searxng/.venv/bin/python \
                /usr/local/searxng/venv/bin/python \
                python3; do
        command -v "$py" >/dev/null 2>&1 && exec "$py" /tmp/enginecheck.py
      done
      echo "no usable python found in image" >&2
      exit 3
    '
  vrc=$?
  set -e
  rm -f "$PYCHECK_FILE"

  case $vrc in
    0) ;;
    4) die "settings.yml references engines this image does not have (listed above).
    Fix the config, or re-run with --no-validate to install anyway." ;;
    *) warn "Could not run the engine preflight (exit $vrc) — skipping it."
       warn "Install continues; check afterwards with: docker logs searxng | grep -i engine" ;;
  esac
fi

log "Starting the stack ..."
( cd "$INSTALL_DIR" && docker compose up -d )

# ---------------------------------------------------------------------------
# Make the install directory self-contained
#
# A copy of this script lands next to the config it produced, so --uninstall and
# re-runs work from $INSTALL_DIR without fetching anything. The manifest records
# what was deployed and from where — useful when a later `docker compose pull`
# changes behaviour and you need to know which ref you were on.
# ---------------------------------------------------------------------------
if [[ -n "$SCRIPT_PATH" && "$SCRIPT_PATH" != "$INSTALL_DIR/install_searxng.sh" ]]; then
  install -m 0755 "$SCRIPT_PATH" "$INSTALL_DIR/install_searxng.sh"
elif [[ -z "$SCRIPT_PATH" && $OFFLINE -eq 0 ]]; then
  # Piped run: pull a copy of ourselves so the install dir is still complete.
  curl -fsSL -H 'Cache-Control: no-cache' "$RAW_BASE/install_searxng.sh" \
    -o "$INSTALL_DIR/install_searxng.sh" 2>/dev/null \
    && chmod 0755 "$INSTALL_DIR/install_searxng.sh" || true
fi

{
  printf 'installed_at   = %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'repo           = %s\n' "$REPO"
  printf 'ref            = %s\n' "$REF"
  printf 'settings_from  = %s\n' "${SETTINGS_ORIGIN:-unknown}"
  printf 'image          = %s\n' "$IMAGE"
  printf 'listen         = %s:%s\n' "$HOST" "$PORT"
  printf 'settings_sha256= %s\n' "$(sha256sum "$INSTALL_DIR/settings.yml" | awk '{print $1}')"
  printf 'limiter_sha256 = %s\n' "$(sha256sum "$INSTALL_DIR/limiter.toml" | awk '{print $1}')"
} > "$INSTALL_DIR/.install-manifest"
chmod 0644 "$INSTALL_DIR/.install-manifest"

# ---------------------------------------------------------------------------
# Health check — poll /healthz, NOT /search.
#
# Polling the search endpoint means every install fires a real metasearch query
# at every upstream engine, repeatedly, in a tight loop. That is a good way to
# earn a CAPTCHA from DuckDuckGo/Startpage/Qwant and a 429 from Brave, which
# then look like config faults. /healthz answers from SearXNG itself and touches
# no upstream. The single real query happens once, in the smoke test below.
# ---------------------------------------------------------------------------
log "Waiting for SearXNG to answer ..."
ok=0
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    ok=1; break
  fi
  sleep 2
done

# Surface any engine that failed to register, even on a successful start —
# SearXNG logs these and carries on, which is how the previous config looked
# healthy while most of it was dead. Name them rather than counting: the count
# alone says nothing about whether it is benign.
# Log line: "... ERROR:searx.engines: (PID 42) <name>: can't register engine ..."
failed_engines="$(docker logs searxng 2>&1 \
  | sed -nE "s/.*\(PID [0-9]+\) (.+): can.t register engine.*/\1/p" \
  | sort -u | tr '\n' ' ' || true)"

echo
if [[ $ok -eq 1 ]]; then
  log "SearXNG is up."
else
  warn "SearXNG did not answer within 60s."
fi

cat <<EOF

  URL            http://${HOST}:${PORT}/
  JSON API       http://${HOST}:${PORT}/search?q=<query>&format=json
  Install dir    $INSTALL_DIR
    settings.yml       <- ${SETTINGS_ORIGIN:-unknown}
    limiter.toml       <- ${LIMITER_ORIGIN:-unknown}
    ca-certificates/     ${CA_COUNT:-0} extra CA(s) trusted by the container
    .env                 API keys + proxy pool (mode 0600, not in the repo)
    egress               ${PROXY_STATE:-direct}
    search logging       ${LOG_STATE:-off}
    docker-compose.yml   generated
    .secret_key          generated, preserved across re-runs
    .install-manifest    what was deployed, and from where

  OpenWebUI -> Admin Settings -> Web Search
    Engine: searxng
    Query URL: http://${HOST}:${PORT}/search?q=<query>&format=json
EOF

# ---------------------------------------------------------------------------
# API key guidance — always tell the operator exactly how to set any key this
# config can use, whether or not it is already set.
# ---------------------------------------------------------------------------
if [[ -n "${KEYED_ENGINES[*]:-}" ]]; then
  echo
  echo "  API keys — set these in $ENV_FILE (mode 0600, never committed):"
  echo
  for _entry in "${KEYED_ENGINES[@]}"; do
    IFS='|' read -r _eng _var _field _placeholder _url <<< "$_entry"
    _v="${!_var:-}"
    if [[ -n "$_v" && ( -z "$_placeholder" || "$_v" != "$_placeholder" ) ]]; then
      printf '    [set]     %-24s %s\n' "$_var" "-> '$_eng' enabled"
    elif [[ -n "$_placeholder" && "$_v" == "$_placeholder" ]]; then
      printf '    [example] %-24s %s\n' "$_var" "-> '$_eng' OFF (still the sample value)"
      printf '              %s\n' "$_url"
    else
      printf '    [not set] %-24s %s\n' "$_var" "-> '$_eng' stays off"
      printf '              %s\n' "$_url"
    fi
  done
  if [[ -n "${KEYS_MISSING[*]:-}" ]]; then
    cat <<KEYHELP

    To enable one:
      sudo nano $ENV_FILE          # set VAR=your-key
      sudo $INSTALL_DIR/install_searxng.sh   # re-run to apply

    Or in a single command:
      sudo $INSTALL_DIR/install_searxng.sh --brave-key <key>

    The key is written into $INSTALL_DIR/settings.yml at install time and is
    never stored in the git repo. Re-running always re-applies it from .env.
KEYHELP
  fi
fi


if [[ -n "${failed_engines// /}" ]]; then
  echo
  warn "Engines that failed to register: ${failed_engines}"
  warn "  'ahmia' and 'torch' are Tor-only and always fail without a Tor proxy — ignore those."
  warn "  Anything else is worth a look:  docker logs searxng | grep -i engine"
fi

# ---------------------------------------------------------------------------
# Smoke test — run a real query through the JSON API and report what came back.
# A 200 response is not proof of a working install: SearXNG answers happily with
# zero results if every engine is failing, which is exactly the failure mode
# this whole config rewrite was about.
# ---------------------------------------------------------------------------
if [[ $ok -eq 1 ]]; then
  echo
  log "Running a test search ..."
  TEST_JSON="$(curl -fsS --max-time 45 --get \
      --data-urlencode 'q=anthropic claude' \
      --data-urlencode 'format=json' \
      "http://127.0.0.1:${PORT}/search" 2>/dev/null || true)"

  if [[ -z "$TEST_JSON" ]]; then
    warn "The test search returned nothing. Try it by hand:"
    warn "  curl -s 'http://127.0.0.1:${PORT}/search?q=test&format=json' | head -c 400"
  elif command -v python3 >/dev/null 2>&1; then
    # The response travels in an env var, not on stdin, because stdin is already
    # carrying the program itself via the heredoc. That also keeps the Python
    # free of shell-quoting hazards.
    SEARXNG_TEST_JSON="$TEST_JSON" python3 - <<'PYSMOKE'
import json, os

raw = os.environ.get("SEARXNG_TEST_JSON", "")
try:
    d = json.loads(raw)
except Exception as exc:
    print(f"  could not parse the JSON response: {exc}")
    raise SystemExit(0)

results = d.get("results", [])
engines = sorted({e for r in results for e in (r.get("engines") or [])})
dead    = d.get("unresponsive_engines") or []

print(f"  results returned : {len(results)}")
line = f"  engines answering: {len(engines)}"
if engines:
    line += "  (" + ", ".join(engines[:12]) + ")"
print(line)

for r in results[:3]:
    print("    - " + (r.get("title") or "").strip()[:68])

if dead:
    print(f"  engines erroring : {len(dead)}")
    for entry in dead[:8]:
        if isinstance(entry, (list, tuple)):
            pair = list(entry) + ["", ""]
            name, reason = pair[0], pair[1]
        else:
            name, reason = entry, ""
        print(f"    ! {name}: {reason}")

if not results:
    print("  NO RESULTS — every engine failed or was filtered out.")
    print("  Check:  docker logs searxng | grep -iE 'engine|error'")
PYSMOKE
  else
    # python3 absent: fall back to a crude but dependency-free count.
    hits="$(printf '%s' "$TEST_JSON" | grep -o '"url"' | wc -l | tr -d ' ')"
    log "  test search returned roughly $hits result URLs"
  fi

  echo
  echo "  Test it yourself any time:"
  echo "    curl -s 'http://127.0.0.1:${PORT}/search?q=test&format=json' | python3 -c 'import json,sys; print(len(json.load(sys.stdin)[\"results\"]), \"results\")'"
  echo "  Which engines are erroring, and why:"
  echo "    http://${HOST}:${PORT}/stats/errors"
fi

# ---------------------------------------------------------------------------
# Five copy-paste checks covering the endpoints that actually matter.
# ---------------------------------------------------------------------------
cat <<TESTS

  Five ways to test it:

  1) Liveness — answers from SearXNG itself, touches no upstream engine:
     curl -fsS http://127.0.0.1:${PORT}/healthz && echo OK

  2) JSON API — the exact call OpenWebUI makes; prints the result count:
     curl -fsS 'http://127.0.0.1:${PORT}/search?q=test&format=json' \\
       | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["results"]), "results")'

  3) Which engines answered, and which failed (the useful one):
     curl -fsS 'http://127.0.0.1:${PORT}/search?q=anthropic+claude&format=json' | python3 -c '
     import json,sys
     from collections import Counter
     d=json.load(sys.stdin)
     c=Counter(e for r in d["results"] for e in (r.get("engines") or []))
     print("answered :", ", ".join(f"{k}({v})" for k,v in c.most_common()) or "NONE")
     for e in (d.get("unresponsive_engines") or []): print("  failed :", e[0], "-", e[1])'

  4) A specific category (swap crypto for finance/tech/news/sports/blogs):
     curl -fsS 'http://127.0.0.1:${PORT}/search?q=bitcoin&categories=crypto&format=json' \\
       | python3 -c 'import json,sys; d=json.load(sys.stdin); [print(" -", r["title"][:60], r["url"][:50]) for r in d["results"][:5]]'

  5) One engine on its own, using its bang shortcut — proves an engine works
     in isolation (!yf yahoo finance, !cg coingecko, !sec edgar, !mwm mwmbl):
     curl -fsS --get --data-urlencode 'q=!cg ethereum' --data-urlencode 'format=json' \\
       http://127.0.0.1:${PORT}/search \\
       | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d["results"]),"results"); print(d.get("unresponsive_engines") or "no errors")'

  Engine health over time:  http://${HOST}:${PORT}/stats/errors
TESTS


if [[ $ok -ne 1 ]]; then
  echo
  warn "Check the logs:  cd $INSTALL_DIR && docker compose logs -f searxng"
  exit 1
fi


cat <<EOF

  Manage the stack:
    cd $INSTALL_DIR
    docker compose ps                          # status
    docker compose logs -f searxng             # tail logs
    docker compose restart searxng             # after editing settings.yml
    docker compose down                        # stop
    sudo $INSTALL_DIR/install_searxng.sh            # re-pull config & restart
    sudo $INSTALL_DIR/install_searxng.sh --uninstall # remove
EOF
