#!/usr/bin/env bash
# Universal Hermes Agent installer for Ubuntu 24.04 (x86_64 and arm64).
#
# Installs Hermes Agent with its supported upstream installer, then optionally
# installs or connects to one inference backend:
#   auto    Reuse a running Ollama/vLLM endpoint, otherwise install Ollama.
#   ollama  Install Ollama, set a 64K context, and pull a model.
#   vllm    Install vLLM in an isolated uv environment and create a service.
#   custom  Configure any OpenAI-compatible endpoint (SGLang, llama.cpp, TGI,
#           LM Studio, a remote vLLM/Ollama server, and so on).
#   none    Install Hermes only and leave provider setup to `hermes model`.
#
# Quick starts — local backends:
#   ./install_hermes_basic.sh
#   ./install_hermes_basic.sh --backend ollama --model qwen3.5:9b
#   ./install_hermes_basic.sh --backend vllm \
#       --model NousResearch/Hermes-3-Llama-3.1-8B
#
# Remote Ollama server:
#   ./install_hermes_basic.sh \
#       --backend ollama \
#       --ip-address 10.0.0.20 \
#       --port 11434 \
#       --model qwen3.5:9b
#
# Remote vLLM server:
#   ./install_hermes_basic.sh \
#       --backend vllm \
#       --host 10.0.0.25 \
#       --port 8000 \
#       --model NousResearch/Hermes-4-14B-FP8 \
#       --context-length 131072
#
# Planned vLLM server (10.11.1.20:8006, served model alias "primary"):
#   ./install_hermes_basic.sh \
#       --backend vllm \
#       --ip-address 10.11.1.20 \
#       --port 8006 \
#       --model primary \
#       --context-length 32767
#
# Authenticated custom OpenAI-compatible server (API key stays out of history):
#   HERMES_INSTALL_API_KEY="secret" ./install_hermes_basic.sh \
#       --backend custom \
#       --base-url https://llm.example.net/v1 \
#       --model organization/model
#
# Other custom OpenAI-compatible server:
#   ./install_hermes_basic.sh --backend custom \
#       --base-url http://gpu-server:8000/v1 --model my-model
#
# Environment equivalents are prefixed HERMES_INSTALL_, except HERMES_HOME:
#   HERMES_INSTALL_BACKEND, HERMES_INSTALL_MODEL, HERMES_INSTALL_BASE_URL,
#   HERMES_INSTALL_HOST, HERMES_INSTALL_PORT, HERMES_INSTALL_SCHEME,
#   HERMES_INSTALL_CONTEXT_LENGTH, HERMES_INSTALL_TARGET_USER,
#   HERMES_INSTALL_API_KEY, HERMES_HOME.

set -Eeuo pipefail
IFS=$'\n\t'

readonly SCRIPT_VERSION="1.1.1"
readonly HERMES_INSTALLER_URL="https://hermes-agent.nousresearch.com/install.sh"
readonly OLLAMA_INSTALLER_URL="https://ollama.com/install.sh"

BACKEND="${HERMES_INSTALL_BACKEND:-auto}"
MODEL="${HERMES_INSTALL_MODEL:-}"
BASE_URL="${HERMES_INSTALL_BASE_URL:-}"
CONTEXT_LENGTH="${HERMES_INSTALL_CONTEXT_LENGTH:-65536}"
TARGET_USER="${HERMES_INSTALL_TARGET_USER:-}"
API_KEY="${HERMES_INSTALL_API_KEY:-}"
SERVER_HOST="${HERMES_INSTALL_HOST:-${HERMES_INSTALL_IP_ADDRESS:-}}"
SERVER_PORT="${HERMES_INSTALL_PORT:-}"
SERVER_SCHEME="${HERMES_INSTALL_SCHEME:-http}"
API_PREFIX="${HERMES_INSTALL_API_PREFIX:-/v1}"

OLLAMA_HOST="${HERMES_INSTALL_OLLAMA_HOST:-127.0.0.1:11434}"
VLLM_PORT="${HERMES_INSTALL_VLLM_PORT:-8000}"
VLLM_TENSOR_PARALLEL_SIZE="${HERMES_INSTALL_VLLM_TP:-1}"
VLLM_TOOL_CALL_PARSER="${HERMES_INSTALL_VLLM_TOOL_PARSER:-hermes}"
VLLM_GPU_MEMORY_UTILIZATION="${HERMES_INSTALL_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

INSTALL_SYSTEM_PACKAGES=1
INSTALL_HERMES=1
INSTALL_BACKEND=1
PULL_MODEL=1
INSTALL_SERVICE=1
CONFIGURE_HERMES=1
RUN_DOCTOR=1
ALLOW_UNSUPPORTED=0
REMOTE_BACKEND=0
BASE_URL_EXPLICIT=0
HERMES_BRANCH="${HERMES_INSTALL_BRANCH:-main}"
HERMES_COMMIT="${HERMES_INSTALL_COMMIT:-}"

TARGET_HOME=""
TARGET_GROUP=""
HERMES_DATA_HOME=""
HERMES_BIN=""
UV_BIN=""
SYSTEMD_AVAILABLE=0
TEMP_DIR=""

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
warn() { printf '\nWARNING: %s\n' "$*" >&2; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

usage() {
    cat <<'EOF'
Usage: install_hermes_basic.sh [options]

Core options:
  --backend MODE              auto|ollama|vllm|sglang|llamacpp|openai|custom|none
                              (default: auto)
  --host HOST                 LLM server hostname or IP address
  --ip-address IP             Alias for --host
  --port PORT                 LLM server port (defaults: Ollama 11434; others 8000)
  --scheme SCHEME             http or https (default: http)
  --api-prefix PATH           API prefix (default: /v1)
  --model ID                  Model served by the selected backend
  --base-url URL              Complete OpenAI-compatible URL; overrides host/port
  --context-length TOKENS     Hermes/backend context length (default: 65536)
  --target-user USER          Account that owns Hermes config and model clients
  --api-key VALUE             API key for a custom endpoint (prefer the
                              HERMES_INSTALL_API_KEY environment variable)

vLLM options:
  --vllm-port PORT            Listening port (default: 8000)
  --tensor-parallel-size N    Number of GPUs (default: 1)
  --tool-call-parser NAME     vLLM tool parser (default: hermes)
  --gpu-memory-utilization N  vLLM GPU memory fraction (default: 0.90)

Ollama options:
  --ollama-host HOST:PORT     Bind address (default: 127.0.0.1:11434)
  --skip-model-pull           Install/configure without downloading a model

Install-control options:
  --skip-system-packages      Do not apt-install prerequisites
  --skip-hermes-install       Keep the existing Hermes installation
  --skip-backend-install      Configure an already-running backend
  --no-service                Do not create/modify persistent backend services
  --skip-config               Do not change Hermes model configuration
  --skip-doctor               Do not run `hermes doctor` at the end
  --hermes-branch NAME        Upstream Hermes git branch (default: main)
  --hermes-commit SHA         Pin the upstream Hermes checkout
  --allow-unsupported         Permit Ubuntu releases other than 24.04
  -h, --help                  Show this help

Remote-server examples:
  install_hermes_basic.sh --backend ollama --ip-address 10.0.0.20 \
    --port 11434 --model qwen3.5:9b

  install_hermes_basic.sh --backend vllm --host llm.example.net \
    --port 8000 --model NousResearch/Hermes-4-14B-FP8

  HERMES_INSTALL_API_KEY=secret install_hermes_basic.sh --backend custom \
    --base-url https://llm.example.net/v1 --model organization/model

Notes:
  * Hermes recommends at least a 64K context for reliable tool use. Smaller
    explicit values are accepted but may reduce agent/tool reliability.
  * vLLM installation requires a supported GPU and can download several GB.
    The default vLLM model generally needs a GPU with about 24 GB of VRAM for
    a 64K context. Use --model and lower vLLM memory settings as appropriate.
  * Re-run with another backend at any time; the selected backend becomes the
    active Hermes model. Existing Hermes config, skills, and sessions remain.
  * Supplying --host/--ip-address or --base-url selects an existing remote
    endpoint. It never installs, starts, or modifies software on that server.
EOF
}

cleanup() {
    if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
        rm -rf -- "$TEMP_DIR"
    fi
}
trap cleanup EXIT

need_value() {
    [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a value"
}

while (($#)); do
    case "$1" in
        --backend) need_value "$@"; BACKEND="$2"; shift 2 ;;
        --backend=*) BACKEND="${1#*=}"; shift ;;
        --host|--backend-host|--ip|--ip-address) need_value "$@"; SERVER_HOST="$2"; shift 2 ;;
        --host=*|--backend-host=*|--ip=*|--ip-address=*) SERVER_HOST="${1#*=}"; shift ;;
        --port|--backend-port) need_value "$@"; SERVER_PORT="$2"; shift 2 ;;
        --port=*|--backend-port=*) SERVER_PORT="${1#*=}"; shift ;;
        --scheme) need_value "$@"; SERVER_SCHEME="$2"; shift 2 ;;
        --scheme=*) SERVER_SCHEME="${1#*=}"; shift ;;
        --api-prefix) need_value "$@"; API_PREFIX="$2"; shift 2 ;;
        --api-prefix=*) API_PREFIX="${1#*=}"; shift ;;
        --model) need_value "$@"; MODEL="$2"; shift 2 ;;
        --model=*) MODEL="${1#*=}"; shift ;;
        --base-url) need_value "$@"; BASE_URL="$2"; shift 2 ;;
        --base-url=*) BASE_URL="${1#*=}"; shift ;;
        --context-length) need_value "$@"; CONTEXT_LENGTH="$2"; shift 2 ;;
        --context-length=*) CONTEXT_LENGTH="${1#*=}"; shift ;;
        --target-user) need_value "$@"; TARGET_USER="$2"; shift 2 ;;
        --target-user=*) TARGET_USER="${1#*=}"; shift ;;
        --api-key) need_value "$@"; API_KEY="$2"; shift 2 ;;
        --api-key=*) API_KEY="${1#*=}"; shift ;;
        --ollama-host) need_value "$@"; OLLAMA_HOST="$2"; shift 2 ;;
        --ollama-host=*) OLLAMA_HOST="${1#*=}"; shift ;;
        --vllm-port) need_value "$@"; VLLM_PORT="$2"; shift 2 ;;
        --vllm-port=*) VLLM_PORT="${1#*=}"; shift ;;
        --tensor-parallel-size) need_value "$@"; VLLM_TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --tensor-parallel-size=*) VLLM_TENSOR_PARALLEL_SIZE="${1#*=}"; shift ;;
        --tool-call-parser) need_value "$@"; VLLM_TOOL_CALL_PARSER="$2"; shift 2 ;;
        --tool-call-parser=*) VLLM_TOOL_CALL_PARSER="${1#*=}"; shift ;;
        --gpu-memory-utilization) need_value "$@"; VLLM_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --gpu-memory-utilization=*) VLLM_GPU_MEMORY_UTILIZATION="${1#*=}"; shift ;;
        --hermes-branch) need_value "$@"; HERMES_BRANCH="$2"; shift 2 ;;
        --hermes-branch=*) HERMES_BRANCH="${1#*=}"; shift ;;
        --hermes-commit) need_value "$@"; HERMES_COMMIT="$2"; shift 2 ;;
        --hermes-commit=*) HERMES_COMMIT="${1#*=}"; shift ;;
        --skip-system-packages) INSTALL_SYSTEM_PACKAGES=0; shift ;;
        --skip-hermes-install) INSTALL_HERMES=0; shift ;;
        --skip-backend-install) INSTALL_BACKEND=0; shift ;;
        --skip-model-pull) PULL_MODEL=0; shift ;;
        --no-service) INSTALL_SERVICE=0; shift ;;
        --skip-config) CONFIGURE_HERMES=0; shift ;;
        --skip-doctor) RUN_DOCTOR=0; shift ;;
        --allow-unsupported) ALLOW_UNSUPPORTED=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --version) printf '%s\n' "$SCRIPT_VERSION"; exit 0 ;;
        *) die "Unknown option: $1 (use --help)" ;;
    esac
done

as_root() {
    if [[ $EUID -eq 0 ]]; then
        "$@"
    else
        command -v sudo >/dev/null 2>&1 || die "sudo is required to install system packages/services"
        sudo -- "$@"
    fi
}

as_user() {
    if [[ "$(id -u)" == "$(id -u "$TARGET_USER")" ]]; then
        env HOME="$TARGET_HOME" USER="$TARGET_USER" LOGNAME="$TARGET_USER" HERMES_HOME="$HERMES_DATA_HOME" "$@"
    elif [[ $EUID -eq 0 ]]; then
        runuser -u "$TARGET_USER" -- env HOME="$TARGET_HOME" USER="$TARGET_USER" LOGNAME="$TARGET_USER" HERMES_HOME="$HERMES_DATA_HOME" "$@"
    else
        sudo -H -u "$TARGET_USER" -- env HOME="$TARGET_HOME" USER="$TARGET_USER" LOGNAME="$TARGET_USER" HERMES_HOME="$HERMES_DATA_HOME" "$@"
    fi
}

download() {
    local url="$1" destination="$2"
    curl --fail --silent --show-error --location \
        --retry 3 --retry-all-errors --connect-timeout 15 \
        --output "$destination" "$url"
    [[ -s "$destination" ]] || die "Downloaded file is empty: $url"
}

is_endpoint_ready() {
    local url="${1%/}"
    local args=(--fail --silent --show-error --max-time 4)
    [[ -z "$API_KEY" ]] || args+=(--header "Authorization: Bearer $API_KEY")
    curl "${args[@]}" "$url/models" >/dev/null 2>&1
}

first_endpoint_model() {
    local url="${1%/}"
    local args=(--fail --silent --show-error --max-time 5)
    [[ -z "$API_KEY" ]] || args+=(--header "Authorization: Bearer $API_KEY")
    curl "${args[@]}" "$url/models" 2>/dev/null \
        | jq -r '.data[0].id // empty' 2>/dev/null || true
}

wait_for_endpoint() {
    local url="$1" timeout="${2:-120}" elapsed=0
    while ((elapsed < timeout)); do
        if is_endpoint_ready "$url"; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

normalize_base_url() {
    local url="$1"
    url="${url%/}"
    # Add the conventional OpenAI prefix only when the caller supplied a bare
    # authority. Preserve explicit paths such as /api/openai or /v1.
    if [[ "$url" =~ ^https?://[^/]+$ ]]; then
        url="$url/v1"
    fi
    printf '%s\n' "$url"
}

prepare_requested_endpoint() {
    [[ -z "$BASE_URL" ]] || BASE_URL_EXPLICIT=1

    # Normalize friendly backend aliases while retaining them in the summary
    # where useful. They all speak an OpenAI-compatible HTTP API to Hermes.
    [[ "$BACKEND" != "llama.cpp" ]] || BACKEND="llamacpp"

    if ((BASE_URL_EXPLICIT == 1)); then
        [[ "$BACKEND" != "none" ]] || die "--base-url cannot be combined with --backend none"
        [[ "$BACKEND" != "auto" ]] || BACKEND="custom"
        REMOTE_BACKEND=1
    elif [[ -n "$SERVER_HOST" ]]; then
        [[ "$BACKEND" != "none" ]] || die "--host cannot be combined with --backend none"
        [[ "$BACKEND" != "auto" ]] || BACKEND="custom"

        if [[ -z "$SERVER_PORT" ]]; then
            if [[ "$BACKEND" == "ollama" ]]; then
                SERVER_PORT=11434
            else
                SERVER_PORT=8000
            fi
        fi

        local url_host="$SERVER_HOST"
        # URL literals require brackets around IPv6 addresses.
        if [[ "$url_host" == *:* && "$url_host" != \[*\] ]]; then
            url_host="[$url_host]"
        fi
        API_PREFIX="/${API_PREFIX#/}"
        [[ "$API_PREFIX" == "/" ]] || API_PREFIX="${API_PREFIX%/}"
        BASE_URL="${SERVER_SCHEME}://${url_host}:${SERVER_PORT}${API_PREFIX}"
        BASE_URL_EXPLICIT=1
        REMOTE_BACKEND=1
    fi

    case "$BACKEND" in
        sglang|llamacpp|openai|custom)
            REMOTE_BACKEND=1
            ;;
    esac

    if ((REMOTE_BACKEND == 1)); then
        INSTALL_BACKEND=0
        INSTALL_SERVICE=0
        PULL_MODEL=0
        [[ -n "$BASE_URL" ]] || die "--backend $BACKEND requires --host/--ip-address or --base-url"
        BASE_URL="$(normalize_base_url "$BASE_URL")"
        [[ "$BASE_URL" =~ ^https?:// ]] || die "Backend URL must begin with http:// or https://"
        log "Using existing $BACKEND server at $BASE_URL"
    fi
}

validate_inputs() {
    case "$BACKEND" in
        auto|ollama|vllm|sglang|llamacpp|llama.cpp|openai|custom|none) ;;
        *) die "Invalid backend '$BACKEND'; choose auto, ollama, vllm, sglang, llamacpp, openai, custom, or none" ;;
    esac

    [[ "$CONTEXT_LENGTH" =~ ^[1-9][0-9]*$ ]] || die "Context length must be a positive integer"
    if ((CONTEXT_LENGTH < 64000)); then
        warn "Context length $CONTEXT_LENGTH is below Hermes' recommended 64000-token minimum; continuing with the explicit override."
    fi
    [[ "$VLLM_PORT" =~ ^[1-9][0-9]*$ ]] && ((VLLM_PORT <= 65535)) \
        || die "vLLM port must be between 1 and 65535"
    [[ "$VLLM_TENSOR_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]] \
        || die "Tensor parallel size must be a positive integer"
    [[ "$VLLM_GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]*[1-9][0-9]*$|^1(\.0+)?$ ]] \
        || die "GPU memory utilization must be greater than 0 and no more than 1"
    [[ "$OLLAMA_HOST" =~ ^[A-Za-z0-9_.-]+:[1-9][0-9]*$ ]] \
        || die "--ollama-host must use HOST:PORT form (IPv4 address or hostname)"
    local ollama_port="${OLLAMA_HOST##*:}"
    ((ollama_port >= 1 && ollama_port <= 65535)) || die "Ollama port must be between 1 and 65535"
    [[ "$MODEL" != *$'\n'* ]] || die "Invalid newline in model name"
    [[ "$BASE_URL" != *$'\n'* ]] || die "Invalid newline in base URL"
    [[ "$SERVER_SCHEME" == "http" || "$SERVER_SCHEME" == "https" ]] \
        || die "--scheme must be http or https"
    if [[ -n "$SERVER_PORT" ]]; then
        [[ "$SERVER_PORT" =~ ^[1-9][0-9]*$ ]] && ((SERVER_PORT <= 65535)) \
            || die "Backend port must be between 1 and 65535"
    fi
    [[ "$SERVER_HOST" =~ ^\[?[A-Za-z0-9_.:%-]*\]?$ ]] \
        || die "Invalid backend host/IP address"
    [[ "$API_PREFIX" == /* && "$API_PREFIX" != *$'\n'* && "$API_PREFIX" != *' '* ]] \
        || die "--api-prefix must be an absolute URL path without spaces"
}

detect_platform() {
    [[ -r /etc/os-release ]] || die "Cannot identify this operating system"
    # shellcheck disable=SC1091
    source /etc/os-release
    [[ "${ID:-}" == "ubuntu" ]] || die "This installer supports Ubuntu; detected ${ID:-unknown}"
    if [[ "${VERSION_ID:-}" != "24.04" && $ALLOW_UNSUPPORTED -ne 1 ]]; then
        die "Ubuntu 24.04 is required; detected ${VERSION_ID:-unknown}. Use --allow-unsupported to continue."
    fi

    case "$(uname -m)" in
        x86_64|aarch64|arm64) ;;
        *) die "Unsupported architecture: $(uname -m)" ;;
    esac

    if command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]]; then
        SYSTEMD_AVAILABLE=1
    fi
}

resolve_target_user() {
    if [[ -z "$TARGET_USER" ]]; then
        if [[ -n "${SUDO_USER:-}" && "${SUDO_USER:-}" != "root" ]]; then
            TARGET_USER="$SUDO_USER"
        else
            TARGET_USER="$(id -un)"
        fi
    fi
    id "$TARGET_USER" >/dev/null 2>&1 || die "Target user does not exist: $TARGET_USER"
    TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
    TARGET_GROUP="$(id -gn "$TARGET_USER")"
    [[ -n "$TARGET_HOME" && -d "$TARGET_HOME" ]] || die "Cannot resolve home directory for $TARGET_USER"
    HERMES_DATA_HOME="${HERMES_HOME:-$TARGET_HOME/.hermes}"

    if [[ "$TARGET_USER" == "root" ]]; then
        warn "Installing for root. Use --target-user USER for a normal-user installation."
    fi
}

install_system_packages() {
    ((INSTALL_SYSTEM_PACKAGES == 1)) || return 0
    log "Installing Ubuntu prerequisites"
    as_root env DEBIAN_FRONTEND=noninteractive apt-get update
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates curl git jq python3 python3-venv python3-pip \
        build-essential pkg-config ripgrep ffmpeg unzip xz-utils zstd \
        procps pciutils lsof
}

check_runtime_prerequisites() {
    require_cmd curl
    require_cmd git
    require_cmd jq
    require_cmd bash
    if [[ $EUID -eq 0 && "$TARGET_USER" != "root" ]]; then
        require_cmd runuser
    fi
}

find_hermes() {
    local candidate
    for candidate in \
        "$TARGET_HOME/.local/bin/hermes" \
        "/usr/local/bin/hermes" \
        "$HERMES_DATA_HOME/venv/bin/hermes"; do
        if [[ -x "$candidate" ]]; then
            HERMES_BIN="$candidate"
            return 0
        fi
    done
    return 1
}

install_hermes() {
    if ((INSTALL_HERMES == 0)); then
        find_hermes || die "Hermes is not installed for $TARGET_USER"
        log "Keeping existing Hermes installation: $HERMES_BIN"
        return 0
    fi

    log "Downloading and running the supported Hermes Agent installer"
    local installer="$TEMP_DIR/hermes-install.sh"
    download "$HERMES_INSTALLER_URL" "$installer"
    bash -n "$installer" || die "Upstream Hermes installer failed its shell syntax check"
    chmod 0755 "$installer"

    local args=(--skip-setup --non-interactive --branch "$HERMES_BRANCH")
    [[ -z "$HERMES_COMMIT" ]] || args+=(--commit "$HERMES_COMMIT")
    as_user bash "$installer" "${args[@]}"
    find_hermes || die "Hermes installer completed, but the hermes command was not found"
    log "Hermes command installed at $HERMES_BIN"
}

detect_auto_backend() {
    [[ "$BACKEND" == "auto" ]] || return 0
    if [[ -n "$BASE_URL" ]]; then
        BACKEND="custom"
    elif is_endpoint_ready "http://127.0.0.1:11434/v1"; then
        BACKEND="ollama"
        BASE_URL="http://127.0.0.1:11434/v1"
        INSTALL_BACKEND=0
    elif is_endpoint_ready "http://127.0.0.1:8000/v1"; then
        BACKEND="vllm"
        BASE_URL="http://127.0.0.1:8000/v1"
        INSTALL_BACKEND=0
    else
        BACKEND="ollama"
    fi
    log "Auto-selected inference backend: $BACKEND"
}

start_ollama_without_systemd() {
    local ollama_bin="$1" log_file="$HERMES_DATA_HOME/logs/ollama.log"
    warn "systemd is unavailable; starting Ollama as a background process (not persistent across reboot)"
    as_user mkdir -p "$HERMES_DATA_HOME/logs"
    as_user bash -c '
        nohup env OLLAMA_HOST="$1" OLLAMA_CONTEXT_LENGTH="$2" "$3" serve \
            >>"$4" 2>&1 </dev/null &
    ' _ "$OLLAMA_HOST" "$CONTEXT_LENGTH" "$ollama_bin" "$log_file"
}

install_ollama_backend() {
    if ((INSTALL_BACKEND == 1)); then
        log "Installing/updating Ollama from its official installer"
        local installer="$TEMP_DIR/ollama-install.sh"
        download "$OLLAMA_INSTALLER_URL" "$installer"
        sh -n "$installer" || die "Upstream Ollama installer failed its shell syntax check"
        chmod 0755 "$installer"
        as_root sh "$installer"
    fi

    local ollama_bin
    ollama_bin="$(command -v ollama || true)"
    [[ -n "$ollama_bin" ]] || die "Ollama is not installed; remove --skip-backend-install and retry"

    if [[ -z "$BASE_URL" ]]; then
        local connect_host="${OLLAMA_HOST%:*}"
        [[ "$connect_host" == "0.0.0.0" || "$connect_host" == "::" || "$connect_host" == "[::]" ]] \
            && connect_host="127.0.0.1"
        BASE_URL="http://${connect_host}:${OLLAMA_HOST##*:}/v1"
    fi
    BASE_URL="$(normalize_base_url "$BASE_URL")"
    [[ -n "$MODEL" ]] || MODEL="qwen3.5:9b"

    if ((INSTALL_SERVICE == 1 && SYSTEMD_AVAILABLE == 1)); then
        log "Configuring Ollama systemd service for a ${CONTEXT_LENGTH}-token context"
        local override="$TEMP_DIR/ollama-hermes.conf"
        printf '[Service]\nEnvironment="OLLAMA_HOST=%s"\nEnvironment="OLLAMA_CONTEXT_LENGTH=%s"\n' \
            "$OLLAMA_HOST" "$CONTEXT_LENGTH" >"$override"
        as_root install -d -m 0755 /etc/systemd/system/ollama.service.d
        as_root install -m 0644 "$override" /etc/systemd/system/ollama.service.d/hermes.conf
        as_root systemctl daemon-reload
        as_root systemctl enable --now ollama.service
        as_root systemctl restart ollama.service
    elif ! is_endpoint_ready "$BASE_URL"; then
        start_ollama_without_systemd "$ollama_bin"
    fi

    if ! wait_for_endpoint "$BASE_URL" 90; then
        die "Ollama did not become ready at $BASE_URL. Check: journalctl -u ollama -n 100"
    fi

    if ((PULL_MODEL == 1)); then
        log "Pulling Ollama model: $MODEL"
        as_user env OLLAMA_HOST="${BASE_URL%/v1}" "$ollama_bin" pull "$MODEL"
    fi
}

find_uv() {
    local candidate
    for candidate in "$TARGET_HOME/.local/bin/uv" /usr/local/bin/uv /usr/bin/uv; do
        if [[ -x "$candidate" ]]; then
            UV_BIN="$candidate"
            return 0
        fi
    done
    return 1
}

start_vllm_without_systemd() {
    local vllm_bin="$1" log_file="$HERMES_DATA_HOME/logs/vllm.log"
    warn "systemd is unavailable; starting vLLM as a background process (not persistent across reboot)"
    as_user mkdir -p "$HERMES_DATA_HOME/logs"
    as_user bash -c '
        nohup "$1" serve "$2" --host 127.0.0.1 --port "$3" \
            --max-model-len "$4" --tensor-parallel-size "$5" \
            --gpu-memory-utilization "$6" --enable-auto-tool-choice \
            --tool-call-parser "$7" >>"$8" 2>&1 </dev/null &
    ' _ "$vllm_bin" "$MODEL" "$VLLM_PORT" "$CONTEXT_LENGTH" \
        "$VLLM_TENSOR_PARALLEL_SIZE" "$VLLM_GPU_MEMORY_UTILIZATION" \
        "$VLLM_TOOL_CALL_PARSER" "$log_file"
}

install_vllm_backend() {
    [[ -n "$MODEL" ]] || MODEL="NousResearch/Hermes-3-Llama-3.1-8B"
    [[ -n "$BASE_URL" ]] || BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
    BASE_URL="$(normalize_base_url "$BASE_URL")"

    local vllm_venv="$TARGET_HOME/.local/share/hermes-vllm"
    local vllm_bin="$vllm_venv/bin/vllm"

    # A server may be running in Docker or on another host without a local
    # vLLM executable. In that case Hermes only needs its OpenAI endpoint.
    if ((INSTALL_BACKEND == 0)) && is_endpoint_ready "$BASE_URL"; then
        log "Using the existing vLLM-compatible endpoint at $BASE_URL"
        return 0
    fi

    if ((INSTALL_BACKEND == 1)); then
        command -v nvidia-smi >/dev/null 2>&1 \
            || die "Automatic vLLM installation currently requires a supported NVIDIA GPU/driver. Use Ollama, or install an AMD/Intel/CPU vLLM build and rerun with --skip-backend-install."
        nvidia-smi >/dev/null 2>&1 || die "nvidia-smi exists but cannot communicate with the NVIDIA driver"
        find_uv || die "uv was not found after Hermes installation"

        log "Installing vLLM in isolated environment: $vllm_venv"
        as_user "$UV_BIN" venv "$vllm_venv" --python 3.12 --seed --managed-python
        as_user "$UV_BIN" pip install --python "$vllm_venv/bin/python" \
            --upgrade vllm --torch-backend=auto
    fi

    if [[ ! -x "$vllm_bin" ]]; then
        vllm_bin="$(command -v vllm || true)"
    fi
    [[ -n "$vllm_bin" && -x "$vllm_bin" ]] \
        || die "vLLM is not installed; remove --skip-backend-install and retry"

    # Do not replace or duplicate an already-running local endpoint.
    if is_endpoint_ready "$BASE_URL"; then
        log "Using the existing vLLM-compatible endpoint at $BASE_URL"
        return 0
    fi

    if ((INSTALL_SERVICE == 1 && SYSTEMD_AVAILABLE == 1)); then
        [[ "$MODEL" != *[[:space:]\"]* && "$VLLM_TOOL_CALL_PARSER" != *[[:space:]\"]* ]] \
            || die "vLLM service model/parser values cannot contain whitespace or quotes"
        log "Creating hermes-vllm.service"
        local service_file="$TEMP_DIR/hermes-vllm.service"
        cat >"$service_file" <<EOF
[Unit]
Description=vLLM OpenAI-compatible server for Hermes Agent
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$TARGET_USER
Group=$TARGET_GROUP
WorkingDirectory=$TARGET_HOME
Environment="HOME=$TARGET_HOME"
Environment="HF_HOME=$TARGET_HOME/.cache/huggingface"
ExecStart=$vllm_bin serve $MODEL --host 127.0.0.1 --port $VLLM_PORT --max-model-len $CONTEXT_LENGTH --tensor-parallel-size $VLLM_TENSOR_PARALLEL_SIZE --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION --enable-auto-tool-choice --tool-call-parser $VLLM_TOOL_CALL_PARSER
Restart=on-failure
RestartSec=10
TimeoutStartSec=infinity

[Install]
WantedBy=multi-user.target
EOF
        as_root install -m 0644 "$service_file" /etc/systemd/system/hermes-vllm.service
        as_root systemctl daemon-reload
        as_root systemctl enable --now hermes-vllm.service
        as_root systemctl restart hermes-vllm.service
    else
        start_vllm_without_systemd "$vllm_bin"
    fi

    log "Waiting briefly for vLLM (the first model download can take much longer)"
    if ! wait_for_endpoint "$BASE_URL" 120; then
        warn "vLLM is still starting at $BASE_URL. Follow logs with: journalctl -fu hermes-vllm"
    fi
}

configure_custom_backend() {
    [[ -n "$BASE_URL" ]] || die "--backend $BACKEND requires --host/--ip-address or --base-url"
    BASE_URL="$(normalize_base_url "$BASE_URL")"
    [[ "$BASE_URL" =~ ^https?:// ]] || die "Custom base URL must begin with http:// or https://"
    if [[ -z "$MODEL" ]]; then
        MODEL="$(first_endpoint_model "$BASE_URL")"
    fi
    [[ -n "$MODEL" ]] || die "--backend $BACKEND requires --model when the endpoint cannot auto-detect one"

    if [[ "$BACKEND" == "ollama" && $REMOTE_BACKEND -eq 1 ]]; then
        warn "The remote Ollama server must set OLLAMA_CONTEXT_LENGTH=$CONTEXT_LENGTH itself; Hermes cannot change a remote server's context setting."
    fi
}

configure_hermes() {
    ((CONFIGURE_HERMES == 1)) || return 0
    [[ "$BACKEND" != "none" ]] || return 0
    [[ -n "$HERMES_BIN" ]] || find_hermes || die "Hermes command not found"

    if [[ -z "$MODEL" ]]; then
        MODEL="$(first_endpoint_model "$BASE_URL")"
    fi
    [[ -n "$MODEL" ]] || die "No model was specified or discovered at $BASE_URL"

    log "Configuring Hermes for $MODEL at $BASE_URL"
    as_user "$HERMES_BIN" config set model.provider custom
    as_user "$HERMES_BIN" config set model.base_url "$BASE_URL"
    as_user "$HERMES_BIN" config set model.default "$MODEL"
    as_user "$HERMES_BIN" config set model.context_length "$CONTEXT_LENGTH"

    if [[ -n "$API_KEY" ]]; then
        # Hermes stores custom endpoint keys in config.yaml. Supplying the key
        # through an environment variable avoids exposing it in shell history.
        as_user "$HERMES_BIN" config set model.api_key "$API_KEY"
    elif [[ "$BACKEND" == "ollama" || "$BACKEND" == "vllm" ]]; then
        as_user "$HERMES_BIN" config set model.api_key ""
    fi
}

verify_install() {
    log "Verifying installation"
    as_user "$HERMES_BIN" --version || true

    if [[ "$BACKEND" != "none" ]]; then
        if is_endpoint_ready "$BASE_URL"; then
            local detected
            detected="$(first_endpoint_model "$BASE_URL")"
            log "Endpoint is ready${detected:+; first advertised model: $detected}"
        else
            warn "Endpoint is not ready yet: $BASE_URL"
        fi
    fi

    if ((RUN_DOCTOR == 1)); then
        as_user "$HERMES_BIN" doctor || warn "Hermes doctor reported issues; review its output above"
    fi
}

print_summary() {
    cat <<EOF

Installation complete.

  Hermes user:   $TARGET_USER
  Hermes home:   $HERMES_DATA_HOME
  Hermes command:$HERMES_BIN
  Backend:       $BACKEND
EOF
    if [[ "$BACKEND" != "none" ]]; then
        cat <<EOF
  Model:         $MODEL
  Endpoint:      $BASE_URL
  Context:       $CONTEXT_LENGTH tokens
EOF
    fi
    cat <<EOF

Start Hermes:
  $HERMES_BIN

Change providers/models later:
  $HERMES_BIN model
EOF
    if [[ "$BACKEND" == "ollama" && $SYSTEMD_AVAILABLE -eq 1 ]]; then
        printf '\nOllama logs:\n  journalctl -fu ollama\n'
    elif [[ "$BACKEND" == "vllm" && $SYSTEMD_AVAILABLE -eq 1 ]]; then
        printf '\nvLLM logs:\n  journalctl -fu hermes-vllm\n'
    fi
}

main() {
    validate_inputs
    detect_platform
    resolve_target_user
    TEMP_DIR="$(mktemp -d -t hermes-installer.XXXXXXXX)"
    # A root-run install executes the upstream Hermes script as TARGET_USER.
    # Its file is 0755, and the directory must also be traversable.
    chmod 0755 "$TEMP_DIR"

    log "Hermes Agent installer v$SCRIPT_VERSION (Ubuntu ${VERSION_ID:-unknown}, $(uname -m))"
    log "Target account: $TARGET_USER ($TARGET_HOME)"

    install_system_packages
    check_runtime_prerequisites
    install_hermes
    prepare_requested_endpoint
    detect_auto_backend

    case "$BACKEND" in
        ollama)
            if ((REMOTE_BACKEND == 1)); then configure_custom_backend; else install_ollama_backend; fi
            ;;
        vllm)
            if ((REMOTE_BACKEND == 1)); then configure_custom_backend; else install_vllm_backend; fi
            ;;
        sglang|llamacpp|openai|custom) configure_custom_backend ;;
        none) log "Skipping inference backend configuration" ;;
    esac

    configure_hermes
    verify_install
    print_summary
}

main "$@"
