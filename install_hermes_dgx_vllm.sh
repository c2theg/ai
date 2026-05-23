#!/usr/bin/env bash
# Hermes Agent — DGX Spark Install Script
# Installs Hermes Agent backed by a local vllm server.
# Detects any already-running vllm instance and uses its loaded model.
# If nothing is running, picks the best Hermes model for available VRAM
# and starts vllm as a systemd service.

set -euo pipefail

# ─── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*" | tee -a "$LOG_FILE"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" | tee -a "$LOG_FILE"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"; }
log_step()  { echo -e "\n${BLUE}${BOLD}━━━ $* ━━━${NC}" | tee -a "$LOG_FILE"; }
log_ok()    { echo -e "${GREEN}${BOLD}✓${NC} $*" | tee -a "$LOG_FILE"; }

# ─── config ───────────────────────────────────────────────────────────────────
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-hermes-local}"
VLLM_SERVICE_NAME="hermes-vllm"
HERMES_DIR="${HERMES_DIR:-$HOME/.hermes}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
LOG_FILE="/tmp/hermes_install_$(date +%Y%m%d_%H%M%S).log"
VLLM_LOG="/var/log/${VLLM_SERVICE_NAME}.log"
MIN_PYTHON_MINOR=11

# Ports to probe for a running vllm instance (checked in order)
VLLM_PROBE_PORTS=(8000 8080 8001 8888 9000)

# ─── model catalogue ──────────────────────────────────────────────────────────
# Format: "hf_repo_id|min_vram_gb|quant_flag|context_length|description"
# Ordered best-first; first model that fits available VRAM wins.
MODELS=(
    "NousResearch/Hermes-3-Llama-3.1-70B-FP8|70|fp8|32768|Hermes-3 70B FP8 (flagship)"
    "NousResearch/Hermes-3-Llama-3.1-70B|140||32768|Hermes-3 70B BF16"
    "NousResearch/Hermes-3-Llama-3.2-11B-Vision-Instruct|22||32768|Hermes-3 11B Vision"
    "NousResearch/Hermes-3-Llama-3.1-8B|16||32768|Hermes-3 8B"
    "NousResearch/Hermes-3-Llama-3.2-3B|8||32768|Hermes-3 3B (fallback)"
)

# Populated during detection / selection
VLLM_RUNNING=false       # true if a live vllm server was found
SELECTED_MODEL=""
SELECTED_QUANT=""
SELECTED_CTX=""
SELECTED_DESC=""
NGPUS=0
VRAM_GB=0
TP=1

# ─── helpers ──────────────────────────────────────────────────────────────────
die() { log_error "$*"; exit 1; }

require_cmd() { command -v "$1" &>/dev/null || die "Required command '$1' not found. $2"; }

total_vram_gb() {
    local mib
    mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
          | awk '{s+=$1} END {print int(s)}')
    echo $(( mib / 1024 ))
}

gpu_count() { nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l; }

tensor_parallel() {
    local count tp=1
    count=$(gpu_count)
    while (( tp * 2 <= count )); do tp=$(( tp * 2 )); done
    echo "$tp"
}

# Returns 0 if HF model is present in local cache
model_cached() {
    local repo="$1"
    local safe="${repo//\//__}"
    [[ -d "${HF_CACHE}/models--${safe/\//_}/snapshots" ]] \
        || [[ -d "${HF_CACHE}/models--${safe}/snapshots" ]]
}

# Query /v1/models on a given host:port and echo the first model id, or nothing
query_vllm_models() {
    local host="$1" port="$2"
    # Try with and without API key; vllm may or may not require one
    local url="http://${host}:${port}/v1/models"
    local resp
    resp=$(curl -sf --max-time 3 "$url" \
                -H "Authorization: Bearer ${VLLM_API_KEY}" 2>/dev/null \
           || curl -sf --max-time 3 "$url" 2>/dev/null \
           || true)
    [[ -z "$resp" ]] && return 1
    # Extract first model id using Python (jq may not be installed)
    $PYTHON - "$resp" <<'PYEOF'
import sys, json
try:
    data = json.loads(sys.argv[1])
    models = data.get("data", [])
    if models:
        print(models[0]["id"])
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
PYEOF
}

# ─── step 0: prerequisites ────────────────────────────────────────────────────
check_prerequisites() {
    log_step "Checking prerequisites"

    [[ "$(uname -s)" == "Linux" ]] || die "This script targets Linux (DGX Spark runs Ubuntu)."

    require_cmd nvidia-smi "Install NVIDIA drivers for the DGX Spark."
    require_cmd curl "sudo apt install curl"
    require_cmd git  "sudo apt install git"

    # Resolve Python 3.11+
    local py=""
    for candidate in python3.12 python3.11 python3; do
        if command -v "$candidate" &>/dev/null; then
            local major minor
            major=$("$candidate" -c "import sys; print(sys.version_info.major)")
            minor=$("$candidate" -c "import sys; print(sys.version_info.minor)")
            if (( major == 3 && minor >= MIN_PYTHON_MINOR )); then
                py="$candidate"; break
            fi
        fi
    done
    [[ -n "$py" ]] || die "Python 3.${MIN_PYTHON_MINOR}+ required. Install via 'sudo apt install python3.11'."
    PYTHON="$py"
    log_ok "Python: $($PYTHON --version)"

    # Verify vllm is importable (user already has it installed)
    if $PYTHON -c "import vllm" 2>/dev/null; then
        local vver
        vver=$($PYTHON -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
        log_ok "vllm: $vver"
    else
        die "vllm is not importable by $PYTHON. Check your environment / virtualenv."
    fi

    log_ok "Prerequisites satisfied."
}

# ─── step 1: detect GPU resources ────────────────────────────────────────────
detect_gpu() {
    log_step "Detecting GPU resources"

    NGPUS=$(gpu_count)
    VRAM_GB=$(total_vram_gb)
    TP=$(tensor_parallel)

    log_info "GPUs        : $NGPUS"
    log_info "Free VRAM   : ${VRAM_GB} GiB (combined)"
    log_info "Tensor par. : $TP"

    (( VRAM_GB >= 8 )) || die "Less than 8 GiB free GPU memory."
}

# ─── step 2: detect already-running vllm ─────────────────────────────────────
detect_running_vllm() {
    log_step "Detecting running vllm server"

    # Build probe list: configured port first, then the rest of the common ports
    local probe_ports=("$VLLM_PORT")
    for p in "${VLLM_PROBE_PORTS[@]}"; do
        [[ "$p" != "$VLLM_PORT" ]] && probe_ports+=("$p")
    done

    for port in "${probe_ports[@]}"; do
        log_info "Probing http://${VLLM_HOST}:${port}/v1/models …"
        local model_id
        if model_id=$(query_vllm_models "$VLLM_HOST" "$port" 2>/dev/null) \
           && [[ -n "$model_id" ]]; then
            VLLM_RUNNING=true
            VLLM_PORT="$port"
            SELECTED_MODEL="$model_id"
            SELECTED_DESC="$model_id (already running)"
            SELECTED_QUANT=""
            SELECTED_CTX="32768"
            log_ok "Found running vllm on port $port — model: $model_id"
            return 0
        fi
    done

    log_info "No running vllm server found — will start one."
}

# ─── step 3: select model (skipped if vllm already running) ──────────────────
select_model() {
    "$VLLM_RUNNING" && return 0

    log_step "Selecting Hermes model"

    local best_repo="" best_quant="" best_ctx="" best_desc=""
    local cached_repo="" cached_quant="" cached_ctx="" cached_desc=""

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r repo min_vram quant ctx desc <<< "$entry"
        if (( VRAM_GB >= min_vram )); then
            [[ -z "$best_repo" ]] && {
                best_repo="$repo"; best_quant="$quant"
                best_ctx="$ctx";  best_desc="$desc"
            }
            if model_cached "$repo" && [[ -z "$cached_repo" ]]; then
                cached_repo="$repo"; cached_quant="$quant"
                cached_ctx="$ctx";  cached_desc="$desc"
            fi
        fi
    done

    [[ -n "$best_repo" ]] \
        || die "No Hermes model fits ${VRAM_GB} GiB VRAM. Free GPU memory and retry."

    if [[ -n "$cached_repo" ]]; then
        log_ok "Cached model found — using: $cached_desc"
        SELECTED_MODEL="$cached_repo"; SELECTED_QUANT="$cached_quant"
        SELECTED_CTX="$cached_ctx";   SELECTED_DESC="$cached_desc"
    else
        log_info "No cached model — will download: $best_desc ($best_repo)"
        log_warn "Ensure HuggingFace access and sufficient disk space."
        SELECTED_MODEL="$best_repo"; SELECTED_QUANT="$best_quant"
        SELECTED_CTX="$best_ctx";   SELECTED_DESC="$best_desc"
    fi
}

# ─── step 4: write & enable vllm systemd service (skipped if already running) ─
setup_vllm_service() {
    "$VLLM_RUNNING" && { log_step "vllm service setup"; log_ok "Skipped — server already running on port ${VLLM_PORT}."; return 0; }

    log_step "Configuring vllm systemd service"

    local quant_flag=""
    [[ -n "$SELECTED_QUANT" ]] && quant_flag="--quantization ${SELECTED_QUANT}"

    local service_file="/etc/systemd/system/${VLLM_SERVICE_NAME}.service"

    sudo tee "$service_file" > /dev/null <<EOF
[Unit]
Description=Hermes vllm OpenAI-compatible API server
After=network.target nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=simple
User=$(whoami)
Environment="CUDA_VISIBLE_DEVICES=all"
Environment="HF_HOME=${HOME}/.cache/huggingface"
ExecStart=$PYTHON -m vllm.entrypoints.openai.api_server \\
    --model ${SELECTED_MODEL} \\
    --host ${VLLM_HOST} \\
    --port ${VLLM_PORT} \\
    --tensor-parallel-size ${TP} \\
    --max-model-len ${SELECTED_CTX} \\
    --served-model-name hermes \\
    --api-key ${VLLM_API_KEY} \\
    --enable-auto-tool-choice \\
    --tool-call-parser hermes \\
    ${quant_flag}
Restart=on-failure
RestartSec=10
StandardOutput=append:${VLLM_LOG}
StandardError=append:${VLLM_LOG}
TimeoutStartSec=600
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable "${VLLM_SERVICE_NAME}"
    log_ok "Service written: $service_file"
}

# ─── step 5: start vllm (skipped if already running) ─────────────────────────
start_vllm() {
    "$VLLM_RUNNING" && return 0

    log_step "Starting vllm service"

    if systemctl is-active --quiet "${VLLM_SERVICE_NAME}" 2>/dev/null; then
        log_info "Stopping stale ${VLLM_SERVICE_NAME}…"
        sudo systemctl stop "${VLLM_SERVICE_NAME}"
    fi

    sudo systemctl start "${VLLM_SERVICE_NAME}"
    log_info "Waiting for vllm to become healthy (model load can take 60–300 s)…"

    local deadline=$(( SECONDS + 300 ))
    while (( SECONDS < deadline )); do
        if curl -sf --max-time 3 "http://${VLLM_HOST}:${VLLM_PORT}/health" \
               -H "Authorization: Bearer ${VLLM_API_KEY}" &>/dev/null; then
            log_ok "vllm healthy at http://${VLLM_HOST}:${VLLM_PORT}"
            VLLM_RUNNING=true
            return 0
        fi
        printf "."
        sleep 5
    done
    echo ""
    log_error "vllm did not become healthy within 300 s."
    log_error "Check: journalctl -u ${VLLM_SERVICE_NAME} -n 50"
    log_warn "Continuing install — start vllm manually when ready."
}

# ─── step 6: install Hermes Agent ────────────────────────────────────────────
install_hermes() {
    log_step "Installing Hermes Agent"

    if [[ -f "$HOME/.local/bin/hermes" ]] || command -v hermes &>/dev/null; then
        log_ok "Hermes already installed — skipping."
        return 0
    fi

    log_info "Running official Hermes install script…"
    curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
    log_ok "Hermes Agent installed."
}

# ─── step 7: configure Hermes to use local vllm ──────────────────────────────
configure_hermes() {
    log_step "Configuring Hermes to use local vllm"

    mkdir -p "$HERMES_DIR"

    # The model name Hermes should request — use "hermes" alias when we own the
    # service (--served-model-name hermes), or the raw model id when attaching
    # to an externally managed server.
    local hermes_model_name
    if "$VLLM_RUNNING" && systemctl is-active --quiet "${VLLM_SERVICE_NAME}" 2>/dev/null; then
        hermes_model_name="hermes"
    else
        hermes_model_name="$SELECTED_MODEL"
    fi

    local config="$HERMES_DIR/config.yaml"

    if [[ -f "$config" ]]; then
        $PYTHON - <<PYEOF
import re, pathlib
p = pathlib.Path("$config")
text = p.read_text()
text = re.sub(r'^llm:.*?(?=^\w|\Z)', '', text, flags=re.MULTILINE|re.DOTALL)
p.write_text(text.rstrip() + "\n")
PYEOF
    else
        touch "$config"
    fi

    cat >> "$config" <<YAML

# ── vllm local backend (set by hermes DGX install script) ──
llm:
  provider: openai
  base_url: "http://${VLLM_HOST}:${VLLM_PORT}/v1"
  api_key: "${VLLM_API_KEY}"
  model: "${hermes_model_name}"
  max_tokens: 4096
  temperature: 0.7
YAML

    log_ok "config.yaml updated: $config"

    # Stamp .env (idempotent: remove old entries first)
    local env_file="$HERMES_DIR/.env"
    touch "$env_file"
    $PYTHON - <<PYEOF
import pathlib, re
p = pathlib.Path("$env_file")
text = p.read_text() if p.exists() else ""
for key in ("OPENAI_API_BASE", "OPENAI_API_KEY", "HERMES_MODEL"):
    text = re.sub(rf'^{key}=.*\n?', '', text, flags=re.MULTILINE)
p.write_text(text.rstrip() + "\n")
PYEOF
    {
        echo "OPENAI_API_BASE=http://${VLLM_HOST}:${VLLM_PORT}/v1"
        echo "OPENAI_API_KEY=${VLLM_API_KEY}"
        echo "HERMES_MODEL=${hermes_model_name}"
    } >> "$env_file"

    log_ok ".env updated: $env_file"
}

# ─── step 8: smoke test ───────────────────────────────────────────────────────
smoke_test() {
    log_step "Smoke test"

    local resp
    resp=$(curl -sf --max-time 5 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" \
                -H "Authorization: Bearer ${VLLM_API_KEY}" 2>/dev/null \
           || curl -sf --max-time 5 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" 2>/dev/null \
           || echo "")

    if [[ -n "$resp" ]]; then
        local listed
        listed=$($PYTHON - "$resp" <<'PYEOF'
import sys, json
try:
    ids = [m["id"] for m in json.loads(sys.argv[1]).get("data", [])]
    print(", ".join(ids) if ids else "(none)")
except Exception:
    print("(parse error)")
PYEOF
        )
        log_ok "Models available: $listed"
    else
        log_warn "Could not reach vllm at http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
        log_warn "Start it manually, then run: curl http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
    fi
}

# ─── summary ──────────────────────────────────────────────────────────────────
print_summary() {
    local vllm_origin
    "$VLLM_RUNNING" && vllm_origin="already running (reused)" || vllm_origin="started by this script"

    echo ""
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗"
    echo -e "║           Hermes DGX Spark — Install Complete            ║"
    echo -e "╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  Model       : ${BOLD}${SELECTED_DESC}${NC}"
    echo -e "  vllm API    : http://${VLLM_HOST}:${VLLM_PORT}/v1  (${vllm_origin})"
    echo -e "  GPUs / VRAM : ${NGPUS} GPU(s) — ${VRAM_GB} GiB free"
    echo ""
    echo -e "  ${BOLD}Commands:${NC}"
    echo -e "    hermes                                   # start the agent"
    echo -e "    systemctl status ${VLLM_SERVICE_NAME}             # vllm status"
    echo -e "    journalctl -u ${VLLM_SERVICE_NAME} -f             # vllm logs"
    echo -e "    curl http://${VLLM_HOST}:${VLLM_PORT}/v1/models   # list models"
    echo ""
    echo -e "  Install log : $LOG_FILE"
    echo ""
}

# ─── main ────────────────────────────────────────────────────────────────────
main() {
    echo -e "${BOLD}${BLUE}"
    echo "  ██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗"
    echo "  ██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝"
    echo "  ███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗"
    echo "  ██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║"
    echo "  ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║"
    echo "  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝"
    echo -e "  DGX Spark Edition${NC}"
    echo ""

    check_prerequisites   # verify vllm installed, Python, nvidia-smi
    detect_gpu            # NGPUS, VRAM_GB, TP
    detect_running_vllm   # probe common ports; sets VLLM_RUNNING + SELECTED_MODEL if found
    select_model          # no-op if VLLM_RUNNING; otherwise pick best fit from catalogue
    setup_vllm_service    # no-op if VLLM_RUNNING; write systemd unit
    start_vllm            # no-op if VLLM_RUNNING; start and wait for health
    install_hermes        # run upstream install if not already present
    configure_hermes      # patch ~/.hermes/config.yaml and .env
    smoke_test            # list models from live API
    print_summary
}

main "$@"
