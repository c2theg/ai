#!/usr/bin/env bash
# Christopher Gray  |  Version: 0.3.1  |  Update: 7/11/2026
# vLLM install, model download, and serve script for DGX Spark / NVIDIA systems
#
# Update Yourself:
#   curl -fsSL -o 'install_ai_spark_vllm.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_ai_spark_vllm.sh' && chmod u+x install_ai_spark_vllm.sh
#   ./install_ai_spark_vllm.sh --start "Qwen3.6-35B-A3B-NVFP4:9000,Qwen3-Reranker-4B"
#
# Move to DGX Spark / GB10:
#   scp install_ai_spark_vllm.sh root@<dgx-ip>:/home/user/install_ai_spark_vllm.sh
#   
#   scp install_ai_spark_vllm.sh user@10.11.1.10:/home/user/install_ai_spark_vllm.sh
#
# Usage:
#   ./install_ai_spark_vllm.sh              — full install: packages, docker, venv, download, serve
#   ./install_ai_spark_vllm.sh --serve-only — skip install/download; jump straight to model serve
#   ./install_ai_spark_vllm.sh -s           — same as --serve-only
#
#   Headless / boot-time serving (no prompts — safe for cron):
#   ./install_ai_spark_vllm.sh --start <spec>        — start 1+ models non-interactively
#       <spec> = model[:port][,model[:port]...]      — model = local dir name, HF repo id,
#                                                      or catalog index; port overrides the
#                                                      catalog default. --start is repeatable.
#       e.g.  --start Qwen3.6-35B-A3B-NVFP4
#             --start "Qwen3.6-35B-A3B-NVFP4:9000,Qwen3-Reranker-4B"
#   ./install_ai_spark_vllm.sh --install-cron <spec> — install an @reboot crontab entry that
#                                                      runs --start <spec> at every boot
#   ./install_ai_spark_vllm.sh --remove-cron         — remove that @reboot entry
#   ./install_ai_spark_vllm.sh --list-models         — print the servable-model catalog and exit
#
# ── Changelog ─────────────────────────────────────────────────────────────────
#
# v0.3.0  7/11/2026
#   - Added nvidia/Qwen3.6-35B-A3B-NVFP4  (catalog idx 22, port 8016, ~22 GB)
#     and unsloth/Qwen3.6-35B-A3B-NVFP4-Fast (catalog idx 23, port 8017, ~22 GB).
#     NVFP4 (4-bit) halves the FP8 footprint — both use --quantization modelopt_fp4.
#   - Headless startup mode: --start "model[:port],..." serves 1+ models with NO
#     prompts (skips install, menus, memory prompts, docker containers, OpenWebUI).
#     Models are matched by local dir name, HF repo id, or catalog index; an
#     optional :port overrides the catalog port. Designed for cron @reboot.
#   - --install-cron <spec> writes the @reboot crontab entry for you (and
#     --remove-cron deletes it). --list-models prints the catalog and exits.
#   - REFACTOR: replaced the ~280 lines of per-index `if is_run_selected N`
#     serve blocks with one _serve_model() dispatching on the HF repo id, plus a
#     single loop over RUN_SELECTED. Serve args can no longer silently mis-map
#     when catalog indices shift (the v0.1.5 bug class is now impossible).
#   - BUGFIX (found during refactor): Qwen3.5-9B is really catalog idx 19 (its
#     _add line sits before the two appended ASR entries), but its serve block
#     checked idx 21 — so selecting 9B launched NOTHING, and selecting
#     Qwen3-ASR-1.7B (really idx 21) launched the 9B serve args. Fixed
#     automatically by the repo-id dispatch above.
#   - _vllm_launch: readiness poll now hits /health (cheaper than /v1/models)
#     every 5s instead of every 15s, printing progress every 30s — small models
#     become ready up to ~14s sooner; log noise unchanged.
#   - _kill_vllm_processes now also kills a previous run's sleep_watchdog.sh so
#     watchdogs no longer stack across restarts.
#   - Removed dead helpers is_dl_selected/is_run_selected (unused after refactor).
#
# v0.2.9  6/27/2026
#   - Per-model sleep prompt: replaced the single "Standard models timeout"
#     prompt with a prompt for every selected servable model. Default shown in
#     brackets is the catalog SLEEP_MIN override if set, else IDLE_SLEEP_MINUTES.
#     User can set a different idle-sleep timeout for each model before serving.
#
# v0.2.8  6/27/2026
#   - Sequential model startup: _vllm_launch now polls GET /v1/models after
#     launch and blocks until the model reports ready (or 12-minute timeout).
#     Each model fully loads before the next one starts, so later models see a
#     stable baseline when the KV-cache profiler runs — no more OOM from racing
#     concurrent startups. Also detects if the process dies mid-load and prints
#     the tail of the log immediately.
#   - Qwen3.5-9B gpu-memory-utilization raised 0.50 → 0.75 and max-model-len
#     capped at 16384. When Nemotron-3-Nano-Omni-30B (~62 GB) is already loaded,
#     0.75×121=90.75 GB budget leaves ~10 GB for KV cache after 9B weights.
#
# v0.2.7  6/26/2026
#   - Raised Qwen3-Reranker-4B gpu-memory-utilization 0.50 → 0.55. With more
#     models now loaded concurrently (4B/2B/9B small models), the KV-cache
#     profiler baseline crept to ~61.2 GB — just 0.71 GB over the 60.5 GB
#     budget at 0.50. New 0.55 budget (66.6 GB) gives ~5 GB KV headroom.
#
# v0.2.6  6/26/2026
#   - Fixed HF repo IDs for small Qwen3.5 dense models: Qwen/Qwen3.5-4B-Instruct
#     and Qwen/Qwen3.5-2B-Instruct → Qwen/Qwen3.5-4B and Qwen/Qwen3.5-2B
#     (the repos have no -Instruct suffix). Updated catalog, local dirs,
#     serve block --served-model-name args, and changelog refs.
#   - Fixed download loop to check exit code: now prints ❌ on failure instead of
#     falsely printing ✅ regardless of whether huggingface-cli succeeded.
#   - Added Qwen/Qwen3.5-9B (catalog idx 21, port 8015, ~18 GB VRAM)
#     with SLEEP_MIN=60 (same idle-sleep pattern as the 2B/4B small models).
#
# ──────────────────────────────────────────────────────────────────────────────

# ─── strict mode ──────────────────────────────────────────────────────────────
# -u: error on unset variables  -o pipefail: propagate pipeline failures
# -e (exit on error) is intentionally omitted — this script uses many
# [ cond ] && action patterns and || true guards that conflict with -e.
set -uo pipefail

# ─── argument parsing ─────────────────────────────────────────────────────────
SERVE_ONLY=0
HEADLESS=0        # set by --start: non-interactive serve of START_SPECS, then exit
LIST_MODELS=0
START_SPECS=""    # comma-separated model[:port] specs collected from --start
CRON_ACTION=""    # install | remove
CRON_SPEC=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --serve-only|-s) SERVE_ONLY=1 ;;
        --start)
            [ "$#" -ge 2 ] || { echo "❌ --start needs a model spec, e.g. --start 'Model:8016,Model2'"; exit 1; }
            shift; START_SPECS="${START_SPECS:+$START_SPECS,}$1" ;;
        --start=*)       START_SPECS="${START_SPECS:+$START_SPECS,}${1#--start=}" ;;
        --install-cron)
            [ "$#" -ge 2 ] || { echo "❌ --install-cron needs a model spec, e.g. --install-cron 'Model:8016'"; exit 1; }
            shift; CRON_ACTION="install"; CRON_SPEC="$1" ;;
        --install-cron=*) CRON_ACTION="install"; CRON_SPEC="${1#--install-cron=}" ;;
        --remove-cron)   CRON_ACTION="remove" ;;
        --list-models)   LIST_MODELS=1 ;;
        -h|--help)
            # Print the Usage block from the header (stop at the first Changelog line).
            sed -n '/^# Usage:/,$p' "$0" | sed -n '1,/^# ── Changelog/p' | sed '$d' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "⚠️  Unknown argument: $1 (see --help)" ;;
    esac
    shift
done

if [ -n "$START_SPECS" ]; then
    HEADLESS=1
    SERVE_ONLY=1   # headless mode never installs or downloads
fi

echo "


 _____             _         _    _          _
|     |___ ___ ___| |_ ___ _| |  | |_ _ _   |_|
|   --|  _| -_| .'|  _| -_| . |  | . | | |   _
|_____|_| |___|__,|_| |___|___|  |___|_  |  |_|
                                     |___|

 _____ _       _     _           _              _____    __    _____
|     | |_ ___|_|___| |_ ___ ___| |_ ___ ___   |     |__|  |  |   __|___ ___ _ _
|   --|   |  _| |_ -|  _| . | . |   | -_|  _|  | | | |  |  |  |  |  |  _| .'| | |
|_____|_|_|_| |_|___|_| |___|  _|_|_|___|_|    |_|_|_|_____|  |_____|_| |__,|_  |
                            |_|                                             |___|


Version:  0.3.1
Last Updated:  7/11/2026

Update Yourself:
    curl -fsSL -o 'install_ai_spark_vllm.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_ai_spark_vllm.sh' && chmod u+x install_ai_spark_vllm.sh

  YOU MUST HAVE A HUGGINGFACE ACCOUNT AND TOKEN TO DOWNLOAD MODELS!
    *** Update 'HF_TOKEN' on line 35 before running this script! ***
        Huggingface models:   https://huggingface.co/models


"

# =============================================
# CONFIGURATION — set these before running
# =============================================
HF_TOKEN=""                  # HuggingFace token — fallback if not in .env
                             # Get yours at: https://huggingface.co/settings/tokens
BASE_DIR="/opt/models"       # All paths derive from here — change this one line to relocate everything

MODELS_DIR="$BASE_DIR/vllm"           # Where all models will be downloaded
VLLM_VENV="$BASE_DIR/vllm-install/.vllm"  # venv created by the vLLM install script
NEMO_VENV="$BASE_DIR/nemo-venv"       # separate venv for NeMo ASR (avoids conflicts with vLLM)

# =============================================
# OPTIONAL FEATURES — toggle on/off
# =============================================
ENABLE_SEARXNG=true          # SearXNG web search engine for OpenWebUI (runs on port 4040)
SEARXNG_PORT=4040            # host port for SearXNG — change if 4040 is in use

BRAVE_SEARCH_API_KEY=""      # Brave Search API key — takes priority over SearXNG when set
                             # Get yours at: https://api.search.brave.com/

# =============================================
# OPENWEBUI AUTO-REGISTRATION
# =============================================
OWUI_ADMIN_EMAIL="admin@local"
OWUI_ADMIN_PASSWORD="Abc123!@#"

# =============================================
# MEMORY & AI INFRASTRUCTURE
# =============================================
IDLE_SLEEP_MINUTES=15        # Offload models to CPU after this many idle minutes (0 = disabled)
                             # Used as the fallback for any model without its own timeout.
STANDARD_SLEEP_MINUTES=15    # Default idle-sleep timeout (minutes) for all Standard models.
                             # The script prompts before serving so this can be overridden
                             # at runtime (0 = never sleep). Press Enter at the prompt to accept.

ENABLE_SQLITE_MEMORY=true    # SQLite DB for structured memory (exact facts, conversations)
SQLITE_RETENTION_DAYS=60     # Delete records older than N days  (≈ 2 months)
SQLITE_MAX_MB=100            # Also prune oldest rows when DB exceeds this size

ENABLE_QDRANT=true           # Qdrant vector DB for long-term semantic memory
QDRANT_HTTP_PORT=6333        # Qdrant REST API port
QDRANT_GRPC_PORT=6334        # Qdrant gRPC port
QDRANT_RETENTION_DAYS=60     # Delete vectors older than N days  (≈ 2 months)
QDRANT_MAX_GB=1              # Warn (and log) when storage exceeds this many GB

MEMORY_DIR="$BASE_DIR/memory"
SQLITE_DB="$MEMORY_DIR/structured_memory.db"
QDRANT_DATA_DIR="$BASE_DIR/qdrant"

# Load .env from same directory as this script — overrides tokens above if set there
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

_env_load() { grep -E "^$1=" "$ENV_FILE" 2>/dev/null | head -1 | cut -d '=' -f2- | tr -d '"' | tr -d "'"; }
_env_save() {
    if grep -qE "^$1=" "$ENV_FILE" 2>/dev/null; then
        sed -i "s|^$1=.*|$1=$2|" "$ENV_FILE"
    else
        echo "$1=$2" >> "$ENV_FILE"
    fi
}

ENV_HF_TOKEN=$(_env_load HF_TOKEN)
if [ -n "$ENV_HF_TOKEN" ]; then
    HF_TOKEN="$ENV_HF_TOKEN"; echo "✅ HF_TOKEN loaded from .env"
elif [ -n "$HF_TOKEN" ]; then
    _env_save HF_TOKEN "$HF_TOKEN"; echo "✅ HF_TOKEN saved to $ENV_FILE"
fi
[ -z "$HF_TOKEN" ] && echo "⚠️  HF_TOKEN is not set — gated models will fail."

ENV_BRAVE_KEY=$(_env_load BRAVE_SEARCH_API_KEY)
[ -n "$ENV_BRAVE_KEY" ] && BRAVE_SEARCH_API_KEY="$ENV_BRAVE_KEY" && echo "✅ BRAVE_SEARCH_API_KEY loaded from .env"
[ -z "$ENV_BRAVE_KEY" ] && [ -n "$BRAVE_SEARCH_API_KEY" ] && _env_save BRAVE_SEARCH_API_KEY "$BRAVE_SEARCH_API_KEY"

ENV_OWUI_EMAIL=$(_env_load OWUI_ADMIN_EMAIL)
[ -n "$ENV_OWUI_EMAIL" ] && OWUI_ADMIN_EMAIL="$ENV_OWUI_EMAIL" && echo "✅ OWUI_ADMIN_EMAIL loaded from .env"
[ -z "$ENV_OWUI_EMAIL" ] && [ -n "$OWUI_ADMIN_EMAIL" ] && _env_save OWUI_ADMIN_EMAIL "$OWUI_ADMIN_EMAIL"

ENV_OWUI_PASS=$(_env_load OWUI_ADMIN_PASSWORD)
[ -n "$ENV_OWUI_PASS" ] && OWUI_ADMIN_PASSWORD="$ENV_OWUI_PASS" && echo "✅ OWUI_ADMIN_PASSWORD loaded from .env"
[ -z "$ENV_OWUI_PASS" ] && [ -n "$OWUI_ADMIN_PASSWORD" ] && _env_save OWUI_ADMIN_PASSWORD "$OWUI_ADMIN_PASSWORD"

if [ -z "$OWUI_ADMIN_EMAIL" ] || [ -z "$OWUI_ADMIN_PASSWORD" ]; then
    echo "⚠️  OWUI credentials not set — visit http://localhost:3000 on first run to create your admin account."
fi

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CATALOG
# Fields: HF_REPO | LOCAL_DIR | DISPLAY_NAME | DISK_GB | VRAM_GB | PORT | CATEGORY | SLEEP_MIN(optional)
# VRAM_GB=0   → CPU/NeMo only — cannot be served via vLLM
# PORT=0      → download-only (ASR/NeMo)
# SLEEP_MIN   → optional per-model idle-sleep timeout in minutes; overrides the
#               global IDLE_SLEEP_MINUTES for this model only.  Omit to use global.
# ─────────────────────────────────────────────────────────────────────────────
MDL_HF=()
MDL_DIR=()
MDL_NAME=()
MDL_DISK=()
MDL_VRAM=()
MDL_PORT=()
MDL_CAT=()
MDL_SLEEP=()

_add() {
    local i=${#MDL_HF[@]}
    MDL_HF[$i]="$1"; MDL_DIR[$i]="$2"; MDL_NAME[$i]="$3"
    MDL_DISK[$i]="$4"; MDL_VRAM[$i]="$5"; MDL_PORT[$i]="$6"; MDL_CAT[$i]="$7"
    MDL_SLEEP[$i]="${8:-}"
}

# ── Standard models ────────────────────────────────────────────────────────────
# Sleep timeout: these all use STANDARD_SLEEP_MINUTES (prompted/overridable at
# runtime). The catalog leaves SLEEP_MIN blank here; it is filled in for the whole
# range below once the user confirms the timeout. The index span is captured in
# STANDARD_INDICES so the chosen value can be applied to exactly these models.
_STD_RANGE_START=${#MDL_HF[@]}
#        HF Repo                                                   Local Dir                               Display Name                          Disk VRAM  Port  Category
_add "Qwen/Qwen3.6-35B-A3B-FP8"                                  "Qwen3.6-35B-A3B-FP8"                   "Qwen3.6-35B-A3B (FP8)"                  35   38   8005  "General"
_add "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"               "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"  "Nemotron-3-Nano-30B-A3B (NVFP4)"        15   18   8006  "General"
_add "Qwen/Qwen3-Coder-30B-A3B-Instruct"                         "Qwen3-Coder-30B-A3B-Instruct"          "Qwen3-Coder-30B-A3B (BF16)"             60   65   8001  "Coding"
_add "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"                  "DeepSeek-R1-Distill-Qwen-32B"          "DeepSeek-R1-Distill-Qwen-32B (BF16)"    64   68   8002  "Reasoning"
_add "google/gemma-4-31B-it"                                     "gemma-4-31B-it"                        "Gemma 4 31B-it (BF16)"                  62   66   8009  "General"
_add "google/gemma-4-26B-A4B-it"                                 "gemma-4-26B-A4B-it"                    "Gemma 4 26B-A4B (BF16)"                 52   56   8007  "General"
_add "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"        "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" "Nemotron-3-Nano-Omni-30B (BF16)"  60   65   8008  "Reasoning"
_add "BAAI/bge-m3"                                               "bge-m3"                                "BGE-M3 (Embeddings)"                     3    4   8011  "Embeddings"
_add "Qwen/Qwen3-Embedding-4B"                                   "Qwen3-Embedding-4B"                    "Qwen3-Embedding-4B (Embeddings)"         8   10   8010  "Embeddings"
_add "BAAI/bge-reranker-v2-m3"                                   "bge-reranker-v2-m3"                    "BGE-Reranker-v2-m3 (Reranking)"          2    3   8020  "Reranking"
_add "Qwen/Qwen3-Reranker-4B"                                    "Qwen3-Reranker-4B"                     "Qwen3-Reranker-4B (Reranking) ★Default"   8    9   8021  "Reranking"
_add "nvidia/parakeet-tdt-0.6b-v3"                               "parakeet-tdt-0.6b-v3"                  "Parakeet-TDT-0.6B v3 (ASR / NeMo)"       1    0      0  "ASR"
_add "nvidia/nemotron-speech-streaming-en-0.6b"                  "nemotron-speech-streaming-en-0.6b"     "Nemotron-Speech-Streaming-0.6B (ASR)"    1    0      0  "ASR"

# Catalog indices of the Standard-models block above — used to apply the
# runtime-chosen sleep timeout to exactly these models.
STANDARD_INDICES=()
for _si in $(seq "$_STD_RANGE_START" $(( ${#MDL_HF[@]} - 1 ))); do STANDARD_INDICES+=("$_si"); done

# ── SUPER LARGE models (120B+ parameters) ─────────────────────────────────────
# Info: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard
# Note: these require nearly the entire GPU — do not run alongside other large models.
# ⚠️  Verify HF repo IDs before downloading — these may require updated values.
_add "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"             "NVIDIA-Nemotron-3-Super-120B-A12B-BF16" "Nemotron-3-Super-120B-A12B (BF16) [SUPER]" 120  115   8030  "Super Large"
_add "Qwen/Qwen3.5-122B-A10B"                                    "Qwen3.5-122B-A10B"                     "Qwen3.5-122B-A10B (BF16) [SUPER]"           122  120   8031  "Super Large"
_add "Qwen/Qwen3.5-122B-A10B-FP8"                               "Qwen3.5-122B-A10B-FP8"                 "Qwen3.5-122B-A10B (FP8) [SUPER] ★Rec"        62   65   8033  "Super Large"
_add "openai/gpt-oss-120b"                                       "gpt-oss-120b"                          "GPT-OSS-120B [SUPER]"                       120  115   8032  "Super Large"

# ── Small models with a custom idle-sleep timeout ─────────────────────────────
# These pass the optional 8th _add field (SLEEP_MIN) = 60, so the sleep watchdog
# offloads them to CPU after 1 hour idle instead of the global IDLE_SLEEP_MINUTES.
# Appended at the end of the catalog so the existing indices above (and their
# matching serve blocks) are not shifted.
#        HF Repo                       Local Dir              Display Name                              Disk VRAM  Port  Category    Sleep(min)
_add "Qwen/Qwen3.5-4B"               "Qwen3.5-4B"           "Qwen3.5-4B (BF16) [1h sleep]"            8   10   8012  "General"   60
_add "Qwen/Qwen3.5-2B"               "Qwen3.5-2B"           "Qwen3.5-2B (BF16) [1h sleep]"            4    5   8013  "General"   60
_add "Qwen/Qwen3.5-9B"               "Qwen3.5-9B"           "Qwen3.5-9B (BF16) [1h sleep]"           18   18   8015  "General"   60

# ── Additional ASR / NeMo model (download-only, not served via vLLM) ───────────
# https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
# Appended here (not next to idx 11/12) to keep existing catalog indices stable.
_add "nvidia/nemotron-3.5-asr-streaming-0.6b" "nemotron-3.5-asr-streaming-0.6b" "Nemotron-3.5-ASR-Streaming-0.6B (ASR)"   1    0      0  "ASR"

# ── ASR model served via vLLM (transcription endpoint, gets a port) ───────────
# https://huggingface.co/Qwen/Qwen3-ASR-1.7B
# Unlike the NeMo ASR models above, this is served by vLLM (--task transcription)
# and exposes POST /v1/audio/transcriptions. PORT is non-zero so it shows up in
# the serve menu and can be selected like any other model.
_add "Qwen/Qwen3-ASR-1.7B"           "Qwen3-ASR-1.7B"       "Qwen3-ASR-1.7B (ASR, served)"            4    5   8014  "ASR"

# ── Qwen3.6-35B-A3B NVFP4 quantizations (added v0.3.0) ────────────────────────
# NVFP4 is ~4-bit: ~20 GB weights vs ~35 GB for the FP8 build at idx 0.
# https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
# https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
#        HF Repo                                Local Dir                       Display Name                              Disk VRAM  Port  Category
_add "nvidia/Qwen3.6-35B-A3B-NVFP4"          "Qwen3.6-35B-A3B-NVFP4"        "Qwen3.6-35B-A3B (NVFP4, nvidia)"         20   22   8016  "General"
_add "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast"    "Qwen3.6-35B-A3B-NVFP4-Fast"   "Qwen3.6-35B-A3B (NVFP4-Fast, unsloth)"   20   22   8017  "General"

MODEL_TOTAL=${#MDL_HF[@]}

# ── Default pre-selected models ────────────────────────────────────────────────
# Qwen3-Reranker-4B (catalog idx 10) is pre-selected for download and serve.
# Users can deselect it in the interactive menus below.
DEFAULT_DL_INDICES=(10)
DEFAULT_SERVE_INDICES=(10)

# ─────────────────────────────────────────────────────────────────────────────
# HEADLESS HELPERS — model resolution, catalog listing, @reboot cron management
# ─────────────────────────────────────────────────────────────────────────────

# Resolve a user-supplied model reference to a catalog index (echoed on stdout).
# Accepts: catalog index, local dir name, full HF repo id, or the repo basename.
# Matching is case-insensitive. Returns 1 if not found or not servable (PORT=0).
_resolve_model() {
    local q="$1" i lq dir_lc hf_lc
    if [[ "$q" =~ ^[0-9]+$ ]]; then
        [ "$q" -lt "$MODEL_TOTAL" ] && [ "${MDL_PORT[$q]}" != "0" ] && { echo "$q"; return 0; }
        return 1
    fi
    lq="$(printf '%s' "$q" | tr '[:upper:]' '[:lower:]')"
    for i in $(seq 0 $((MODEL_TOTAL - 1))); do
        [ "${MDL_PORT[$i]}" = "0" ] && continue
        dir_lc="$(printf '%s' "${MDL_DIR[$i]}" | tr '[:upper:]' '[:lower:]')"
        hf_lc="$(printf '%s' "${MDL_HF[$i]}"  | tr '[:upper:]' '[:lower:]')"
        if [ "$lq" = "$dir_lc" ] || [ "$lq" = "$hf_lc" ] || [ "$lq" = "${hf_lc##*/}" ]; then
            echo "$i"; return 0
        fi
    done
    return 1
}

# Print every servable model (PORT != 0) — the names accepted by --start.
_list_servable_models() {
    echo ""
    printf "  %-4s  %-42s  %-52s  %5s  %6s\n" "Idx" "Name for --start (local dir)" "HF repo" "Port" "VRAM"
    printf "  %-4s  %-42s  %-52s  %5s  %6s\n" "---" "----------------------------" "-------" "----" "----"
    local i
    for i in $(seq 0 $((MODEL_TOTAL - 1))); do
        [ "${MDL_PORT[$i]}" = "0" ] && continue
        printf "  %-4d  %-42s  %-52s  %5s  %3d GB\n" \
            "$i" "${MDL_DIR[$i]}" "${MDL_HF[$i]}" "${MDL_PORT[$i]}" "${MDL_VRAM[$i]}"
    done
    echo ""
}

# Parse a "model[:port],model[:port],..." spec into RUN_SELECTED, applying any
# per-model port overrides directly to MDL_PORT so logs / watchdog / status all
# see the overridden port. Exits with the catalog listed on any bad entry.
_resolve_start_specs() {
    local spec="$1" part m p idx
    local -a parts
    IFS=',' read -ra parts <<< "$spec"
    for part in "${parts[@]}"; do
        part="${part//[[:space:]]/}"
        [ -z "$part" ] && continue
        m="${part%%:*}"; p=""
        [[ "$part" == *:* ]] && p="${part##*:}"
        if ! idx=$(_resolve_model "$m"); then
            echo "❌ Unknown or non-servable model: '$m'"
            echo "   Use one of the names below (or a catalog index):"
            _list_servable_models
            exit 1
        fi
        if [ -n "$p" ]; then
            [[ "$p" =~ ^[0-9]+$ ]] || { echo "❌ Invalid port '$p' for model '$m'"; exit 1; }
            MDL_PORT[$idx]="$p"
        fi
        RUN_SELECTED+=("$idx")
        echo "  ✅ --start: ${MDL_NAME[$idx]}  →  port ${MDL_PORT[$idx]}"
    done
    [ "${#RUN_SELECTED[@]}" -eq 0 ] && { echo "❌ --start: no models resolved from '$spec'"; exit 1; }
}

# Install/replace the @reboot crontab entry that re-launches models at boot.
# The 60s sleep gives the network / GPU driver time to come up; adjust if needed.
_install_boot_cron() {
    local spec="$1"
    # Validate the whole spec first — do not write a broken crontab entry.
    local -a parts; local part m p
    IFS=',' read -ra parts <<< "$spec"
    for part in "${parts[@]}"; do
        part="${part//[[:space:]]/}"
        [ -z "$part" ] && continue
        m="${part%%:*}"; p=""
        [[ "$part" == *:* ]] && p="${part##*:}"
        _resolve_model "$m" >/dev/null || { echo "❌ Unknown model '$m' in cron spec:"; _list_servable_models; exit 1; }
        [ -n "$p" ] && ! [[ "$p" =~ ^[0-9]+$ ]] && { echo "❌ Invalid port '$p' for '$m'"; exit 1; }
    done

    local script_path="$SCRIPT_DIR/$(basename "$0")"
    local cron_line="@reboot sleep 60 && mkdir -p '$BASE_DIR/logs' && '$script_path' --start '$spec' >> '$BASE_DIR/logs/startup_vllm.log' 2>&1"
    ( crontab -l 2>/dev/null | grep -vE "$(basename "$0")' --start" ; echo "$cron_line" ) | crontab -
    echo "✅ @reboot cron entry installed:"
    echo "   $cron_line"
    echo "   Boot log: $BASE_DIR/logs/startup_vllm.log"
    echo "   Remove with: $0 --remove-cron"
}

_remove_boot_cron() {
    if crontab -l 2>/dev/null | grep -qE "$(basename "$0")' --start"; then
        crontab -l 2>/dev/null | grep -vE "$(basename "$0")' --start" | crontab -
        echo "✅ @reboot vLLM startup cron entry removed."
    else
        echo "ℹ️  No @reboot vLLM startup cron entry found — nothing to remove."
    fi
}

# ── One-shot actions: list catalog / manage cron, then exit ──────────────────
if [ "$LIST_MODELS" -eq 1 ]; then
    _list_servable_models
    exit 0
fi
if [ "$CRON_ACTION" = "install" ]; then
    _install_boot_cron "$CRON_SPEC"; exit 0
elif [ "$CRON_ACTION" = "remove" ]; then
    _remove_boot_cron; exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# STATUS SNAPSHOT — memory utilization + models currently live in vLLM
# Probes every non-zero catalog port's /v1/models endpoint. Called once at
# startup (shows what a previous run left running) and once at the end.
# Arg: $1 = label for the section header (e.g. "STARTUP", "FINAL").
# ─────────────────────────────────────────────────────────────────────────────
_show_vllm_status() {
    local label="$1"
    echo ""
    echo "  ══ vLLM / MEMORY STATUS — ${label} ══════════════════════════════"

    # ---- System RAM (used / total) ----
    local mt ma
    mt=$(awk '/^MemTotal:/{print $2}'     /proc/meminfo 2>/dev/null)
    ma=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null)
    if [[ "$mt" =~ ^[0-9]+$ ]] && [[ "$ma" =~ ^[0-9]+$ ]]; then
        printf "  RAM     : %d / %d GB used  (%d%%)\n" \
            "$(( (mt - ma) / 1024 / 1024 ))" "$(( mt / 1024 / 1024 ))" "$(( (mt - ma) * 100 / mt ))"
    else
        echo "  RAM     : /proc/meminfo unavailable"
    fi

    # ---- GPU (best effort; unified-memory systems report N/A for memory) ----
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        local gutil gmu gmt
        gutil=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
        gmu=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
        gmt=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
        [[ "$gutil" =~ ^[0-9]+$ ]] && printf "  GPU     : %d%% compute utilization\n" "$gutil"
        if [[ "$gmu" =~ ^[0-9]+$ ]] && [[ "$gmt" =~ ^[0-9]+$ ]]; then
            printf "  GPU mem : %d / %d MiB used\n" "$gmu" "$gmt"
        else
            echo "  GPU mem : shared with system RAM (nvidia-smi reports N/A)"
        fi
    else
        echo "  GPU     : nvidia-smi not available"
    fi

    # ---- Models currently live in vLLM (probe known catalog ports) ----
    echo "  Models live in vLLM:"
    local found=0 seen=""
    for _i in $(seq 0 $((MODEL_TOTAL - 1))); do
        local p="${MDL_PORT[$_i]}"
        [ "$p" = "0" ] && continue
        case " $seen " in *" $p "*) continue ;; esac
        seen="$seen $p"
        local resp
        resp=$(curl -sf --max-time 2 "http://localhost:${p}/v1/models" 2>/dev/null) || continue
        local ids
        if command -v jq &>/dev/null; then
            ids=$(echo "$resp" | jq -r '.data[].id' 2>/dev/null | paste -sd ',' -)
        else
            ids=$(echo "$resp" | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' \
                  | sed -E 's/.*"([^"]*)"$/\1/' | paste -sd ',' -)
        fi
        [ -z "$ids" ] && ids="(up — model id unknown)"
        printf "    port %-6s : %s\n" "$p" "$ids"
        found=1
    done
    [ "$found" = "0" ] && echo "    (none responding)"
    echo "  ════════════════════════════════════════════════════════════════"
}

# Startup snapshot — what a previous run left running, before the clean-start kill.
_show_vllm_status "STARTUP"

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE CHECKBOX SELECTION
# ─────────────────────────────────────────────────────────────────────────────

_checkbox_menu() {
    # Args: $1=title  $2=servable_only (true|false)  $3=result_var_name  $4=defaults_var (optional)
    local title="$1" servable_only="$2" result_var="$3" defaults_var="${4:-}"

    local -a menu_map=()
    for i in $(seq 0 $((MODEL_TOTAL - 1))); do
        [ "$servable_only" = "true" ] && [ "${MDL_PORT[$i]}" = "0" ] && continue
        menu_map+=("$i")
    done

    # ── Display order: group by type, then size (VRAM, ascending), then name ───
    # COSMETIC ONLY. This reorders what the menu shows without changing catalog
    # indices, so every serve block, DEFAULT_*_INDICES, and STANDARD_INDICES keep
    # working unchanged. (Physically reordering the _add lines would shift indices
    # and mis-map serve blocks — the v0.1.5 bug.)
    if [ "${#menu_map[@]}" -gt 1 ]; then
        local -a _sortable=()
        local _mi _crank
        for _mi in "${menu_map[@]}"; do
            case "${MDL_CAT[$_mi]}" in
                General)       _crank=1 ;;
                Coding)        _crank=2 ;;
                Reasoning)     _crank=3 ;;
                Embeddings)    _crank=4 ;;
                Reranking)     _crank=5 ;;
                ASR)           _crank=6 ;;
                "Super Large") _crank=7 ;;
                *)             _crank=9 ;;
            esac
            _sortable+=("$(printf '%d|%04d|%s|%d' "$_crank" "${MDL_VRAM[$_mi]}" "${MDL_NAME[$_mi]}" "$_mi")")
        done
        menu_map=()
        while IFS='|' read -r _ _ _ _mi; do menu_map+=("$_mi"); done \
            < <(printf '%s\n' "${_sortable[@]}" | sort -t'|' -k1,1n -k2,2n -k3,3)
    fi

    local count=${#menu_map[@]}

    local -a sel=()
    for j in $(seq 0 $((count - 1))); do sel[$j]=0; done

    # Pre-select any default catalog indices passed via defaults_var
    if [ -n "$defaults_var" ]; then
        local -n _defs_ref="$defaults_var"
        for def_idx in "${_defs_ref[@]+${_defs_ref[@]}}"; do
            for j in $(seq 0 $((count - 1))); do
                [ "${menu_map[$j]}" = "$def_idx" ] && sel[$j]=1
            done
        done
    fi

    while true; do
        echo ""
        echo "  $title"
        printf "  %-4s  %-3s  %-50s  %7s  %10s  %-12s\n" \
               "Num" " " "Model" "Disk" "VRAM" "Category"
        printf "  %-4s  %-3s  %-50s  %7s  %10s  %-12s\n" \
               "---" "---" "-----" "------" "----------" "--------"

        for j in $(seq 0 $((count - 1))); do
            local i="${menu_map[$j]}"
            local mark="[ ]"; [ "${sel[$j]}" = "1" ] && mark="[x]"
            local vram_disp="CPU"
            [ "${MDL_VRAM[$i]}" -gt 0 ] && vram_disp="${MDL_VRAM[$i]} GB"
            printf "  %-4d  %s  %-50s  %4d GB  %10s  %-12s\n" \
                "$((j+1))" "$mark" "${MDL_NAME[$i]}" "${MDL_DISK[$i]}" "$vram_disp" "${MDL_CAT[$i]}"
        done

        local selected_count=0
        for j in $(seq 0 $((count - 1))); do [ "${sel[$j]}" = "1" ] && selected_count=$((selected_count+1)); done
        echo ""
        printf "  [%d selected]  Toggle: type number(s) separated by spaces\n" "$selected_count"
        echo "  Commands: a=select all  n=clear all  d=done"
        printf "  > "
        read -r input

        case "$input" in
            d|done|"") break ;;
            a|all) for j in $(seq 0 $((count - 1))); do sel[$j]=1; done ;;
            n|none|clear) for j in $(seq 0 $((count - 1))); do sel[$j]=0; done ;;
            *)
                for num in $input; do
                    if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "$count" ]; then
                        local j=$((num - 1))
                        [ "${sel[$j]}" = "1" ] && sel[$j]=0 || sel[$j]=1
                    fi
                done ;;
        esac
    done

    local -a result=()
    for j in $(seq 0 $((count - 1))); do
        [ "${sel[$j]}" = "1" ] && result+=("${menu_map[$j]}")
    done
    eval "$result_var=(\"\${result[@]+\"\${result[@]}\"}\") "
}

# ─────────────────────────────────────────────────────────────────────────────
# VRAM PRE-FLIGHT CHECK
# ─────────────────────────────────────────────────────────────────────────────

# Budget required model memory against system RAM (for unified-memory systems
# like the DGX Spark / GB10, where the GPU and CPU share one memory pool) and
# do a live memory-pressure check against what is actually free right now.
# Arg: $1 = total GB the selected models require.
_check_system_ram_budget() {
    local total_required="$1"
    local mem_total_kb mem_avail_kb
    mem_total_kb=$(awk '/^MemTotal:/{print $2}'     /proc/meminfo 2>/dev/null)
    mem_avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null)

    if ! [[ "$mem_total_kb" =~ ^[0-9]+$ ]]; then
        echo "  ⚠️  Could not read /proc/meminfo — skipping memory check."
        echo "     Selected models require ~${total_required} GB; ensure your RAM fits."
        return 0
    fi

    local total_gb=$(( mem_total_kb / 1024 / 1024 ))
    local avail_gb=0
    [[ "$mem_avail_kb" =~ ^[0-9]+$ ]] && avail_gb=$(( mem_avail_kb / 1024 / 1024 ))
    # 85% safe limit leaves headroom for the OS, KV cache, and the OpenWebUI /
    # Qdrant / SearXNG containers that share the same RAM.
    local safe_gb=$(( total_gb * 85 / 100 ))

    local used_gb=$(( total_gb - avail_gb ))
    printf "  %-20s : %3d GB   total physical RAM on this box\n"               "System RAM total" "$total_gb"
    printf "  %-20s : %3d GB   in use right now (OS, containers, old models)\n" "In use now"        "$used_gb"
    printf "  %-20s : %3d GB   free right now\n"                               "Available now"     "$avail_gb"
    printf "  %-20s : %3d GB   most you should load (85%% of total)\n"          "Safe limit"        "$safe_gb"
    printf "  %-20s : %3d GB   sum of selected models' VRAM estimates\n"        "Models require"    "$total_required"
    echo ""

    # Two distinct conditions, with different meanings and advice:
    #   capacity  — models exceed the safe limit; they won't fit even when idle.
    #   transient — models fit overall, but not enough is free this instant
    #               (usually a previous run's models, killed later in this script).
    local pressure="" kind=""
    if [ "$total_required" -gt "$safe_gb" ]; then
        kind="capacity"
        pressure="Models need ~${total_required} GB, over the ${safe_gb} GB safe limit (85% of ${total_gb} GB)."
    elif [ "$avail_gb" -gt 0 ] && [ "$total_required" -gt "$avail_gb" ]; then
        kind="transient"
        pressure="Models need ~${total_required} GB but only ${avail_gb} GB is free this instant (${used_gb} GB already in use)."
    fi

    if [ -n "$pressure" ]; then
        echo "  ⚠️  MEMORY PRESSURE: $pressure"
        if [ "$kind" = "transient" ]; then
            echo "     This is almost always because a PREVIOUS run's models are still"
            echo "     loaded (see the 'STARTUP' snapshot above). This script kills old"
            echo "     vLLM processes a few steps from now, which frees that RAM — so since"
            echo "     ${total_required} GB is under the ${safe_gb} GB safe limit, you can most"
            echo "     likely continue. If models then fail, watch their logs for OOM errors."
        else
            echo "     ${total_required} GB won't fit safely even on an idle box. KV cache,"
            echo "     OpenWebUI, Qdrant, and the OS also draw from this same shared pool."
            echo "     Deselect some models, or pick smaller / quantized (FP8/NVFP4) variants."
        fi
        echo -n "  Continue anyway? [y/N]: "
        read -r confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    else
        echo "  ✅ Memory check OK — need ~${total_required} GB; ${avail_gb} GB free now,"
        echo "     ${safe_gb} GB safe limit, ${total_gb} GB total."
    fi
}

# Kill any running vLLM model processes (a previous run's, or this script's).
# Shared by the optional pre-check shutdown and the later clean-start.
_kill_vllm_processes() {
    pkill -9 -f "vllm serve"        2>/dev/null || true
    pkill -9 -f "vllm.entrypoints"  2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore"  2>/dev/null || true
    pkill -9 -f "vllm.engine"       2>/dev/null || true
    # Also stop a previous run's sleep watchdog so watchdogs don't stack.
    pkill -f "sleep_watchdog.sh"    2>/dev/null || true
}

# Ask whether to shut down a previous run's still-loaded models before the memory
# check, so "Available now" reflects a clean slate instead of stale reservations.
# Only prompts when vLLM processes are actually detected.
_maybe_shutdown_existing_models() {
    pgrep -f "vllm serve" >/dev/null 2>&1 || pgrep -f "vllm.entrypoints" >/dev/null 2>&1 || return 0

    echo ""
    echo "  ⚠️  vLLM model(s) from a previous run are still loaded and holding memory"
    echo "      (see the STARTUP snapshot above). Shutting them down now frees that"
    echo "      memory so the budget check below reflects a clean slate."
    echo -n "  Shut down existing models now? [Y/n]: "
    read -r _kill_ans
    if [[ "$_kill_ans" =~ ^[Nn]$ ]]; then
        echo "  → Keeping existing models; 'Available now' will still include them."
    else
        echo "  → Stopping existing vLLM processes..."
        _kill_vllm_processes
        sleep 3
        echo "  ✅ Existing models stopped — memory freed."
    fi
}

_check_vram() {
    local total_required=0
    for idx in "${RUN_SELECTED[@]}"; do
        total_required=$((total_required + MDL_VRAM[idx]))
    done
    [ "$total_required" -eq 0 ] && return 0

    echo ""
    echo "  ── VRAM Budget ──────────────────────────────────────────"

    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        local total_mib
        total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)

        # Unified-memory systems (e.g. DGX Spark / GB10) report memory.total as
        # "[N/A]". The GPU shares system RAM, so budget against that instead.
        if ! [[ "$total_mib" =~ ^[0-9]+$ ]]; then
            echo "  ℹ️  Unified memory detected (nvidia-smi VRAM = '${total_mib:-N/A}')."
            echo "     GPU and CPU share one pool — budgeting against system RAM."
            _check_system_ram_budget "$total_required"
            echo "  ─────────────────────────────────────────────────────────"
            return 0
        fi

        local total_gb=$(( total_mib / 1024 ))
        local safe_gb=$(( total_gb * 90 / 100 ))

        printf "  %-20s : %d GB\n" "GPU total" "$total_gb"
        printf "  %-20s : %d GB  (90%%)\n" "Safe limit" "$safe_gb"
        printf "  %-20s : %d GB\n" "Models require" "$total_required"
        echo ""

        local has_super=0
        for idx in "${RUN_SELECTED[@]}"; do
            [ "${MDL_CAT[$idx]}" = "Super Large" ] && has_super=1 && break
        done

        if [ "$has_super" = "1" ] && [ "${#RUN_SELECTED[@]}" -gt 1 ]; then
            echo "  ⚠️  WARNING: You selected a SUPER LARGE model alongside other models."
            echo "     Super Large models (120B+) need nearly all GPU VRAM."
            echo "     Running multiple large models simultaneously will likely fail."
            echo -n "  Continue anyway? [y/N]: "
            read -r confirm
            [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
        elif [ "$total_required" -gt "$safe_gb" ]; then
            echo "  ⚠️  WARNING: Selected models require ~${total_required} GB but safe limit is ${safe_gb} GB."
            echo "     Consider deselecting some models."
            echo -n "  Continue anyway? [y/N]: "
            read -r confirm
            [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
        else
            echo "  ✅ VRAM check OK — ${total_required} GB needed, ${total_gb} GB available"
        fi
    else
        echo "  ⚠️  nvidia-smi not available — budgeting against system RAM instead."
        _check_system_ram_budget "$total_required"
    fi
    echo "  ─────────────────────────────────────────────────────────"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Select models to download  (skipped in --serve-only mode)
# ─────────────────────────────────────────────────────────────────────────────
DL_SELECTED=()
if [ "$SERVE_ONLY" -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  STEP 1 of 2 — Select models to DOWNLOAD"
    echo "════════════════════════════════════════════════════════════════════"
    _checkbox_menu "Available models (toggle with numbers, d=done):" "false" DL_SELECTED DEFAULT_DL_INDICES
else
    echo ""
    echo "  ⏭️  --serve-only: skipping download step (Step 1)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Select models to serve
# ─────────────────────────────────────────────────────────────────────────────
RUN_SELECTED=()
if [ "$HEADLESS" -eq 1 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  HEADLESS MODE (--start) — no prompts, serving requested models"
    echo "════════════════════════════════════════════════════════════════════"
    _resolve_start_specs "$START_SPECS"
else
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    if [ "$SERVE_ONLY" -eq 0 ]; then
        echo "  STEP 2 of 2 — Select models to SERVE with vLLM"
    else
        echo "  Select models to SERVE with vLLM"
    fi
    echo "  (ASR/NeMo models are download-only and excluded from this list)"
    echo "════════════════════════════════════════════════════════════════════"
    _checkbox_menu "Models to serve with vLLM (toggle with numbers, d=done):" "true" RUN_SELECTED DEFAULT_SERVE_INDICES

    # Offer to free a previous run's models before measuring available memory.
    _maybe_shutdown_existing_models

    _check_vram
fi

echo ""
[ "$SERVE_ONLY" -eq 0 ] && echo "  Download : ${#DL_SELECTED[@]} model(s) selected"
echo "  Serve    : ${#RUN_SELECTED[@]} model(s) selected"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL IDLE-SLEEP TIMEOUT — prompt for each selected servable model
# Models offload weights to CPU after N idle minutes (auto-wake on next request).
# Default shown in brackets: catalog SLEEP_MIN override if set, else IDLE_SLEEP_MINUTES.
# ─────────────────────────────────────────────────────────────────────────────
_has_servable=0
for _idx in "${RUN_SELECTED[@]}"; do
    [ "${MDL_PORT[$_idx]}" = "0" ] && continue
    _has_servable=1; break
done

# Headless mode: never prompt — each model keeps its catalog SLEEP_MIN override,
# or falls back to the global IDLE_SLEEP_MINUTES (handled by the watchdog setup).
if [ "$_has_servable" = "1" ] && [ "$HEADLESS" -eq 0 ]; then
    echo "  ── Per-model idle-sleep timeout ─────────────────────────────────"
    echo "  Each model offloads weights to CPU after N idle minutes, then"
    echo "  auto-wakes on the next request.  Press Enter to accept the default."
    echo ""
    for _idx in "${RUN_SELECTED[@]}"; do
        [ "${MDL_PORT[$_idx]}" = "0" ] && continue
        _default="${MDL_SLEEP[$_idx]:-$IDLE_SLEEP_MINUTES}"
        [ -z "$_default" ] && _default=0
        printf "  %-48s [default %s min, 0=never]: " "${MDL_NAME[$_idx]}" "$_default"
        read -r _sleep_input
        if [ -z "$_sleep_input" ]; then
            MDL_SLEEP[$_idx]="$_default"
        elif [[ "$_sleep_input" =~ ^[0-9]+$ ]]; then
            MDL_SLEEP[$_idx]="$_sleep_input"
        else
            echo "  ⚠️  Invalid — using default ${_default} min."
            MDL_SLEEP[$_idx]="$_default"
        fi
    done
    echo ""
    echo "  Sleep timeouts confirmed:"
    for _idx in "${RUN_SELECTED[@]}"; do
        [ "${MDL_PORT[$_idx]}" = "0" ] && continue
        _mins="${MDL_SLEEP[$_idx]:-0}"
        if [ "$_mins" -eq 0 ] 2>/dev/null; then
            printf "    %-48s : never\n" "${MDL_NAME[$_idx]}"
        else
            printf "    %-48s : %d min\n" "${MDL_NAME[$_idx]}" "$_mins"
        fi
    done
    echo "  ─────────────────────────────────────────────────────────────────"
    echo ""
fi

if [ "$SERVE_ONLY" -eq 1 ]; then
    # ── Serve-only mode: skip all install/download steps ──────────────────────
    echo "⏭️  --serve-only: skipping apt, docker, venv, NeMo, HF, and download steps."
    echo ""

    # Find an existing venv so the VLLM_BIN search has a venv path to try first.
    VENV_DIR=""
    for candidate in "$VLLM_VENV" "$HOME/vllm-install/.vllm" "/home/cgray/vllm-install/.vllm"; do
        if [ -x "$candidate/bin/python" ]; then
            VENV_DIR="$candidate"
            echo "✅ Using existing venv at $VENV_DIR"
            break
        fi
    done
    [ -z "$VENV_DIR" ] && VENV_DIR="$VLLM_VENV"  # fallback; VLLM_BIN PATH search will cover it

else
    # ── Full install mode ──────────────────────────────────────────────────────
    sudo apt update
    sudo apt install -y --no-install-recommends wget curl gnupg2 git libgl1 libglib2.0-0
    sudo apt install -y jq
    sudo apt install -y python3.12-dev python3-dev build-essential ninja-build

    #-------- Docker / Containers ------------
    if command -v docker >/dev/null 2>&1; then
        echo "✅ Docker is installed. Version: $(docker --version)"
    else
        echo "❌ Docker is not installed."
        echo " You need docker first before running this. This will download a docker installer and run it for you. "
        wget -O "install_docker.sh" https://raw.githubusercontent.com/c2theg/srvBuilds/refs/heads/master/install_docker.sh
        chmod u+x install_docker.sh
        ./install_docker.sh
    fi

    #--- SETUP vLLM on DGX Spark ---
    curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash

    VENV_PIP=""
    VENV_DIR=""
    for candidate in "$VLLM_VENV" "$HOME/vllm-install/.vllm" "/home/cgray/vllm-install/.vllm"; do
        if [ -x "$candidate/bin/pip" ]; then
            VENV_PIP="$candidate/bin/pip"
            VENV_DIR="$candidate"
            echo "✅ Using vLLM venv at $candidate"
            break
        fi
    done

    if [ -z "$VENV_PIP" ]; then
        echo "⚠️  vLLM venv not found — creating dedicated downloader venv at $VLLM_VENV"
        python3 -m venv "$VLLM_VENV"
        VENV_PIP="$VLLM_VENV/bin/pip"
        VENV_DIR="$VLLM_VENV"
    fi

    if ! "$VENV_DIR/bin/python" -c "import vllm" 2>/dev/null; then
        echo "⚠️  vllm not found in venv — installing via pip..."
        "$VENV_PIP" install -U vllm
        if "$VENV_DIR/bin/python" -c "import vllm" 2>/dev/null; then
            echo "✅ vllm installed successfully"
        else
            echo "❌ vllm install failed — check pip output above"
        fi
    else
        echo "✅ vllm already installed: $("$VENV_DIR/bin/python" -c 'import vllm; print(vllm.__version__)')"
    fi

    "$VENV_PIP" install -U "huggingface_hub[cli]" sentence-transformers

    if [ -x "$VENV_DIR/bin/hf" ]; then
        HF_CLI="$VENV_DIR/bin/hf"
        HF_LOGIN="$HF_CLI auth login"
    else
        HF_CLI="$VENV_DIR/bin/huggingface-cli"
        HF_LOGIN="$HF_CLI login"
    fi
    echo "✅ Using HF CLI: $HF_CLI"

    python3 -m venv "$NEMO_VENV"
    "$NEMO_VENV/bin/pip" install -U pip
    "$NEMO_VENV/bin/pip" install "nemo_toolkit[asr]"

    if [ -n "$HF_TOKEN" ]; then
        $HF_LOGIN --token "$HF_TOKEN"
        HF_AUTH="--token $HF_TOKEN"
    else
        echo "⚠️  HF_TOKEN not set — gated models will fail."
        HF_AUTH=""
    fi

    mkdir -p "$MODELS_DIR"
    HF_DL="$HF_CLI download $HF_AUTH"

    # ── Download selected models ───────────────────────────────────────────────
    if [ "${#DL_SELECTED[@]}" -eq 0 ]; then
        echo "⏭️  No models selected for download — skipping."
    else
        for idx in "${DL_SELECTED[@]}"; do
            echo ""
            echo "--- Downloading ${MDL_NAME[$idx]} ---"
            echo "    HF repo  : ${MDL_HF[$idx]}"
            echo "    Local dir: $MODELS_DIR/${MDL_DIR[$idx]}"
            if [ "${MDL_CAT[$idx]}" = "Super Large" ]; then
                echo "    ⚠️  SUPER LARGE model (~${MDL_DISK[$idx]} GB) — this will take a while."
                echo "    ℹ️  Nemotron-3-Super info: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard"
            fi
            if $HF_DL "${MDL_HF[$idx]}" --local-dir "$MODELS_DIR/${MDL_DIR[$idx]}"; then
                echo "✅ ${MDL_NAME[$idx]} downloaded"
            else
                echo "❌ ${MDL_NAME[$idx]} download FAILED (see error above) — skipping."
            fi
        done
    fi

    echo ""
    echo "✅ All selected models downloaded to $MODELS_DIR"
fi  # end SERVE_ONLY check

# ─────────────────────────────────────────────────────────────────────────────
# SERVE selected models with vLLM
# ─────────────────────────────────────────────────────────────────────────────
VLLM_LOGS="$BASE_DIR/logs"
mkdir -p "$VLLM_LOGS"
echo "✅ Log directory: $VLLM_LOGS"

VLLM_BIN=""
for candidate in \
    "$VENV_DIR/bin/vllm" \
    "$HOME/vllm-install/.vllm/bin/vllm" \
    "$HOME/.local/bin/vllm" \
    "/usr/local/bin/vllm" \
    "$(find "$HOME/vllm-install" -name vllm -type f 2>/dev/null | head -1)"; do
    if [ -x "$candidate" ]; then
        VLLM_BIN="$candidate"
        echo "✅ Found vllm binary at $VLLM_BIN"
        break
    fi
done

if [ -z "$VLLM_BIN" ]; then
    echo "⚠️  vllm not found in venv candidates — checking PATH..."
    if command -v vllm &>/dev/null; then
        VLLM_BIN="$(command -v vllm)"
        echo "✅ Found vllm on PATH: $VLLM_BIN"
    else
        echo "❌ CRITICAL: vllm binary not found anywhere."
        echo "   Searched venv: $VENV_DIR"
        echo "   Run: source $VENV_DIR/bin/activate && pip install -U vllm"
        echo "   Serve section will be skipped."
    fi
fi

vllm_serve() {
    if [ -n "$VLLM_BIN" ]; then
        "$VLLM_BIN" serve "$@"
    else
        echo "⚠️  vllm not found — trying python module fallback"
        "$VENV_DIR/bin/python" -m vllm.entrypoints.openai.api_server "$@"
    fi
}

# Helper: launch one model, echo command, wait 2s, confirm process is alive
# Usage: _vllm_launch <catalog_idx> [extra vllm args...]
_vllm_launch() {
    local idx="$1"; shift
    local name="${MDL_NAME[$idx]}"
    local dir="${MDL_DIR[$idx]}"
    local port="${MDL_PORT[$idx]}"
    local model_path="$MODELS_DIR/$dir"
    local log_file="$VLLM_LOGS/vllm-${port}.log"

    if [ ! -f "$model_path/config.json" ]; then
        echo "⚠️  [idx $idx] $name — model not found at $model_path"
        echo "     Was it downloaded? Re-run and select this model in Step 1."
        return 1
    fi

    local vllm_label
    if [ -n "$VLLM_BIN" ]; then
        vllm_label="$VLLM_BIN serve"
    else
        vllm_label="python3 -m vllm.entrypoints.openai.api_server"
    fi

    echo ""
    echo "--- Starting [idx $idx] $name on port $port ---"
    echo "    Model : $model_path"
    echo "    Log   : $log_file"
    echo "    CMD   : $vllm_label $model_path --host 0.0.0.0 --port $port --enable-sleep-mode $*"

    vllm_serve "$model_path" --host 0.0.0.0 --port "$port" --enable-sleep-mode "$@" >> "$log_file" 2>&1 &
    local launch_pid=$!
    sleep 2
    if ! kill -0 "$launch_pid" 2>/dev/null; then
        echo "⚠️  $name (pid $launch_pid) exited immediately — last 30 lines of log:"
        tail -30 "$log_file" 2>/dev/null | sed 's/^/   | /'
        echo "   → Full log: cat $log_file"
        return 1
    fi

    echo "✅ $name launched  pid=$launch_pid  port=$port"
    echo "   → Log: tail -f $log_file"
    echo "   ⏳ Waiting for model to finish loading before starting the next one..."

    # Poll every 5s (was 15s) so small models are detected ready up to ~14s
    # sooner; print progress only every 30s so the log stays as quiet as before.
    # /health is cheaper than /v1/models and returns 200 once the engine is up.
    local elapsed=0 timeout=720 poll=5  # 12-minute max (large models take ~5-10 min)
    while true; do
        if ! kill -0 "$launch_pid" 2>/dev/null; then
            echo "   ❌ $name process died during loading — last 30 lines:"
            tail -30 "$log_file" 2>/dev/null | sed 's/^/   | /'
            return 1
        fi
        if curl -sf --max-time 5 "http://localhost:${port}/health" > /dev/null 2>&1 || \
           curl -sf --max-time 5 "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "   ✅ $name ready on port $port  (${elapsed}s)"
            echo "   → Status: curl -s http://localhost:${port}/v1/models | jq ."
            return 0
        fi
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "   ⚠️  $name not ready after ${timeout}s — moving on; check: tail -f $log_file"
            return 0
        fi
        sleep "$poll"
        elapsed=$((elapsed + poll))
        [ $((elapsed % 30)) -eq 0 ] && printf "   [%ds] still loading...\n" "$elapsed"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# SLEEP WATCHDOG — polls vLLM /metrics and offloads idle models to CPU
# Args: one "port:idle_seconds" pair per model to watch. The idle threshold is
#       per-port, so models can carry different sleep timeouts (catalog SLEEP_MIN).
# ─────────────────────────────────────────────────────────────────────────────

# Build _WATCH_PAIRS ("port:idle_seconds") from RUN_SELECTED. Each model uses its
# MDL_SLEEP value (catalog override or the runtime-prompted value), falling back
# to the global IDLE_SLEEP_MINUTES; models with an effective 0 are skipped.
_build_watch_pairs() {
    _WATCH_PAIRS=()
    local _idx _p _mins
    for _idx in "${RUN_SELECTED[@]}"; do
        _p="${MDL_PORT[$_idx]}"
        [ "$_p" = "0" ] && continue
        _mins="${MDL_SLEEP[$_idx]:-}"
        [ -z "$_mins" ] && _mins="$IDLE_SLEEP_MINUTES"
        [ "$_mins" -gt 0 ] 2>/dev/null || continue
        _WATCH_PAIRS+=("${_p}:$((_mins * 60))")
    done
}

_start_sleep_watchdog() {
    local -a watch_pairs=("$@")
    [ "${#watch_pairs[@]}" -eq 0 ] && return 0

    # Build the port list and the port→idle-seconds map from the pairs.
    local ports_str="" idle_map=""
    for pair in "${watch_pairs[@]}"; do
        local p="${pair%%:*}" s="${pair##*:}"
        ports_str="${ports_str}${ports_str:+ }${p}"
        idle_map="${idle_map}${idle_map:+ }[${p}]=${s}"
    done

    local watchdog_script="$VLLM_LOGS/sleep_watchdog.sh"
    cat > "$watchdog_script" << 'WATCHDOG_EOF'
#!/usr/bin/env bash
# vLLM sleep watchdog — generated by install_ai_spark_vllm.sh
# Per-port idle threshold (seconds) — a model sleeps once idle past its own value.
declare -A IDLE_SECS_BY_PORT=( __IDLE_MAP__ )
PORTS=(__PORTS__)
declare -A last_request_count
declare -A last_active

for port in "${PORTS[@]}"; do
    cnt=$(curl -sf --max-time 5 "http://localhost:${port}/metrics" 2>/dev/null \
        | awk '/^vllm:e2e_request_latency_seconds_count/{sum+=$2} END{print int(sum+0)}')
    last_request_count[$port]="${cnt:-0}"
    last_active[$port]=$(date +%s)
done

while true; do
    sleep 60
    now=$(date +%s)
    for port in "${PORTS[@]}"; do
        curl -sf --max-time 3 "http://localhost:${port}/health" > /dev/null 2>&1 || continue
        new_cnt=$(curl -sf --max-time 5 "http://localhost:${port}/metrics" 2>/dev/null \
            | awk '/^vllm:e2e_request_latency_seconds_count/{sum+=$2} END{print int(sum+0)}')
        new_cnt="${new_cnt:-0}"
        idle_secs="${IDLE_SECS_BY_PORT[$port]:-900}"
        if [ "$new_cnt" != "${last_request_count[$port]:-0}" ]; then
            last_request_count[$port]="$new_cnt"
            last_active[$port]=$now
        else
            idle=$(( now - ${last_active[$port]:-$now} ))
            if [ "$idle" -ge "$idle_secs" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] port ${port}: idle $((idle/60))m (threshold $((idle_secs/60))m) — sleeping (offloading to CPU)"
                # Try /v1/sleep first (vLLM ≥0.20.x), fall back to /sleep (older)
                if curl -sf -X POST "http://localhost:${port}/v1/sleep" > /dev/null 2>&1 || \
                   curl -sf -X POST "http://localhost:${port}/sleep"    > /dev/null 2>&1; then
                    last_active[$port]=$now
                fi
            fi
        fi
    done
done
WATCHDOG_EOF

    sed -i \
        -e "s|__IDLE_MAP__|${idle_map}|g" \
        -e "s|__PORTS__|${ports_str}|g" \
        "$watchdog_script"
    chmod +x "$watchdog_script"

    nohup "$watchdog_script" >> "$VLLM_LOGS/sleep_watchdog.log" 2>&1 &
    local wpid=$!
    echo "✅ Sleep watchdog started  pid=$wpid"
    echo "   Per-port idle thresholds (min):"
    for pair in "${watch_pairs[@]}"; do
        printf "     port %-6s : %d min\n" "${pair%%:*}" "$(( ${pair##*:} / 60 ))"
    done
    echo "   Log: tail -f $VLLM_LOGS/sleep_watchdog.log"
    echo "   Sleep API: POST http://localhost:<port>/sleep  |  POST .../wake_up"
}

# ─────────────────────────────────────────────────────────────────────────────
# SQLITE STRUCTURED MEMORY — exact fact storage, 2-month / 100 MB retention
# ─────────────────────────────────────────────────────────────────────────────
_setup_sqlite_memory() {
    mkdir -p "$MEMORY_DIR"
    if ! command -v sqlite3 &>/dev/null; then
        echo "⚠️  sqlite3 not found — installing..."
        sudo apt install -y sqlite3
    fi

    sqlite3 "$SQLITE_DB" << 'SQL_EOF'
CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    category    TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    source      TEXT,
    confidence  REAL DEFAULT 1.0,
    tags        TEXT,
    UNIQUE(category, key)
);
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
    content     TEXT NOT NULL,
    model       TEXT,
    tokens_used INTEGER
);
CREATE INDEX IF NOT EXISTS idx_facts_created  ON facts(created_at);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_conv_created   ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_conv_session   ON conversations(session_id);
SQL_EOF

    echo "✅ SQLite structured memory initialized: $SQLITE_DB"

    cat > "$MEMORY_DIR/sqlite_maintain.sh" << MAINT_EOF
#!/usr/bin/env bash
DB="$SQLITE_DB"
MAX_MB=$SQLITE_MAX_MB
RETENTION_DAYS=$SQLITE_RETENTION_DAYS

sqlite3 "\$DB" "DELETE FROM facts         WHERE created_at < datetime('now', '-\${RETENTION_DAYS} days');"
sqlite3 "\$DB" "DELETE FROM conversations WHERE created_at < datetime('now', '-\${RETENTION_DAYS} days');"
sqlite3 "\$DB" "VACUUM;"

SIZE_MB=\$(du -sm "\$DB" 2>/dev/null | cut -f1)
while [ "\${SIZE_MB:-0}" -gt "\$MAX_MB" ]; do
    sqlite3 "\$DB" "DELETE FROM facts         WHERE id IN (SELECT id FROM facts         ORDER BY created_at ASC LIMIT 500);"
    sqlite3 "\$DB" "DELETE FROM conversations WHERE id IN (SELECT id FROM conversations ORDER BY created_at ASC LIMIT 500);"
    sqlite3 "\$DB" "VACUUM;"
    SIZE_MB=\$(du -sm "\$DB" 2>/dev/null | cut -f1)
done
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] SQLite maintenance done. Size: \${SIZE_MB:-?}MB / ${SQLITE_MAX_MB}MB max"
MAINT_EOF

    chmod +x "$MEMORY_DIR/sqlite_maintain.sh"
    (crontab -l 2>/dev/null | grep -v "sqlite_maintain"; \
     echo "0 3 * * * $MEMORY_DIR/sqlite_maintain.sh >> $VLLM_LOGS/sqlite_maintain.log 2>&1") | crontab -
    echo "✅ SQLite retention: ${SQLITE_RETENTION_DAYS} days / ${SQLITE_MAX_MB} MB — daily cleanup at 3am"
    echo "   DB path : $SQLITE_DB"
    echo "   Maintain: $MEMORY_DIR/sqlite_maintain.sh"
}

# ─────────────────────────────────────────────────────────────────────────────
# QDRANT VECTOR MEMORY — semantic memory, 2-month / 1 GB retention
# ─────────────────────────────────────────────────────────────────────────────
_setup_qdrant() {
    mkdir -p "$QDRANT_DATA_DIR/storage" "$QDRANT_DATA_DIR/config"

    cat > "$QDRANT_DATA_DIR/config/production.yaml" << 'QDRANT_CFG_EOF'
storage:
  storage_path: /qdrant/storage
service:
  http_port: 6333
  grpc_port: 6334
QDRANT_CFG_EOF

    docker pull qdrant/qdrant:latest
    docker run -d \
        --name qdrant \
        --network host \
        -v "$QDRANT_DATA_DIR/storage:/qdrant/storage:rw" \
        -v "$QDRANT_DATA_DIR/config:/qdrant/config:ro" \
        qdrant/qdrant:latest

    echo "✅ Qdrant vector DB started"
    echo "   HTTP API : http://localhost:${QDRANT_HTTP_PORT}"
    echo "   gRPC     : localhost:${QDRANT_GRPC_PORT}"
    echo "   Dashboard: http://localhost:${QDRANT_HTTP_PORT}/dashboard"

    cat > "$QDRANT_DATA_DIR/qdrant_maintain.sh" << QDRANT_MAINT_EOF
#!/usr/bin/env bash
QDRANT_URL="http://localhost:${QDRANT_HTTP_PORT}"
MAX_GB=$QDRANT_MAX_GB
DATA_DIR="$QDRANT_DATA_DIR/storage"
CUTOFF=\$(date -d "-${QDRANT_RETENTION_DAYS} days" +%s 2>/dev/null || \
          date -v -${QDRANT_RETENTION_DAYS}d     +%s 2>/dev/null)

for col in \$(curl -sf "\$QDRANT_URL/collections" 2>/dev/null \
             | jq -r '.result.collections[].name' 2>/dev/null); do
    curl -sf -X POST "\$QDRANT_URL/collections/\${col}/points/delete" \
        -H "Content-Type: application/json" \
        -d "{\"filter\":{\"must\":[{\"key\":\"created_at\",\"range\":{\"lt\":\$CUTOFF}}]}}" > /dev/null
done

SIZE_BYTES=\$(du -sb "\$DATA_DIR" 2>/dev/null | cut -f1)
SIZE_GB=\$(( \${SIZE_BYTES:-0} / 1073741824 ))
if [ "\${SIZE_GB:-0}" -gt "\$MAX_GB" ]; then
    echo "[\$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Qdrant storage \${SIZE_GB}GB exceeds \${MAX_GB}GB limit"
    echo "   Prune collections manually or reduce QDRANT_RETENTION_DAYS."
fi
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Qdrant maintenance done. Storage: \${SIZE_GB:-?}GB / ${QDRANT_MAX_GB}GB max"
QDRANT_MAINT_EOF

    chmod +x "$QDRANT_DATA_DIR/qdrant_maintain.sh"
    (crontab -l 2>/dev/null | grep -v "qdrant_maintain"; \
     echo "0 4 * * * $QDRANT_DATA_DIR/qdrant_maintain.sh >> $VLLM_LOGS/qdrant_maintain.log 2>&1") | crontab -
    echo "✅ Qdrant retention: ${QDRANT_RETENTION_DAYS} days / ${QDRANT_MAX_GB} GB — daily cleanup at 4am"
    echo "   Note: store 'created_at' (Unix epoch) as a payload field in each point for age-based pruning."
    echo "   Maintain: $QDRANT_DATA_DIR/qdrant_maintain.sh"
}

# Headless mode is additive: it must not kill models/containers another run (or
# another cron entry) already started, so the clean start only runs interactively.
if [ "$HEADLESS" -eq 0 ]; then
    echo "--- Clean start: killing all vLLM processes and removing old logs ---"
    docker stop open-webui searxng qdrant 2>/dev/null || true
    docker rm   open-webui searxng qdrant 2>/dev/null || true
    _kill_vllm_processes
    sleep 3
    rm -f "$VLLM_LOGS"/vllm-*.log
    echo "✅ Old vLLM processes killed and logs cleared"
fi

export TORCH_FLOAT32_MATMUL_PRECISION=high

if [ "${#RUN_SELECTED[@]}" -eq 0 ]; then
    echo "⏭️  No models selected to serve — skipping vLLM startup."
else
    echo ""
    echo "  ── Models queued to serve ───────────────────────────────────────"
    for idx in "${RUN_SELECTED[@]}"; do
        [ "${MDL_PORT[$idx]}" = "0" ] && continue
        printf "    [catalog idx %2d]  %-50s  port %s\n" \
            "$idx" "${MDL_NAME[$idx]}" "${MDL_PORT[$idx]}"
    done
    echo "  ─────────────────────────────────────────────────────────────────"
fi

# ─────────────────────────────────────────────────────────────────────────────
# --gpu-memory-utilization on unified memory (DGX Spark / GB10, ~121 GB shared):
# vLLM's KV-cache profiler measures TOTAL system GPU memory in use as the
# baseline — this includes weights from ALL other running vLLM processes, not
# just this one. So when a 35B model is already loaded (~38 GB), even a tiny
# model needs a budget > (38 + its own weights) GB, i.e. utilization > 0.38.
# That is why embedding/reranker models use 0.45-0.50 here despite being small:
# the fraction buys enough headroom over the 35B's footprint to allow KV cache
# allocation. These fractions "over-subscribe" the pool on paper but in practice
# only 1-2 models are hot at a time (sleep watchdog offloads the rest to CPU).
#   Rough guide on a 121 GB box:  0.40 ≈ 48 GB,  0.50 ≈ 61 GB,  0.75 ≈ 91 GB.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL SERVE ARGS — _serve_model dispatches on the HF REPO ID, not the
# catalog index, so inserting/reordering _add lines can never mis-map a model
# to the wrong serve block (the v0.1.5 index-mismatch bug class is gone).
# A servable model with no dedicated entry falls through to safe generic
# defaults in the * arm. ASR/NeMo download-only models (PORT=0) are skipped.
#   NeMo ASR usage: python3 -c "
#     import nemo.collections.asr as nemo_asr
#     model = nemo_asr.models.EncDecRNNTBPEModel.restore_from('$MODELS_DIR/parakeet-tdt-0.6b-v3/model.nemo')
#     print(model.transcribe(['your_audio.wav']))"
# ─────────────────────────────────────────────────────────────────────────────
_serve_model() {
    local idx="$1"
    [ "${MDL_PORT[$idx]}" = "0" ] && return 0

    case "${MDL_HF[$idx]}" in

    "Qwen/Qwen3.6-35B-A3B-FP8")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.6-35B-A3B" \
            --dtype auto \
            --gpu-memory-utilization 0.40 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    # NVFP4 (~4-bit): ~20 GB weights vs ~35 GB FP8 — 0.30 × 121 ≈ 36 GB budget
    # leaves ~14 GB KV headroom when served alone. vLLM auto-detects modelopt
    # NVFP4 from the checkpoint config; the explicit flag matches the Nemotron
    # NVFP4 entry below — drop it if your vLLM build errors on it.
    "nvidia/Qwen3.6-35B-A3B-NVFP4")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.6-35B-A3B-NVFP4" \
            --dtype auto \
            --quantization modelopt_fp4 \
            --gpu-memory-utilization 0.30 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    # Unsloth's "Fast" repack of the same NVFP4 checkpoint — same footprint.
    "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.6-35B-A3B-NVFP4-Fast" \
            --dtype auto \
            --quantization modelopt_fp4 \
            --gpu-memory-utilization 0.30 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4")
        _vllm_launch "$idx" \
            --served-model-name "Nemotron-3-Nano-30B-NVFP4" \
            --dtype auto \
            --quantization modelopt_fp4 \
            --gpu-memory-utilization 0.20 \
            --max-model-len 32768 \
            --max-num-seqs 178 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    "Qwen/Qwen3-Coder-30B-A3B-Instruct")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3-Coder-30B" \
            --dtype auto \
            --gpu-memory-utilization 0.62 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code
        ;;

    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        _vllm_launch "$idx" \
            --served-model-name "DeepSeek-R1-Distill-Qwen-32B" \
            --dtype auto \
            --gpu-memory-utilization 0.65 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code
        ;;

    "google/gemma-4-31B-it")
        _vllm_launch "$idx" \
            --served-model-name "gemma-4-31B" \
            --dtype auto \
            --gpu-memory-utilization 0.60 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    "google/gemma-4-26B-A4B-it")
        _vllm_launch "$idx" \
            --served-model-name "gemma-4-26B-A4B" \
            --dtype auto \
            --gpu-memory-utilization 0.55 \
            --max-model-len 16384 \
            --max-num-batched-tokens 4096 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")
        _vllm_launch "$idx" \
            --served-model-name "Nemotron-3-Nano-Omni-30B-A3B" \
            --dtype bfloat16 \
            --gpu-memory-utilization 0.62 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code
        ;;

    # gpu-memory-utilization is intentionally high (0.45) for the tiny embed/
    # rerank models: on unified memory vLLM's KV-cache profiler includes ALL
    # running processes' GPU footprint in the baseline, so a loaded 35B's ~38 GB
    # shows up here. Budget must exceed (38 + model) GB.
    # max-model-len capped at 8192 — embedding models default to 40960 (huge KV).
    "BAAI/bge-m3")
        _vllm_launch "$idx" \
            --served-model-name "bge-m3" \
            --dtype auto \
            --gpu-memory-utilization 0.45 \
            --max-model-len 8192 \
            --trust-remote-code
        ;;

    "Qwen/Qwen3-Embedding-4B")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3-Embedding-4B" \
            --dtype auto \
            --gpu-memory-utilization 0.60 \
            --max-model-len 8192 \
            --trust-remote-code
        ;;

    "BAAI/bge-reranker-v2-m3")
        _vllm_launch "$idx" \
            --served-model-name "bge-reranker-v2-m3" \
            --dtype auto \
            --gpu-memory-utilization 0.45 \
            --max-model-len 8192 \
            --trust-remote-code
        ;;

    # Generative yes/no reranker — clients score via logprobs on "yes"/"no"
    # tokens: POST /v1/completions with logprobs=1; compare P("yes") vs P("no").
    "Qwen/Qwen3-Reranker-4B")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3-Reranker-4B" \
            --dtype auto \
            --gpu-memory-utilization 0.55 \
            --max-model-len 10000 \
            --enable-prefix-caching \
            --max-logprobs 20 \
            --trust-remote-code
        ;;

    # ── SUPER LARGE (120B+): need nearly the whole GPU — don't co-run others ──
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16")
        echo "   ℹ️  Model info: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard"
        echo "   ⚠️  SUPER LARGE — needs ~115 GB VRAM. Ensure no other large models are running."
        _vllm_launch "$idx" \
            --served-model-name "Nemotron-3-Super-120B-A12B" \
            --dtype auto \
            --gpu-memory-utilization 0.93 \
            --max-model-len 8192 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    "Qwen/Qwen3.5-122B-A10B")
        echo "   ⚠️  SUPER LARGE — needs ~120 GB VRAM. Ensure no other large models are running."
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.5-122B-A10B" \
            --dtype auto \
            --gpu-memory-utilization 0.93 \
            --max-model-len 8192 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    # FP8 uses ~62 GB VRAM vs ~120 GB for BF16 — fits easily, longer context.
    "Qwen/Qwen3.5-122B-A10B-FP8")
        echo "   ★  FP8 version: ~62 GB VRAM, faster inference, longer context than BF16"
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.5-122B-A10B-FP8" \
            --dtype auto \
            --gpu-memory-utilization 0.75 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    "openai/gpt-oss-120b")
        echo "   ⚠️  SUPER LARGE — needs ~115 GB VRAM. Ensure no other large models are running."
        _vllm_launch "$idx" \
            --served-model-name "GPT-OSS-120B" \
            --dtype auto \
            --gpu-memory-utilization 0.93 \
            --max-model-len 8192 \
            --enable-prefix-caching \
            --trust-remote-code
        ;;

    # ── Small models (1-hour idle-sleep via catalog SLEEP_MIN=60) ─────────────
    "Qwen/Qwen3.5-4B")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.5-4B" \
            --dtype auto \
            --gpu-memory-utilization 0.50 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    "Qwen/Qwen3.5-2B")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.5-2B" \
            --dtype auto \
            --gpu-memory-utilization 0.45 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    # gpu-memory-utilization 0.75: when Nemotron-3-Nano-Omni-30B (~62 GB) is the
    # co-resident model, 0.75×121=90.75 GB budget leaves ~10 GB KV after weights.
    "Qwen/Qwen3.5-9B")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.5-9B" \
            --dtype auto \
            --gpu-memory-utilization 0.75 \
            --max-model-len 16384 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    # Served via vLLM's transcription task — POST /v1/audio/transcriptions.
    # STT endpoint, not chat: skipped in OpenWebUI chat auto-registration; wire
    # it in under Admin Settings → Audio → STT (URL http://localhost:8014/v1).
    "Qwen/Qwen3-ASR-1.7B")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3-ASR-1.7B" \
            --dtype auto \
            --gpu-memory-utilization 0.07 \
            --trust-remote-code
        ;;

    *)  # No dedicated serve entry — conservative generic defaults.
        echo "   ℹ️  ${MDL_HF[$idx]} has no dedicated serve entry — using generic defaults."
        _vllm_launch "$idx" \
            --served-model-name "${MDL_DIR[$idx]}" \
            --dtype auto \
            --gpu-memory-utilization 0.50 \
            --max-model-len 16384 \
            --trust-remote-code
        ;;
    esac
}

# ── Launch every selected servable model, one at a time ───────────────────────
for idx in "${RUN_SELECTED[@]}"; do
    _serve_model "$idx"
done

# ── Headless mode ends here: start the watchdog, show status, and exit ────────
# (containers, OpenWebUI registration, and memory setup are interactive-run-only)
if [ "$HEADLESS" -eq 1 ]; then
    _build_watch_pairs
    if [ "${#_WATCH_PAIRS[@]}" -gt 0 ]; then
        echo "--- Starting vLLM sleep watchdog ---"
        _start_sleep_watchdog "${_WATCH_PAIRS[@]}"
    fi
    _show_vllm_status "FINAL"
    echo "✅ Headless start complete."
    exit 0
fi

#---------------------------------------------------------------------------------------------------------------
#--- SearXNG (web search backend for OpenWebUI) ---
if [ "$ENABLE_SEARXNG" = "true" ]; then
    echo "--- Starting SearXNG container ---"
    mkdir -p "$BASE_DIR/searxng"

    if [ ! -f "$BASE_DIR/searxng/settings.yml" ]; then
        SEARXNG_SECRET=$(openssl rand -hex 32 2>/dev/null || echo "change-me-$(date +%s)")
        cat > "$BASE_DIR/searxng/settings.yml" << SEARXNG_EOF
use_default_settings: true

server:
  secret_key: "$SEARXNG_SECRET"
  bind_address: "0.0.0.0:$SEARXNG_PORT"

search:
  formats:
    - html
    - json
SEARXNG_EOF
        echo "✅ SearXNG settings.yml created at $BASE_DIR/searxng/settings.yml"
    fi

    docker pull searxng/searxng:latest
    docker run -d \
        --name searxng \
        --network host \
        -v "$BASE_DIR/searxng:/etc/searxng:rw" \
        searxng/searxng:latest
    echo "✅ SearXNG starting on http://localhost:$SEARXNG_PORT"
fi

#--- Start OpenWebUI ---
echo "--- Starting OpenWebUI container ---"
docker pull ghcr.io/open-webui/open-webui:main

# Pick the first served NON-ASR model's port as the primary OpenWebUI chat
# endpoint. ASR models are transcription endpoints (not chat), so they must not
# become the primary — they are also skipped in the registration loop below.
OWUI_PRIMARY_PORT=8005
for first_run_idx in "${RUN_SELECTED[@]}"; do
    [ "${MDL_PORT[$first_run_idx]}" = "0" ] && continue
    [ "${MDL_CAT[$first_run_idx]}" = "ASR" ] && continue
    OWUI_PRIMARY_PORT="${MDL_PORT[$first_run_idx]}"
    break
done

OWUI_ENV_ARGS=(
    -e PORT=3000
    -e "OPENAI_API_BASE_URL=http://localhost:${OWUI_PRIMARY_PORT}/v1"
    -e OPENAI_API_KEY=sk-no-key-required
)

if [ -n "$BRAVE_SEARCH_API_KEY" ]; then
    echo "   → Web search: Brave Search API"
    OWUI_ENV_ARGS+=(-e ENABLE_RAG_WEB_SEARCH=true -e WEB_SEARCH_ENGINE=brave -e "BRAVE_SEARCH_API_KEY=$BRAVE_SEARCH_API_KEY")
elif [ "$ENABLE_SEARXNG" = "true" ]; then
    echo "   → Web search: SearXNG (port $SEARXNG_PORT)"
    OWUI_ENV_ARGS+=(-e ENABLE_RAG_WEB_SEARCH=true -e WEB_SEARCH_ENGINE=searxng -e "SEARXNG_QUERY_URL=http://localhost:${SEARXNG_PORT}/search?q=<query>&format=json")
else
    echo "   → Web search: disabled"
fi

docker run -d \
    --name open-webui \
    --network host \
    -v open-webui:/app/backend/data \
    "${OWUI_ENV_ARGS[@]}" \
    ghcr.io/open-webui/open-webui:main

echo "Waiting for OpenWebUI to be ready..."
OWUI_TIMEOUT=300
OWUI_ELAPSED=0
until curl -sf http://localhost:3000/health > /dev/null 2>&1; do
    if [ "$OWUI_ELAPSED" -ge "$OWUI_TIMEOUT" ]; then
        echo ""
        echo "⚠️  OpenWebUI did not become ready after ${OWUI_TIMEOUT}s — check: docker logs open-webui"
        break
    fi
    printf "  [%ds] waiting...\n" "$OWUI_ELAPSED"
    sleep 5
    OWUI_ELAPSED=$((OWUI_ELAPSED + 5))
done

if [ "$OWUI_ELAPSED" -lt "$OWUI_TIMEOUT" ]; then
    echo "✅ OpenWebUI ready at http://localhost:3000"

    # Build URL list dynamically from whatever is actually running
    OWUI_URLS=""
    OWUI_KEYS=""
    OWUI_MANUAL=""
    _owui_add() {
        if [ -z "$OWUI_URLS" ]; then
            OWUI_URLS="\"$1\""; OWUI_KEYS='"sk-no-key-required"'
        else
            OWUI_URLS="$OWUI_URLS,\"$1\""; OWUI_KEYS="$OWUI_KEYS,\"sk-no-key-required\""
        fi
        OWUI_MANUAL="$OWUI_MANUAL\n     $1   ($2)"
    }

    for idx in "${RUN_SELECTED[@]}"; do
        port="${MDL_PORT[$idx]}"
        [ "$port" = "0" ] && continue
        # ASR models expose /v1/audio/transcriptions, not chat — registering them
        # as chat connections would create a broken entry. Wire them into
        # OpenWebUI under Admin → Audio → STT instead.
        if [ "${MDL_CAT[$idx]}" = "ASR" ]; then
            echo "   ℹ️  ${MDL_NAME[$idx]} (port ${port}) is an ASR/transcription endpoint —"
            echo "      skipping chat registration. Add it under Admin → Audio → STT:"
            echo "      OpenAI-compatible URL: http://localhost:${port}/v1"
            continue
        fi
        dir="${MDL_DIR[$idx]}"
        if [ -f "$MODELS_DIR/$dir/config.json" ]; then
            _owui_add "http://localhost:${port}/v1" "${MDL_NAME[$idx]}"
        fi
    done

    if [ -n "$OWUI_ADMIN_EMAIL" ] && [ -n "$OWUI_ADMIN_PASSWORD" ] && [ -n "$OWUI_URLS" ]; then
        echo "--- Auto-registering model connections in OpenWebUI ---"
        OWUI_TOKEN=$(curl -sf -X POST http://localhost:3000/api/v1/auths/signin \
            -H "Content-Type: application/json" \
            -d "{\"email\":\"$OWUI_ADMIN_EMAIL\",\"password\":\"$OWUI_ADMIN_PASSWORD\"}" \
            | jq -r '.token // empty')

        if [ -z "$OWUI_TOKEN" ]; then
            echo "   Sign-in failed — attempting to create admin account..."
            curl -sf -X POST http://localhost:3000/api/v1/auths/signup \
                -H "Content-Type: application/json" \
                -d "{\"name\":\"Admin\",\"email\":\"$OWUI_ADMIN_EMAIL\",\"password\":\"$OWUI_ADMIN_PASSWORD\"}" \
                > /dev/null
            OWUI_TOKEN=$(curl -sf -X POST http://localhost:3000/api/v1/auths/signin \
                -H "Content-Type: application/json" \
                -d "{\"email\":\"$OWUI_ADMIN_EMAIL\",\"password\":\"$OWUI_ADMIN_PASSWORD\"}" \
                | jq -r '.token // empty')
        fi

        if [ -n "$OWUI_TOKEN" ]; then
            curl -sf -X POST http://localhost:3000/api/v1/openai/config/update \
                -H "Authorization: Bearer $OWUI_TOKEN" \
                -H "Content-Type: application/json" \
                -d "{\"ENABLE_OPENAI_API\":true,\"OPENAI_API_BASE_URLS\":[$OWUI_URLS],\"OPENAI_API_KEYS\":[$OWUI_KEYS]}" \
                > /dev/null
            echo "✅ All model connections registered in OpenWebUI"
            printf "   Registered:%b\n" "$OWUI_MANUAL"
        else
            echo "⚠️  OpenWebUI login failed — check OWUI_ADMIN_EMAIL / OWUI_ADMIN_PASSWORD"
        fi
    elif [ -z "$OWUI_URLS" ]; then
        echo "⚠️  No models are running — nothing to register in OpenWebUI."
    else
        echo ""
        echo "   Set OWUI_ADMIN_EMAIL and OWUI_ADMIN_PASSWORD to auto-register connections."
        echo "   Or add them manually: Admin Settings → Connections → + Add Connection"
        printf "%b\n" "$OWUI_MANUAL"
    fi

    echo ""
    echo "  ⏳ Allow 5-10 minutes for vLLM models to finish loading before they appear."
fi

#---------------------------------------------------------------------------------------------------------------
#--- SQLite Structured Memory ---
if [ "$ENABLE_SQLITE_MEMORY" = "true" ]; then
    echo "--- Setting up SQLite structured memory ---"
    _setup_sqlite_memory
fi

#--- Qdrant Vector Memory ---
if [ "$ENABLE_QDRANT" = "true" ]; then
    echo "--- Starting Qdrant vector DB ---"
    _setup_qdrant
fi

#--- Sleep Watchdog ---
# Each model's effective idle timeout is its catalog SLEEP_MIN override, or the
# global IDLE_SLEEP_MINUTES when no override is set. A model with an effective
# timeout of 0 (or empty global) is skipped, so per-model timeouts still apply
# even if the global default is disabled.
if [ "${#RUN_SELECTED[@]}" -gt 0 ]; then
    _build_watch_pairs
    if [ "${#_WATCH_PAIRS[@]}" -gt 0 ]; then
        echo "--- Starting vLLM sleep watchdog ---"
        _start_sleep_watchdog "${_WATCH_PAIRS[@]}"
    else
        echo "--- Sleep watchdog skipped (no served model has an idle-sleep timeout) ---"
    fi
fi

echo ""
echo "--- Disk usage: $BASE_DIR ---"
du -sh "$BASE_DIR" 2>/dev/null
echo ""
echo "--- Per-model breakdown ---"
du -sh "$MODELS_DIR"/*/  2>/dev/null | sort -rh

echo ""
nvidia-smi

echo ""
echo "---- Monitor vLLM Startups ----"
echo "  Run any of the following to tail a model's log:"
for idx in "${RUN_SELECTED[@]}"; do
    port="${MDL_PORT[$idx]}"
    [ "$port" = "0" ] && continue
    echo "    ${MDL_NAME[$idx]} (port ${port}):"
    echo "      tail -f $VLLM_LOGS/vllm-${port}.log"
done
echo ""

# Final snapshot — memory + which models have come up so far.
_show_vllm_status "FINAL"
echo "  ℹ️  vLLM models take ~5-10 min to finish loading; any not listed above are"
echo "     likely still starting. Re-check later with: curl -s http://localhost:<port>/v1/models | jq ."
echo ""
