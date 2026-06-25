#!/usr/bin/env bash
# Christopher Gray  |  Version: 0.2.0  |  Update: 6/25/2026
# vLLM install, model download, and serve script for DGX Spark / NVIDIA systems
#
# Update Yourself:
#   wget --no-cache -O 'install_ai_spark_vllm.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_ai_spark_vllm.sh' && chmod u+x install_ai_spark_vllm.sh
#
# Usage:
#   ./install_ai_spark_vllm.sh              — full install: packages, docker, venv, download, serve
#   ./install_ai_spark_vllm.sh --serve-only — skip install/download; jump straight to model serve
#   ./install_ai_spark_vllm.sh -s           — same as --serve-only
#
# ── Changelog ─────────────────────────────────────────────────────────────────
#
# v0.2.0  6/25/2026
#   - Per-model sleep timeout: the catalog _add() helper now takes an optional
#     8th field SLEEP_MIN (idle minutes). When set, it overrides the global
#     IDLE_SLEEP_MINUTES for that model only. The sleep watchdog was refactored
#     from a single global IDLE_SECS to a per-port map (IDLE_SECS_BY_PORT), so
#     different models can sleep on different schedules. Models with a SLEEP_MIN
#     override sleep even if IDLE_SLEEP_MINUTES is 0 (global default disabled).
#   - Added Qwen/Qwen3.5-4B-Instruct  (catalog idx 17, port 8012, ~10 GB VRAM)
#     and Qwen/Qwen3.5-2B-Instruct    (catalog idx 18, port 8013, ~5 GB VRAM),
#     both with SLEEP_MIN=60 (offload to CPU after 1 hour idle). Appended at the
#     end of the catalog so existing indices and serve blocks are not shifted.
#   - Added nvidia/nemotron-3.5-asr-streaming-0.6b (catalog idx 19) as a
#     download-only ASR/NeMo model (VRAM=0, PORT=0) — not served via vLLM.
#   - Added Qwen/Qwen3-ASR-1.7B (catalog idx 20, port 8014, ~5 GB VRAM) served
#     via vLLM's --task transcription (POST /v1/audio/transcriptions). ASR-class
#     served models are skipped in the OpenWebUI chat auto-registration (they are
#     STT endpoints, not chat) — wire them in under Admin → Audio → STT instead.
#
# v0.1.9  6/25/2026
#   - Model sleep: vLLM servers now launch with --enable-sleep-mode; a watchdog
#     process monitors idle time and offloads model weights to CPU after
#     IDLE_SLEEP_MINUTES (default 15) of inactivity.  Models auto-wake on the
#     next inference request.  Watchdog log: $BASE_DIR/logs/sleep_watchdog.log
#   - Added Qwen/Qwen3-Reranker-4B (catalog idx 10, port 8021, ~9 GB VRAM)
#     Pre-selected by default (download + serve).  Generative yes/no reranker;
#     clients score via logprobs on the "yes"/"no" tokens.  Uses max-model-len
#     10000 and --max-logprobs 20 as recommended by HF model card.
#   - Added SQLite structured memory ($BASE_DIR/memory/structured_memory.db)
#     Schema: facts (key/value/category) + conversations tables.
#     Retention: 60 days OR 100 MB — daily cleanup cron at 3am.
#   - Added Qdrant vector DB container (port 6333 HTTP, 6334 gRPC)
#     Long-term semantic memory for facts, notes, docs, and past conversations.
#     Retention: 60 days OR 1 GB storage — daily cleanup cron at 4am.
#     Dashboard: http://localhost:6333/dashboard
#
# v0.1.8  5/26/2026
#   - Added Qwen/Qwen3.5-122B-A10B-FP8 (catalog idx 14, port 8033, ~62 GB VRAM)
#     FP8 cuts VRAM roughly in half vs BF16, allowing max-model-len 32768 and
#     much faster inference on the GB10 GPU.
#
# v0.1.7  5/26/2026
#   - Lowered --gpu-memory-utilization for SUPER LARGE models from 0.96 → 0.93
#     Root cause: GPU driver already occupies ~6 GiB at startup, so
#     0.96 × 121.69 GiB = 116.82 GiB > 115.48 GiB free → vLLM ValueError.
#     0.93 × 121.69 GiB = 113.17 GiB < 115.48 GiB free → passes pre-check.
#
# v0.1.6  5/26/2026
#   - Added --serve-only / -s flag: skips apt, docker, venv, NeMo, HF, and
#     download steps — jumps straight to model selection and vLLM serve.
#     Use this when models are already downloaded and you just want to (re)start.
#
# v0.1.5  5/26/2026
#   - Fixed index mismatch: gemma-4-31B-it was inserted at catalog pos 4 but
#     all serve blocks still used the pre-insertion indices, so every model from
#     pos 4 onward mapped to the wrong serve block and never launched
#   - Replaced hardcoded serve blocks with _vllm_launch helper that:
#       * echoes the exact command before running it
#       * waits 2s and checks if the process survived (tails log on fast exit)
#   - Added pre-serve summary listing each model, catalog index, and port
#   - Added VLLM_BIN null-check with fallback PATH search before attempting serve
#   - Added explicit log-dir creation with confirmation echo
#
# v0.1.4  5/26/2026
#   - Fixed OOM on SUPER LARGE models (120B+): reduced --max-model-len 32768→8192
#     and raised --gpu-memory-utilization 0.93→0.96 to leave room for KV cache
#     (128GB GPU with ~117GB of FP8 weights leaves only ~2GB for KV cache at
#      32768 context — 8192 is the safe max for single-GPU 120B+ models)
#
# v0.1.3  5/25/2026
#   - Added google/gemma-4-31B-it (idx 14, port 8009, ~62 GB BF16)
#
# v0.1.2  5/25/2026
#   - Fixed correct HuggingFace repo IDs for all three SUPER LARGE models:
#       nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
#       Qwen/Qwen3.5-122B-A10B
#       openai/gpt-oss-120b
#
# v0.1.1  5/25/2026
#   - Fixed script exiting immediately after banner: dropped -e from set -euo pipefail
#     ([ cond ] && action patterns return exit code 1 when condition is false, which
#      triggered immediate exit under strict -e mode)
#
# v0.1.0  5/25/2026
#   - Switched shebang from #!/bin/sh to #!/usr/bin/env bash (required for arrays)
#   - Interactive checkbox model selection: choose which models to download and serve
#   - VRAM pre-flight check — calculates total VRAM needed before launching anything
#   - Model catalog with disk size and VRAM estimates for every model
#   - SUPER LARGE section (120B+ params): Nemotron-3-Super, Qwen3.5-122B, GPT-OSS-120B
#   - Fixed log-redirection bug in all vllm_serve background calls (missing \ continuation)
#   - Replaced ENABLE_* booleans with runtime interactive selection
#   - OpenWebUI registration now loops dynamically over selected models
#
# v0.0.57  5/24/2026  (baseline before rewrite)
#   - Original static ENABLE_QWEN35 / ENABLE_NEMOTRON / ENABLE_GEMMA4 flags
# ──────────────────────────────────────────────────────────────────────────────

# ─── strict mode ──────────────────────────────────────────────────────────────
# -u: error on unset variables  -o pipefail: propagate pipeline failures
# -e (exit on error) is intentionally omitted — this script uses many
# [ cond ] && action patterns and || true guards that conflict with -e.
set -uo pipefail

SERVE_ONLY=0
[ "${1:-}" = "--serve-only" ] && SERVE_ONLY=1
[ "${1:-}" = "-s"           ] && SERVE_ONLY=1

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


Version:  0.2.0
Last Updated:  6/25/2026

Update Yourself:
    wget --no-cache -O 'install_ai_spark_vllm.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_ai_spark_vllm.sh' && chmod u+x install_ai_spark_vllm.sh


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
_add "Qwen/Qwen3.5-4B-Instruct"      "Qwen3.5-4B-Instruct"  "Qwen3.5-4B-Instruct (BF16) [1h sleep]"   8   10   8012  "General"   60
_add "Qwen/Qwen3.5-2B-Instruct"      "Qwen3.5-2B-Instruct"  "Qwen3.5-2B-Instruct (BF16) [1h sleep]"   4    5   8013  "General"   60

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

MODEL_TOTAL=${#MDL_HF[@]}

# ── Default pre-selected models ────────────────────────────────────────────────
# Qwen3-Reranker-4B (catalog idx 10) is pre-selected for download and serve.
# Users can deselect it in the interactive menus below.
DEFAULT_DL_INDICES=(10)
DEFAULT_SERVE_INDICES=(10)

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
        echo "  ⚠️  nvidia-smi not available — skipping VRAM check"
    fi
    echo "  ─────────────────────────────────────────────────────────"
}

is_dl_selected()  { for i in "${DL_SELECTED[@]}";  do [ "$i" = "$1" ] && return 0; done; return 1; }
is_run_selected() { for i in "${RUN_SELECTED[@]}"; do [ "$i" = "$1" ] && return 0; done; return 1; }

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
echo ""
echo "════════════════════════════════════════════════════════════════════"
if [ "$SERVE_ONLY" -eq 0 ]; then
    echo "  STEP 2 of 2 — Select models to SERVE with vLLM"
else
    echo "  Select models to SERVE with vLLM"
fi
echo "  (ASR/NeMo models are download-only and excluded from this list)"
echo "════════════════════════════════════════════════════════════════════"
RUN_SELECTED=()
_checkbox_menu "Models to serve with vLLM (toggle with numbers, d=done):" "true" RUN_SELECTED DEFAULT_SERVE_INDICES

_check_vram

echo ""
[ "$SERVE_ONLY" -eq 0 ] && echo "  Download : ${#DL_SELECTED[@]} model(s) selected"
echo "  Serve    : ${#RUN_SELECTED[@]} model(s) selected"
echo ""

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
            $HF_DL "${MDL_HF[$idx]}" --local-dir "$MODELS_DIR/${MDL_DIR[$idx]}"
            echo "✅ ${MDL_NAME[$idx]} downloaded"
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
    if kill -0 "$launch_pid" 2>/dev/null; then
        echo "✅ $name started  pid=$launch_pid  port=$port"
        echo "   → Logs  : tail -f $log_file"
        echo "   → Status: curl -s http://localhost:${port}/v1/models | jq ."
    else
        echo "⚠️  $name (pid $launch_pid) exited immediately — last 30 lines of log:"
        tail -30 "$log_file" 2>/dev/null | sed 's/^/   | /'
        echo "   → Full log: cat $log_file"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# SLEEP WATCHDOG — polls vLLM /metrics and offloads idle models to CPU
# Args: one "port:idle_seconds" pair per model to watch. The idle threshold is
#       per-port, so models can carry different sleep timeouts (catalog SLEEP_MIN).
# ─────────────────────────────────────────────────────────────────────────────
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
                curl -sf -X POST "http://localhost:${port}/sleep" > /dev/null 2>&1 && \
                    last_active[$port]=$now
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

echo "--- Clean start: killing all vLLM processes and removing old logs ---"
docker stop open-webui searxng qdrant 2>/dev/null || true
docker rm   open-webui searxng qdrant 2>/dev/null || true
pkill -9 -f "vllm serve"        2>/dev/null || true
pkill -9 -f "vllm.entrypoints"  2>/dev/null || true
pkill -9 -f "VLLM::EngineCore"  2>/dev/null || true
pkill -9 -f "vllm.engine"       2>/dev/null || true
sleep 3
rm -f "$VLLM_LOGS"/vllm-*.log
echo "✅ Old vLLM processes killed and logs cleared"

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

# NOTE: is_run_selected checks the actual catalog index stored in RUN_SELECTED.
# Catalog indices are assigned by _add() in order of appearance above — they
# must match here exactly, or the wrong model serve block will fire.

# ── catalog idx 0: Qwen3.6-35B-A3B-FP8  (port 8005) ──────────────────────────
if is_run_selected 0; then
    _vllm_launch 0 \
        --served-model-name "Qwen3.6-35B-A3B" \
        --dtype auto \
        --gpu-memory-utilization 0.73 \
        --max-model-len 32768 \
        --enable-prefix-caching \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --chat-template-kwargs '{"enable_thinking": false}'
fi

# ── catalog idx 1: NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4  (port 8006) ─────────
if is_run_selected 1; then
    _vllm_launch 1 \
        --served-model-name "Nemotron-3-Nano-30B-NVFP4" \
        --dtype auto \
        --quantization modelopt_fp4 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32768 \
        --max-num-seqs 178 \
        --enable-prefix-caching \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser hermes
fi

# ── catalog idx 2: Qwen3-Coder-30B-A3B-Instruct  (port 8001) ─────────────────
if is_run_selected 2; then
    _vllm_launch 2 \
        --served-model-name "Qwen3-Coder-30B" \
        --dtype auto \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32768 \
        --enable-prefix-caching \
        --trust-remote-code
fi

# ── catalog idx 3: DeepSeek-R1-Distill-Qwen-32B  (port 8002) ─────────────────
if is_run_selected 3; then
    _vllm_launch 3 \
        --served-model-name "DeepSeek-R1-Distill-Qwen-32B" \
        --dtype auto \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32768 \
        --enable-prefix-caching \
        --trust-remote-code
fi

# ── catalog idx 4: gemma-4-31B-it  (port 8009) ────────────────────────────────
if is_run_selected 4; then
    _vllm_launch 4 \
        --served-model-name "gemma-4-31B" \
        --dtype auto \
        --gpu-memory-utilization 0.60 \
        --max-model-len 32768 \
        --enable-prefix-caching \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser hermes
fi

# ── catalog idx 5: gemma-4-26B-A4B-it  (port 8007) ───────────────────────────
if is_run_selected 5; then
    _vllm_launch 5 \
        --served-model-name "gemma-4-26B-A4B" \
        --dtype auto \
        --gpu-memory-utilization 0.55 \
        --max-model-len 16384 \
        --max-num-batched-tokens 4096 \
        --enable-prefix-caching \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser hermes
fi

# ── catalog idx 6: Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16  (port 8008) ───
if is_run_selected 6; then
    _vllm_launch 6 \
        --served-model-name "Nemotron-3-Nano-Omni-30B-A3B" \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32768 \
        --enable-prefix-caching \
        --trust-remote-code
fi

# ── catalog idx 7: BAAI/bge-m3  (port 8011) ───────────────────────────────────
if is_run_selected 7; then
    _vllm_launch 7 \
        --served-model-name "bge-m3" \
        --task embedding \
        --dtype auto \
        --gpu-memory-utilization 0.30 \
        --trust-remote-code
fi

# ── catalog idx 8: Qwen3-Embedding-4B  (port 8010) ────────────────────────────
if is_run_selected 8; then
    _vllm_launch 8 \
        --served-model-name "Qwen3-Embedding-4B" \
        --task embedding \
        --dtype auto \
        --gpu-memory-utilization 0.50 \
        --trust-remote-code
fi

# ── catalog idx 9: bge-reranker-v2-m3  (port 8020) ────────────────────────────
if is_run_selected 9; then
    _vllm_launch 9 \
        --served-model-name "bge-reranker-v2-m3" \
        --task classify \
        --dtype auto \
        --gpu-memory-utilization 0.50 \
        --trust-remote-code
fi

# ── catalog idx 10: Qwen3-Reranker-4B  (port 8021) ★ Default ─────────────────
# Generative yes/no reranker — clients score via logprobs on "yes"/"no" tokens.
# Usage: POST /v1/completions with logprobs=1; compare P("yes") vs P("no").
if is_run_selected 10; then
    _vllm_launch 10 \
        --served-model-name "Qwen3-Reranker-4B" \
        --dtype auto \
        --gpu-memory-utilization 0.80 \
        --max-model-len 10000 \
        --enable-prefix-caching \
        --max-logprobs 20 \
        --trust-remote-code
fi

# ── catalog idx 11, 12 & 19: ASR / NeMo — download only, not served via vLLM ──
# (idx 19 = nemotron-3.5-asr-streaming-0.6b, appended at the end of the catalog)
# To use: python3 -c "
#   import nemo.collections.asr as nemo_asr
#   model = nemo_asr.models.EncDecRNNTBPEModel.restore_from('$MODELS_DIR/parakeet-tdt-0.6b-v3/model.nemo')
#   print(model.transcribe(['your_audio.wav']))"

# ─────────────────────────────────────────────────────────────────────────────
# SUPER LARGE MODELS (120B+ parameters)
# Info: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard
# ⚠️  These require nearly the entire GPU. Do NOT run alongside other large models.
# ─────────────────────────────────────────────────────────────────────────────

# ── catalog idx 13: Nemotron-3-Super-120B-A12B  (port 8030) ───────────────────
if is_run_selected 13; then
    echo "   ℹ️  Model info: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard"
    echo "   ⚠️  SUPER LARGE — needs ~115 GB VRAM. Ensure no other large models are running."
    _vllm_launch 13 \
        --served-model-name "Nemotron-3-Super-120B-A12B" \
        --dtype auto \
        --gpu-memory-utilization 0.93 \
        --max-model-len 8192 \
        --enable-prefix-caching \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser hermes
fi

# ── catalog idx 14: Qwen3.5-122B-A10B  (port 8031) ────────────────────────────
if is_run_selected 14; then
    echo "   ⚠️  SUPER LARGE — needs ~120 GB VRAM. Ensure no other large models are running."
    _vllm_launch 14 \
        --served-model-name "Qwen3.5-122B-A10B" \
        --dtype auto \
        --gpu-memory-utilization 0.93 \
        --max-model-len 8192 \
        --enable-prefix-caching \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser hermes
fi

# ── catalog idx 15: Qwen3.5-122B-A10B-FP8  (port 8033) ★ Recommended ──────────
# FP8 uses ~62 GB VRAM vs ~120 GB for BF16 — fits easily, allows longer context
if is_run_selected 15; then
    echo "   ★  FP8 version: ~62 GB VRAM, faster inference, longer context than BF16"
    _vllm_launch 15 \
        --served-model-name "Qwen3.5-122B-A10B-FP8" \
        --dtype auto \
        --gpu-memory-utilization 0.93 \
        --max-model-len 32768 \
        --enable-prefix-caching \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser hermes
fi

# ── catalog idx 16: GPT-OSS-120B  (port 8032) ─────────────────────────────────
if is_run_selected 16; then
    echo "   ⚠️  SUPER LARGE — needs ~115 GB VRAM. Ensure no other large models are running."
    _vllm_launch 16 \
        --served-model-name "GPT-OSS-120B" \
        --dtype auto \
        --gpu-memory-utilization 0.93 \
        --max-model-len 8192 \
        --enable-prefix-caching \
        --trust-remote-code
fi

# ─────────────────────────────────────────────────────────────────────────────
# SMALL MODELS WITH 1-HOUR IDLE-SLEEP  (catalog SLEEP_MIN=60)
# Same serve path as the standard models; the longer sleep timeout is handled
# entirely by the watchdog via each model's MDL_SLEEP entry (see catalog above).
# ─────────────────────────────────────────────────────────────────────────────

# ── catalog idx 17: Qwen3.5-4B-Instruct  (port 8012)  [1h idle-sleep] ─────────
if is_run_selected 17; then
    _vllm_launch 17 \
        --served-model-name "Qwen3.5-4B-Instruct" \
        --dtype auto \
        --gpu-memory-utilization 0.30 \
        --max-model-len 32768 \
        --enable-prefix-caching \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser hermes
fi

# ── catalog idx 18: Qwen3.5-2B-Instruct  (port 8013)  [1h idle-sleep] ─────────
if is_run_selected 18; then
    _vllm_launch 18 \
        --served-model-name "Qwen3.5-2B-Instruct" \
        --dtype auto \
        --gpu-memory-utilization 0.20 \
        --max-model-len 32768 \
        --enable-prefix-caching \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser hermes
fi

# ── catalog idx 20: Qwen3-ASR-1.7B  (port 8014)  [served as transcription] ────
# Served via vLLM's transcription task — exposes POST /v1/audio/transcriptions.
# NOTE: this is an audio/STT endpoint, not chat, so it is intentionally skipped
# in the OpenWebUI chat auto-registration below. To use it in OpenWebUI, set it
# under Admin Settings → Audio → STT (OpenAI-compatible URL http://localhost:8014/v1).
if is_run_selected 20; then
    _vllm_launch 20 \
        --served-model-name "Qwen3-ASR-1.7B" \
        --task transcription \
        --dtype auto \
        --gpu-memory-utilization 0.20 \
        --trust-remote-code
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
    _WATCH_PAIRS=()
    for _idx in "${RUN_SELECTED[@]}"; do
        _p="${MDL_PORT[$_idx]}"
        [ "$_p" = "0" ] && continue
        _mins="${MDL_SLEEP[$_idx]:-}"
        [ -z "$_mins" ] && _mins="$IDLE_SLEEP_MINUTES"
        [ "$_mins" -gt 0 ] 2>/dev/null || continue
        _WATCH_PAIRS+=("${_p}:$((_mins * 60))")
    done
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
