#!/usr/bin/env bash
# Christopher Gray  |  Version: 0.1.9  |  Update: 6/18/2026
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
# v0.1.9  6/18/2026
#   - Removed all four 120B+ "SUPER LARGE" models (Nemotron-3-Super-120B,
#     Qwen3.5-122B BF16+FP8, GPT-OSS-120B). On a 128 GB unified GB10 / DGX Spark
#     they load ~116 GB of weights then get OOM-killed (SIGKILL) during KV-cache
#     allocation — confirmed in the wild. They need a discrete ≥140 GB GPU.
#   - Fixed the VRAM pre-flight check to be unified-memory aware: nvidia-smi
#     reports no memory.total on the GB10, so it now falls back to total system
#     RAM (the real budget for weights + KV cache) and warns before an OOM.
#   - OpenWebUI readiness wait now fails fast if the container exits, and gates
#     the "ready"/auto-register block on a real /health check.
#   - Remaining catalog all fits the 128 GB GB10: Qwen3.6-35B-A3B-FP8,
#     Nemotron-3-Nano-30B-NVFP4, Gemma-4-31B, Gemma-4-26B-A4B, etc.
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


Version:  0.1.9
Last Updated:  6/18/2026

Update Yourself:
    wget --no-cache -O 'install_ai_spark.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_ai_spark.sh' && chmod u+x install_ai_spark.sh


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
# Fields: HF_REPO | LOCAL_DIR | DISPLAY_NAME | DISK_GB | VRAM_GB | PORT | CATEGORY
# VRAM_GB=0 → CPU/NeMo only — cannot be served via vLLM
# PORT=0    → download-only (ASR/NeMo)
# ─────────────────────────────────────────────────────────────────────────────
MDL_HF=()
MDL_DIR=()
MDL_NAME=()
MDL_DISK=()
MDL_VRAM=()
MDL_PORT=()
MDL_CAT=()

_add() {
    local i=${#MDL_HF[@]}
    MDL_HF[$i]="$1"; MDL_DIR[$i]="$2"; MDL_NAME[$i]="$3"
    MDL_DISK[$i]="$4"; MDL_VRAM[$i]="$5"; MDL_PORT[$i]="$6"; MDL_CAT[$i]="$7"
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
_add "nvidia/parakeet-tdt-0.6b-v3"                               "parakeet-tdt-0.6b-v3"                  "Parakeet-TDT-0.6B v3 (ASR / NeMo)"       1    0      0  "ASR"
_add "nvidia/nemotron-speech-streaming-en-0.6b"                  "nemotron-speech-streaming-en-0.6b"     "Nemotron-Speech-Streaming-0.6B (ASR)"    1    0      0  "ASR"

# ── 120B+ "SUPER LARGE" models intentionally omitted (removed v0.1.9) ─────────
# A 120B model is ~115–120 GB of weights (BF16 or FP8 alike), which does not fit
# on a 128 GB unified GB10 with any room left for the KV cache → vLLM is
# OOM-killed right after weight load. They need a discrete ≥140 GB GPU or
# multi-GPU tensor-parallel. Re-add them here (and a serve block below) if you
# move to such hardware.

MODEL_TOTAL=${#MDL_HF[@]}

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE CHECKBOX SELECTION
# ─────────────────────────────────────────────────────────────────────────────

_checkbox_menu() {
    # Args: $1=title  $2=servable_only (true|false)  $3=result_var_name
    local title="$1" servable_only="$2" result_var="$3"

    local -a menu_map=()
    for i in $(seq 0 $((MODEL_TOTAL - 1))); do
        [ "$servable_only" = "true" ] && [ "${MDL_PORT[$i]}" = "0" ] && continue
        menu_map+=("$i")
    done
    local count=${#menu_map[@]}

    local -a sel=()
    for j in $(seq 0 $((count - 1))); do sel[$j]=0; done

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

    # On discrete GPUs nvidia-smi reports memory.total. On a GB10 / DGX Spark the GPU
    # shares one unified LPDDR5 pool with the CPU, so memory.total is "[N/A]"/Not
    # Supported — fall back to total system RAM, which is the real budget for
    # weights + KV cache. Without this fallback the old check produced a broken
    # arithmetic value and silently failed to warn before an OOM.
    local total_gb="" mem_source=""
    if command -v nvidia-smi &>/dev/null; then
        local total_mib
        total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
        if [[ "$total_mib" =~ ^[0-9]+$ ]] && [ "$total_mib" -gt 0 ]; then
            total_gb=$(( total_mib / 1024 ))
            mem_source="GPU VRAM"
        fi
    fi
    if [ -z "$total_gb" ]; then
        local total_kb
        total_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null)
        if [[ "$total_kb" =~ ^[0-9]+$ ]]; then
            total_gb=$(( total_kb / 1024 / 1024 ))
            mem_source="unified system RAM (GB10/shared pool)"
        fi
    fi

    if [ -z "$total_gb" ]; then
        echo "  ⚠️  Could not determine memory budget — skipping check"
        echo "  ─────────────────────────────────────────────────────────"
        return 0
    fi

    local safe_gb=$(( total_gb * 90 / 100 ))
    printf "  %-20s : %d GB  (%s)\n" "Memory total" "$total_gb" "$mem_source"
    printf "  %-20s : %d GB  (90%%)\n" "Safe limit" "$safe_gb"
    printf "  %-20s : %d GB\n" "Models require" "$total_required"
    echo ""

    local has_super=0
    for idx in "${RUN_SELECTED[@]}"; do
        [ "${MDL_CAT[$idx]}" = "Super Large" ] && has_super=1 && break
    done

    if [ "$total_required" -gt "$safe_gb" ]; then
        echo "  ⚠️  WARNING: selected models need ~${total_required} GB but the safe limit is ${safe_gb} GB."
        if [ "$has_super" = "1" ]; then
            echo "     A 120B+ (Super Large) model loads ~116 GB of weights. On a 128 GB unified"
            echo "     GB10 that leaves no room for KV cache/activations, so vLLM is OOM-killed"
            echo "     (SIGKILL) right after weight load. Prefer the 30–35B A3B models instead."
        else
            echo "     Consider deselecting some models."
        fi
        echo -n "  Continue anyway? [y/N]: "
        read -r confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    elif [ "$has_super" = "1" ] && [ "${#RUN_SELECTED[@]}" -gt 1 ]; then
        echo "  ⚠️  WARNING: a Super Large model selected alongside others — they will not fit together."
        echo -n "  Continue anyway? [y/N]: "
        read -r confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    else
        echo "  ✅ Memory check OK — ${total_required} GB needed, ${total_gb} GB available"
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
    _checkbox_menu "Available models (toggle with numbers, d=done):" "false" DL_SELECTED
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
_checkbox_menu "Models to serve with vLLM (toggle with numbers, d=done):" "true" RUN_SELECTED

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
    echo "    CMD   : $vllm_label $model_path --host 0.0.0.0 --port $port $*"

    vllm_serve "$model_path" --host 0.0.0.0 --port "$port" "$@" >> "$log_file" 2>&1 &
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

echo "--- Clean start: killing all vLLM processes and removing old logs ---"
docker stop open-webui searxng 2>/dev/null || true
docker rm   open-webui searxng 2>/dev/null || true
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

# ── catalog idx 10 & 11: ASR / NeMo — download only, not served via vLLM ──────
# To use: python3 -c "
#   import nemo.collections.asr as nemo_asr
#   model = nemo_asr.models.EncDecRNNTBPEModel.restore_from('$MODELS_DIR/parakeet-tdt-0.6b-v3/model.nemo')
#   print(model.transcribe(['your_audio.wav']))"

# ─────────────────────────────────────────────────────────────────────────────
# 120B+ "SUPER LARGE" models were removed in v0.1.9 — they do not fit on a 128 GB
# unified GB10 / DGX Spark (a 120B model is ~115–120 GB of weights whether BF16 or
# FP8, leaving no room for KV cache → OOM-killed right after weight load). They
# need a discrete ≥140 GB GPU or multi-GPU tensor-parallel. Re-add the catalog
# entries and serve blocks here if you move to such hardware.
# ─────────────────────────────────────────────────────────────────────────────

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

# Pick the first served model's port as the primary OpenWebUI endpoint
OWUI_PRIMARY_PORT=8005
if [ "${#RUN_SELECTED[@]}" -gt 0 ]; then
    first_run_idx="${RUN_SELECTED[0]}"
    OWUI_PRIMARY_PORT="${MDL_PORT[$first_run_idx]}"
fi

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
    # Fail fast: if the container exited, don't burn the full timeout waiting on a dead box.
    if ! docker ps --format '{{.Names}}' | grep -qx 'open-webui'; then
        echo ""
        echo "❌ open-webui container is not running — it exited during startup."
        echo "   Last 40 lines of logs:"
        docker logs --tail 40 open-webui 2>&1 | sed 's/^/   | /'
        echo "   → Full log: docker logs open-webui"
        break
    fi
    if [ "$OWUI_ELAPSED" -ge "$OWUI_TIMEOUT" ]; then
        echo ""
        echo "⚠️  OpenWebUI still not ready after ${OWUI_TIMEOUT}s (container is up). Recent logs:"
        docker logs --tail 20 open-webui 2>&1 | sed 's/^/   | /'
        echo "   → Keep watching: docker logs -f open-webui"
        break
    fi
    printf "  [%ds] waiting... (container up; first boot downloads the embedding model)\n" "$OWUI_ELAPSED"
    sleep 5
    OWUI_ELAPSED=$((OWUI_ELAPSED + 5))
done

if curl -sf http://localhost:3000/health > /dev/null 2>&1; then
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
