#!/usr/bin/env bash
# Christopher Gray  |  Version: 0.1.1  |  Update: 5/25/2026
# vLLM install, model download, and serve script for DGX Spark / NVIDIA systems
#
# What's New in 0.1.0:
#   - Interactive checkbox model selection: choose which models to download and which to serve
#   - VRAM pre-flight check — calculates total VRAM needed before launching anything
#   - Model catalog with disk size and VRAM estimates for every model
#   - SUPER LARGE section: Nemotron-3-Super-120B-A12B, Qwen3.5-122B-A10B, GPT-OSS-120B
#   - Fixed log-redirection bug in all vllm_serve background calls (missing \ continuation)
#   - Replaced ENABLE_* booleans with runtime interactive selection
#
# Update Yourself:
#   wget --no-cache -O 'install_ai_spark.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_ai_spark.sh' && chmod u+x install_ai_spark.sh
#
# Usage: ./install_ai_spark.sh
#   You will be prompted interactively to select which models to download and serve.

# ─── strict mode ──────────────────────────────────────────────────────────────
# -u: error on unset variables  -o pipefail: propagate pipeline failures
# -e (exit on error) is intentionally omitted — this script uses many
# [ cond ] && action patterns and || true guards that conflict with -e.
set -uo pipefail

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


Version:  0.1.0
Last Updated:  5/25/2026

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
_add "google/gemma-4-26B-A4B-it"                                 "gemma-4-26B-A4B-it"                    "Gemma 4 26B-A4B (BF16)"                 52   56   8007  "General"
_add "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"        "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" "Nemotron-3-Nano-Omni-30B (BF16)"  60   65   8008  "Reasoning"
_add "BAAI/bge-m3"                                               "bge-m3"                                "BGE-M3 (Embeddings)"                     3    4   8011  "Embeddings"
_add "Qwen/Qwen3-Embedding-4B"                                   "Qwen3-Embedding-4B"                    "Qwen3-Embedding-4B (Embeddings)"          8   10   8010  "Embeddings"
_add "BAAI/bge-reranker-v2-m3"                                   "bge-reranker-v2-m3"                    "BGE-Reranker-v2-m3 (Reranking)"           2    3   8020  "Reranking"
_add "nvidia/parakeet-tdt-0.6b-v3"                               "parakeet-tdt-0.6b-v3"                  "Parakeet-TDT-0.6B v3 (ASR / NeMo)"        1    0      0  "ASR"
_add "nvidia/nemotron-speech-streaming-en-0.6b"                  "nemotron-speech-streaming-en-0.6b"     "Nemotron-Speech-Streaming-0.6B (ASR)"     1    0      0  "ASR"

# ── SUPER LARGE models (120B+ parameters) ─────────────────────────────────────
# Info: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard
# Note: these require nearly the entire GPU — do not run alongside other large models.
# ⚠️  Verify HF repo IDs before downloading — these may require updated values.
_add "nvidia/Nemotron-3-Super-120B-A12B"                         "Nemotron-3-Super-120B-A12B"            "Nemotron-3-Super-120B-A12B [SUPER]"     120  115   8030  "Super Large"
_add "Qwen/Qwen3.5-122B-A10B-Instruct"                           "Qwen3.5-122B-A10B-Instruct"            "Qwen3.5-122B-A10B (MoE) [SUPER]"        122  120   8031  "Super Large"
_add "nvidia/GPT-OSS-120B"                                       "GPT-OSS-120B"                          "GPT-OSS-120B [SUPER]"                   120  115   8032  "Super Large"
# ^^^ GPT-OSS-120B: confirm HF repo ID before running — placeholder above

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
# STEP 1 — Select models to download
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 1 of 2 — Select models to DOWNLOAD"
echo "════════════════════════════════════════════════════════════════════"
DL_SELECTED=()
_checkbox_menu "Available models (toggle with numbers, d=done):" "false" DL_SELECTED

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Select models to serve
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 2 of 2 — Select models to SERVE with vLLM"
echo "  (ASR/NeMo models are download-only and excluded from this list)"
echo "════════════════════════════════════════════════════════════════════"
RUN_SELECTED=()
_checkbox_menu "Models to serve with vLLM (toggle with numbers, d=done):" "true" RUN_SELECTED

_check_vram

echo ""
echo "  Download : ${#DL_SELECTED[@]} model(s) selected"
echo "  Serve    : ${#RUN_SELECTED[@]} model(s) selected"
echo ""

#--------------------------
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

#------ Download & install models -----

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

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD selected models
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# SERVE selected models with vLLM
# ─────────────────────────────────────────────────────────────────────────────
VLLM_LOGS="$BASE_DIR/logs"
mkdir -p "$VLLM_LOGS"

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

vllm_serve() {
    if [ -n "$VLLM_BIN" ]; then
        "$VLLM_BIN" serve "$@"
    else
        echo "⚠️  vllm not found — trying python module fallback"
        "$VENV_DIR/bin/python" -m vllm.entrypoints.openai.api_server "$@"
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
fi

# ── idx 0: Qwen3.6-35B-A3B-FP8  (port 8005) ──────────────────────────────────
if is_run_selected 0; then
    if [ -f "$MODELS_DIR/Qwen3.6-35B-A3B-FP8/config.json" ]; then
        echo "--- Starting vLLM: Qwen3.6-35B-A3B-FP8 on port 8005 ---"
        vllm_serve "$MODELS_DIR/Qwen3.6-35B-A3B-FP8" \
            --host 0.0.0.0 --port 8005 \
            --served-model-name "Qwen3.6-35B-A3B" \
            --dtype auto \
            --gpu-memory-utilization 0.73 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            --chat-template-kwargs '{"enable_thinking": false}' \
            >> "$VLLM_LOGS/vllm-8005.log" 2>&1 &
        echo "✅ Qwen3.6-35B-A3B-FP8 starting on port 8005 (pid $!)"
        echo "   → Logs  : tail -f $VLLM_LOGS/vllm-8005.log"
        echo "   → Status: curl -s http://localhost:8005/v1/models | jq ."
    else
        echo "⚠️  Qwen3.6-35B-A3B-FP8 not found in $MODELS_DIR — was it downloaded?"
    fi
fi

# ── idx 1: NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4  (port 8006) ─────────────────
if is_run_selected 1; then
    if [ -f "$MODELS_DIR/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/config.json" ]; then
        echo "--- Starting vLLM: NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 on port 8006 ---"
        vllm_serve "$MODELS_DIR/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" \
            --host 0.0.0.0 --port 8006 \
            --served-model-name "Nemotron-3-Nano-30B-NVFP4" \
            --dtype auto \
            --quantization modelopt_fp4 \
            --gpu-memory-utilization 0.85 \
            --max-model-len 32768 \
            --max-num-seqs 178 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            >> "$VLLM_LOGS/vllm-8006.log" 2>&1 &
        echo "✅ Nemotron-3-Nano-30B-NVFP4 starting on port 8006 (pid $!)"
        echo "   → Logs  : tail -f $VLLM_LOGS/vllm-8006.log"
        echo "   → Status: curl -s http://localhost:8006/v1/models | jq ."
        echo "   → Add to OpenWebUI: Admin Settings → Connections → http://localhost:8006/v1"
    else
        echo "⚠️  NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 not found in $MODELS_DIR — was it downloaded?"
    fi
fi

# ── idx 2: Qwen3-Coder-30B-A3B-Instruct  (port 8001) ─────────────────────────
if is_run_selected 2; then
    if [ -f "$MODELS_DIR/Qwen3-Coder-30B-A3B-Instruct/config.json" ]; then
        echo "--- Starting vLLM: Qwen3-Coder-30B-A3B-Instruct on port 8001 ---"
        vllm_serve "$MODELS_DIR/Qwen3-Coder-30B-A3B-Instruct" \
            --host 0.0.0.0 --port 8001 \
            --served-model-name "Qwen3-Coder-30B" \
            --dtype auto \
            --gpu-memory-utilization 0.85 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            >> "$VLLM_LOGS/vllm-8001.log" 2>&1 &
        echo "✅ Qwen3-Coder-30B-A3B starting on port 8001 (pid $!)"
        echo "   → Logs  : tail -f $VLLM_LOGS/vllm-8001.log"
        echo "   → Status: curl -s http://localhost:8001/v1/models | jq ."
    else
        echo "⚠️  Qwen3-Coder-30B-A3B-Instruct not found in $MODELS_DIR — was it downloaded?"
    fi
fi

# ── idx 3: DeepSeek-R1-Distill-Qwen-32B  (port 8002) ─────────────────────────
if is_run_selected 3; then
    if [ -f "$MODELS_DIR/DeepSeek-R1-Distill-Qwen-32B/config.json" ]; then
        echo "--- Starting vLLM: DeepSeek-R1-Distill-Qwen-32B on port 8002 ---"
        vllm_serve "$MODELS_DIR/DeepSeek-R1-Distill-Qwen-32B" \
            --host 0.0.0.0 --port 8002 \
            --served-model-name "DeepSeek-R1-Distill-Qwen-32B" \
            --dtype auto \
            --gpu-memory-utilization 0.85 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            >> "$VLLM_LOGS/vllm-8002.log" 2>&1 &
        echo "✅ DeepSeek-R1-Distill-Qwen-32B starting on port 8002 (pid $!)"
        echo "   → Logs  : tail -f $VLLM_LOGS/vllm-8002.log"
        echo "   → Status: curl -s http://localhost:8002/v1/models | jq ."
    else
        echo "⚠️  DeepSeek-R1-Distill-Qwen-32B not found in $MODELS_DIR — was it downloaded?"
    fi
fi

# ── idx 4: gemma-4-26B-A4B-it  (port 8007) ───────────────────────────────────
if is_run_selected 4; then
    if [ -f "$MODELS_DIR/gemma-4-26B-A4B-it/config.json" ]; then
        echo "--- Starting vLLM: gemma-4-26B-A4B-it on port 8007 ---"
        vllm_serve "$MODELS_DIR/gemma-4-26B-A4B-it" \
            --host 0.0.0.0 --port 8007 \
            --served-model-name "gemma-4-26B-A4B" \
            --dtype auto \
            --gpu-memory-utilization 0.55 \
            --max-model-len 16384 \
            --max-num-batched-tokens 4096 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            >> "$VLLM_LOGS/vllm-8007.log" 2>&1 &
        echo "✅ gemma-4-26B-A4B starting on port 8007 (pid $!)"
        echo "   → Logs  : tail -f $VLLM_LOGS/vllm-8007.log"
        echo "   → Status: curl -s http://localhost:8007/v1/models | jq ."
    else
        echo "⚠️  gemma-4-26B-A4B-it not found in $MODELS_DIR — was it downloaded?"
    fi
fi

# ── idx 5: Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16  (port 8008) ──────────
if is_run_selected 5; then
    if [ -f "$MODELS_DIR/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/config.json" ]; then
        echo "--- Starting vLLM: Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 on port 8008 ---"
        vllm_serve "$MODELS_DIR/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" \
            --host 0.0.0.0 --port 8008 \
            --served-model-name "Nemotron-3-Nano-Omni-30B-A3B" \
            --dtype bfloat16 \
            --gpu-memory-utilization 0.85 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            >> "$VLLM_LOGS/vllm-8008.log" 2>&1 &
        echo "✅ Nemotron-3-Nano-Omni-30B-A3B starting on port 8008 (pid $!)"
        echo "   → Logs  : tail -f $VLLM_LOGS/vllm-8008.log"
        echo "   → Status: curl -s http://localhost:8008/v1/models | jq ."
    else
        echo "⚠️  Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 not found in $MODELS_DIR — was it downloaded?"
    fi
fi

# ── idx 6: BAAI/bge-m3  (port 8011) ──────────────────────────────────────────
if is_run_selected 6; then
    if [ -f "$MODELS_DIR/bge-m3/config.json" ]; then
        echo "--- Starting vLLM: bge-m3 on port 8011 ---"
        vllm_serve "$MODELS_DIR/bge-m3" \
            --host 0.0.0.0 --port 8011 \
            --served-model-name "bge-m3" \
            --task embedding \
            --dtype auto \
            --gpu-memory-utilization 0.30 \
            --trust-remote-code \
            >> "$VLLM_LOGS/vllm-8011.log" 2>&1 &
        echo "✅ bge-m3 starting on port 8011 (pid $!)"
        echo "   → Logs  : tail -f $VLLM_LOGS/vllm-8011.log"
    else
        echo "⚠️  bge-m3 not found in $MODELS_DIR — was it downloaded?"
    fi
fi

# ── idx 7: Qwen3-Embedding-4B  (port 8010) ────────────────────────────────────
if is_run_selected 7; then
    if [ -f "$MODELS_DIR/Qwen3-Embedding-4B/config.json" ]; then
        echo "--- Starting vLLM: Qwen3-Embedding-4B on port 8010 ---"
        vllm_serve "$MODELS_DIR/Qwen3-Embedding-4B" \
            --host 0.0.0.0 --port 8010 \
            --served-model-name "Qwen3-Embedding-4B" \
            --task embedding \
            --dtype auto \
            --gpu-memory-utilization 0.50 \
            --trust-remote-code \
            >> "$VLLM_LOGS/vllm-8010.log" 2>&1 &
        echo "✅ Qwen3-Embedding-4B starting on port 8010 (pid $!)"
        echo "   → Logs  : tail -f $VLLM_LOGS/vllm-8010.log"
        echo "   → Status: curl -s http://localhost:8010/v1/models | jq ."
    else
        echo "⚠️  Qwen3-Embedding-4B not found in $MODELS_DIR — was it downloaded?"
    fi
fi

# ── idx 8: bge-reranker-v2-m3  (port 8020) ────────────────────────────────────
if is_run_selected 8; then
    if [ -f "$MODELS_DIR/bge-reranker-v2-m3/config.json" ]; then
        echo "--- Starting vLLM: bge-reranker-v2-m3 on port 8020 ---"
        vllm_serve "$MODELS_DIR/bge-reranker-v2-m3" \
            --host 0.0.0.0 --port 8020 \
            --served-model-name "bge-reranker-v2-m3" \
            --task classify \
            --dtype auto \
            --gpu-memory-utilization 0.50 \
            --trust-remote-code \
            >> "$VLLM_LOGS/vllm-8020.log" 2>&1 &
        echo "✅ bge-reranker-v2-m3 starting on port 8020 (pid $!)"
        echo "   → Logs  : tail -f $VLLM_LOGS/vllm-8020.log"
        echo "   → Status: curl -s http://localhost:8020/v1/models | jq ."
    else
        echo "⚠️  bge-reranker-v2-m3 not found in $MODELS_DIR — was it downloaded?"
    fi
fi

# ── idx 9 & 10: ASR / NeMo models — download only, not served via vLLM ────────
# To use: python3 -c "
#   import nemo.collections.asr as nemo_asr
#   model = nemo_asr.models.EncDecRNNTBPEModel.restore_from('$MODELS_DIR/parakeet-tdt-0.6b-v3/model.nemo')
#   print(model.transcribe(['your_audio.wav']))"

# ─────────────────────────────────────────────────────────────────────────────
# SUPER LARGE MODELS (120B+ parameters)
# Info: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard
# ⚠️  These require nearly the entire GPU. Do NOT run alongside other large models.
# ⚠️  Verify HF repo IDs before downloading — see comments above _add entries.
# ─────────────────────────────────────────────────────────────────────────────

# ── idx 11: Nemotron-3-Super-120B-A12B  (port 8030) ───────────────────────────
if is_run_selected 11; then
    if [ -f "$MODELS_DIR/Nemotron-3-Super-120B-A12B/config.json" ]; then
        echo "--- Starting vLLM: Nemotron-3-Super-120B-A12B on port 8030 ---"
        echo "   ℹ️  Model info: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard"
        echo "   ⚠️  SUPER LARGE — needs ~115 GB VRAM. Ensure no other large models are running."
        vllm_serve "$MODELS_DIR/Nemotron-3-Super-120B-A12B" \
            --host 0.0.0.0 --port 8030 \
            --served-model-name "Nemotron-3-Super-120B-A12B" \
            --dtype auto \
            --gpu-memory-utilization 0.93 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            >> "$VLLM_LOGS/vllm-8030.log" 2>&1 &
        echo "✅ Nemotron-3-Super-120B-A12B starting on port 8030 (pid $!)"
        echo "   → Logs  : tail -f $VLLM_LOGS/vllm-8030.log"
        echo "   → Status: curl -s http://localhost:8030/v1/models | jq ."
    else
        echo "⚠️  Nemotron-3-Super-120B-A12B not found in $MODELS_DIR — was it downloaded?"
    fi
fi

# ── idx 12: Qwen3.5-122B-A10B-Instruct  (port 8031) ──────────────────────────
if is_run_selected 12; then
    if [ -f "$MODELS_DIR/Qwen3.5-122B-A10B-Instruct/config.json" ]; then
        echo "--- Starting vLLM: Qwen3.5-122B-A10B-Instruct on port 8031 ---"
        echo "   ⚠️  SUPER LARGE — needs ~120 GB VRAM. Ensure no other large models are running."
        vllm_serve "$MODELS_DIR/Qwen3.5-122B-A10B-Instruct" \
            --host 0.0.0.0 --port 8031 \
            --served-model-name "Qwen3.5-122B-A10B" \
            --dtype auto \
            --gpu-memory-utilization 0.93 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            >> "$VLLM_LOGS/vllm-8031.log" 2>&1 &
        echo "✅ Qwen3.5-122B-A10B starting on port 8031 (pid $!)"
        echo "   → Logs  : tail -f $VLLM_LOGS/vllm-8031.log"
        echo "   → Status: curl -s http://localhost:8031/v1/models | jq ."
    else
        echo "⚠️  Qwen3.5-122B-A10B-Instruct not found in $MODELS_DIR — was it downloaded?"
    fi
fi

# ── idx 13: GPT-OSS-120B  (port 8032) ─────────────────────────────────────────
if is_run_selected 13; then
    if [ -f "$MODELS_DIR/GPT-OSS-120B/config.json" ]; then
        echo "--- Starting vLLM: GPT-OSS-120B on port 8032 ---"
        echo "   ⚠️  SUPER LARGE — needs ~115 GB VRAM. Ensure no other large models are running."
        vllm_serve "$MODELS_DIR/GPT-OSS-120B" \
            --host 0.0.0.0 --port 8032 \
            --served-model-name "GPT-OSS-120B" \
            --dtype auto \
            --gpu-memory-utilization 0.93 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            >> "$VLLM_LOGS/vllm-8032.log" 2>&1 &
        echo "✅ GPT-OSS-120B starting on port 8032 (pid $!)"
        echo "   → Logs  : tail -f $VLLM_LOGS/vllm-8032.log"
        echo "   → Status: curl -s http://localhost:8032/v1/models | jq ."
    else
        echo "⚠️  GPT-OSS-120B not found in $MODELS_DIR — was it downloaded?"
    fi
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
