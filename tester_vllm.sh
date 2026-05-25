#!/usr/bin/env bash
# Christopher Gray  |  Version: 0.0.17  |  Update: 5/25/2026
# vLLM smoke test — checks health, discovers models, and sends sample prompts
#
#
#Update Yourself:
#  wget --no-cache -O 'tester_vllm.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/tester_vllm.sh' && chmod u+x tester_vllm.sh
#
#
# Usage: ./tester_vllm.sh [HOST] [PORT]
#   HOST defaults to localhost, PORT defaults to 8000
set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-8000}"
PORT_EXPLICIT="${2+yes}"   # non-empty only if user passed a port argument
BASE_URL="http://${HOST}:${PORT}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

pass() { echo -e "${GREEN}[PASS]${RESET} $*"; }
fail() { echo -e "${RED}[FAIL]${RESET} $*"; }
info() { echo -e "${CYAN}[INFO]${RESET} $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
header() { echo -e "\n${BOLD}${CYAN}==> $*${RESET}"; }

FAILURES=0

require_cmd() {
    command -v "$1" &>/dev/null || { fail "Required command not found: $1"; exit 1; }
}

require_cmd curl
require_cmd jq

# ── Helper: HTTP GET with timeout ─────────────────────────────────────────────
http_get() {
    local url="$1"
    curl -sf --max-time 10 "$url" 2>/dev/null
}

# ── Helper: POST JSON ─────────────────────────────────────────────────────────
http_post() {
    local url="$1"
    local body="$2"
    curl -sf --max-time 60 \
        -H "Content-Type: application/json" \
        -d "$body" \
        "$url" 2>/dev/null
}

# ─────────────────────────────────────────────────────────────────────────────
header "Target: ${BASE_URL}"

# ── System Hardware ────────────────────────────────────────────────────────────
header "System Hardware"
OS_TYPE=$(uname -s)
ARCH=$(uname -m)
info "Platform : ${OS_TYPE} / ${ARCH}"

if [ "$OS_TYPE" = "Linux" ]; then
    CPU_NAME=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || \
               lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2 | xargs || echo "unknown")
    CPU_CORES=$(nproc 2>/dev/null || echo "?")
    RAM_INFO=$(free -h 2>/dev/null | awk '/^Mem:/{printf "total=%s  used=%s  free=%s", $2, $3, $4}' || echo "unknown")
    info "CPU      : ${CPU_NAME}  (${CPU_CORES} cores)"
    info "RAM      : ${RAM_INFO}"
elif [ "$OS_TYPE" = "Darwin" ]; then
    CPU_NAME=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
    TOTAL_RAM=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.1f GB", $1/1073741824}' || echo "unknown")
    info "CPU      : ${CPU_NAME}"
    info "RAM      : ${TOTAL_RAM}"
fi

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    DRIVER_VER=$(nvidia-smi 2>/dev/null | grep -oP 'Driver Version: \K[\d.]+' || echo "n/a")
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' || echo "n/a")
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 | xargs || echo "?")
    info "GPU      : NVIDIA  (driver=${DRIVER_VER}  CUDA=${CUDA_VER}  count=${GPU_COUNT})"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=',' read -r idx name mtot mused mfree util temp; do
            name=$(echo "$name" | xargs); mtot=$(echo "$mtot" | xargs)
            mused=$(echo "$mused" | xargs); mfree=$(echo "$mfree" | xargs)
            util=$(echo "$util" | xargs); temp=$(echo "$temp" | xargs)
            echo "  [GPU ${idx}] ${name}"
            echo "           VRAM : ${mtot} MiB total  |  ${mused} MiB used  |  ${mfree} MiB free"
            echo "           Util : ${util}%  |  Temp: ${temp}°C"
        done
elif command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; then
    ROCM_VER=$(rocminfo 2>/dev/null | grep -oP 'ROCm Version: \K[\d.]+' || echo "n/a")
    info "GPU      : AMD ROCm  (version=${ROCM_VER})"
    rocm-smi --showmeminfo vram --showuse --showtemp 2>/dev/null | grep -v '^$' | sed 's/^/  /' || \
        rocm-smi 2>/dev/null | grep -v '^$' | sed 's/^/  /' || true
elif [ "$OS_TYPE" = "Darwin" ]; then
    GPU_INFO=$(system_profiler SPDisplaysDataType 2>/dev/null | \
        grep -E 'Chipset Model|Total Number of Cores|VRAM|Metal' | \
        sed 's/^ *//' | paste -sd'  ' - || echo "unknown")
    info "GPU      : Apple Silicon / Metal  —  ${GPU_INFO}"
else
    warn "GPU      : None detected — CPU-only inference"
fi

# ── 0. vLLM Instance Discovery ────────────────────────────────────────────────
header "0. vLLM Instance Discovery"
VLLM_PROCS=$(ps aux 2>/dev/null | grep -E 'vllm serve|vllm\.entrypoints\.openai' | grep -v grep || true)

if [ -z "$VLLM_PROCS" ]; then
    warn "No running vLLM processes found on this host"
else
    PROC_COUNT=$(echo "$VLLM_PROCS" | wc -l | tr -d ' ')
    info "Found ${PROC_COUNT} vLLM instance(s) running:"
    DISCOVERED_PORTS=()
    while IFS= read -r line; do
        V_PID=$(echo "$line" | awk '{print $2}')
        V_PORT=$(echo "$line" | grep -oP '(?<=--port )\d+' || true)
        V_HOST_BIND=$(echo "$line" | grep -oP '(?<=--host )\S+' || true)
        V_MODEL_NAME=$(echo "$line" | grep -oP '(?<=--served-model-name )\S+' || true)
        if [ -z "$V_MODEL_NAME" ]; then
            V_MODEL_NAME=$(echo "$line" | grep -oP '(?<=vllm serve )\S+' || true)
        fi
        [ -z "$V_PORT" ]       && V_PORT="unknown"
        [ -z "$V_HOST_BIND" ]  && V_HOST_BIND="0.0.0.0"
        [ -z "$V_MODEL_NAME" ] && V_MODEL_NAME="unknown"
        echo "  PID ${V_PID}  |  port=${V_PORT}  host=${V_HOST_BIND}  model=${V_MODEL_NAME}"
        [ "$V_PORT" != "unknown" ] && DISCOVERED_PORTS+=("$V_PORT")
    done <<< "$VLLM_PROCS"

    # Auto-correct port if user didn't specify one and vLLM isn't on the default
    if [ -z "${PORT_EXPLICIT}" ] && [ "${#DISCOVERED_PORTS[@]}" -eq 1 ] && [ "${DISCOVERED_PORTS[0]}" != "$PORT" ]; then
        warn "Auto-switching to discovered port ${DISCOVERED_PORTS[0]} (was: ${PORT})"
        PORT="${DISCOVERED_PORTS[0]}"
        BASE_URL="http://${HOST}:${PORT}"
        info "Updated target: ${BASE_URL}"
    elif [ -z "${PORT_EXPLICIT}" ] && [ "${#DISCOVERED_PORTS[@]}" -gt 1 ]; then
        warn "Multiple instances found — using first: port ${DISCOVERED_PORTS[0]}"
        warn "Pass a port explicitly to target a specific instance: $0 ${HOST} <port>"
        PORT="${DISCOVERED_PORTS[0]}"
        BASE_URL="http://${HOST}:${PORT}"
        info "Updated target: ${BASE_URL}"
    fi
fi

# ── 1. Reachability ───────────────────────────────────────────────────────────
header "1. Reachability"
if curl -sf --max-time 5 "${BASE_URL}" &>/dev/null || \
   curl -sf --max-time 5 "${BASE_URL}/health" &>/dev/null || \
   curl -sf --max-time 5 "${BASE_URL}/v1/models" &>/dev/null; then
    pass "Host is reachable at ${BASE_URL}"
else
    fail "Cannot reach ${BASE_URL}"
    echo "  Hint: check that vLLM is running and the host/port are correct."
    exit 1
fi

# ── 2. Health endpoint ────────────────────────────────────────────────────────
header "2. Health check  (GET /health)"
HEALTH=$(http_get "${BASE_URL}/health" || true)
if [ -n "$HEALTH" ]; then
    pass "Health endpoint responded: ${HEALTH}"
else
    warn "/health returned empty or no response (may be unsupported on this version)"
fi

# ── 3. Model list ─────────────────────────────────────────────────────────────
header "3. Model list  (GET /v1/models)"
MODELS_JSON=$(http_get "${BASE_URL}/v1/models" || true)
if [ -z "$MODELS_JSON" ]; then
    fail "No response from /v1/models"
    ((FAILURES++))
else
    MODEL_COUNT=$(echo "$MODELS_JSON" | jq '.data | length' 2>/dev/null || echo 0)
    if [ "$MODEL_COUNT" -eq 0 ]; then
        warn "Model list is empty"
        ((FAILURES++))
    else
        pass "Found ${MODEL_COUNT} model(s):"
        echo "$MODELS_JSON" | jq -r '.data[] | "  • \(.id)  (owned_by: \(.owned_by // "n/a"))"'
        # Pick the first model for subsequent tests
        FIRST_MODEL=$(echo "$MODELS_JSON" | jq -r '.data[0].id')
        info "Using model for tests: ${FIRST_MODEL}"
    fi
fi

# ── 4. Server info / version ──────────────────────────────────────────────────
header "4. Server info"
for path in "/version" "/v1/version" "/info"; do
    RESP=$(http_get "${BASE_URL}${path}" || true)
    if [ -n "$RESP" ]; then
        pass "${path}: ${RESP}"
    fi
done

# ── 5. OpenAI-compatible chat completion ──────────────────────────────────────
header "5. Chat completion  (POST /v1/chat/completions)"
if [ -z "${FIRST_MODEL:-}" ]; then
    warn "Skipping — no model discovered"
else
    CHAT_BODY=$(jq -n \
        --arg model "$FIRST_MODEL" \
        '{
            model: $model,
            max_tokens: 120,
            temperature: 0.1,
            messages: [
                {role: "system", content: "You are a helpful assistant. Be concise."},
                {role: "user",   content: "What model are you and what are your key capabilities? Answer in 2-3 sentences."}
            ]
        }')
    CHAT_RESP=$(http_post "${BASE_URL}/v1/chat/completions" "$CHAT_BODY" || true)
    if [ -z "$CHAT_RESP" ]; then
        fail "No response from /v1/chat/completions"
        ((FAILURES++))
    else
        CHAT_TEXT=$(echo "$CHAT_RESP" | jq -r '.choices[0].message.content' 2>/dev/null || true)
        USAGE=$(echo "$CHAT_RESP"     | jq -r '"prompt=\(.usage.prompt_tokens) completion=\(.usage.completion_tokens) total=\(.usage.total_tokens)"' 2>/dev/null || true)
        if [ -n "$CHAT_TEXT" ] && [ "$CHAT_TEXT" != "null" ]; then
            pass "Chat completion succeeded"
            echo "  Response : ${CHAT_TEXT}"
            echo "  Tokens   : ${USAGE}"
        else
            fail "Chat response malformed"
            echo "  Raw: ${CHAT_RESP}" | head -c 400
            ((FAILURES++))
        fi
    fi
fi

# ── 6. OpenAI-compatible text completion ──────────────────────────────────────
header "6. Text completion  (POST /v1/completions)"
if [ -z "${FIRST_MODEL:-}" ]; then
    warn "Skipping — no model discovered"
else
    COMP_BODY=$(jq -n \
        --arg model "$FIRST_MODEL" \
        '{
            model: $model,
            prompt: "The capital of France is",
            max_tokens: 20,
            temperature: 0
        }')
    COMP_RESP=$(http_post "${BASE_URL}/v1/completions" "$COMP_BODY" || true)
    if [ -z "$COMP_RESP" ]; then
        warn "/v1/completions not supported or returned no response (expected for chat-only models)"
    else
        COMP_TEXT=$(echo "$COMP_RESP" | jq -r '.choices[0].text' 2>/dev/null || true)
        if [ -n "$COMP_TEXT" ] && [ "$COMP_TEXT" != "null" ]; then
            pass "Text completion succeeded: \"The capital of France is${COMP_TEXT}\""
        else
            warn "Text completion response malformed (may be unsupported)"
        fi
    fi
fi

# ── 7. Embeddings endpoint ────────────────────────────────────────────────────
header "7. Embeddings  (POST /v1/embeddings)"
if [ -z "${FIRST_MODEL:-}" ]; then
    warn "Skipping — no model discovered"
else
    EMB_BODY=$(jq -n \
        --arg model "$FIRST_MODEL" \
        '{model: $model, input: "Hello, world!"}')
    EMB_RESP=$(http_post "${BASE_URL}/v1/embeddings" "$EMB_BODY" || true)
    if [ -z "$EMB_RESP" ]; then
        warn "/v1/embeddings not supported (expected — embeddings require a dedicated embedding model)"
    else
        EMB_LEN=$(echo "$EMB_RESP" | jq '.data[0].embedding | length' 2>/dev/null || echo 0)
        if [ "$EMB_LEN" -gt 0 ]; then
            pass "Embeddings returned vector of length ${EMB_LEN}"
        else
            warn "Embeddings endpoint responded but no vector returned"
        fi
    fi
fi

# ── 8. Sampling parameters / model introspection via chat ─────────────────────
header "8. Model self-description prompts"
if [ -z "${FIRST_MODEL:-}" ]; then
    warn "Skipping — no model discovered"
else
    PROMPTS=(
        "What is your context window length in tokens?"
        "List any special capabilities you have, such as vision, code, tool use, or multilingual support."
        "What languages can you respond in?"
    )
    for PROMPT in "${PROMPTS[@]}"; do
        BODY=$(jq -n \
            --arg model "$FIRST_MODEL" \
            --arg prompt "$PROMPT" \
            '{
                model: $model,
                max_tokens: 100,
                temperature: 0.1,
                messages: [{role: "user", content: $prompt}]
            }')
        RESP=$(http_post "${BASE_URL}/v1/chat/completions" "$BODY" || true)
        TEXT=$(echo "$RESP" | jq -r '.choices[0].message.content' 2>/dev/null || true)
        if [ -n "$TEXT" ] && [ "$TEXT" != "null" ]; then
            echo -e "  ${BOLD}Q:${RESET} ${PROMPT}"
            echo    "  A: ${TEXT}"
            echo
        else
            warn "No response for: ${PROMPT}"
        fi
    done
fi

# ── 9. Streaming check ────────────────────────────────────────────────────────
header "9. Streaming  (POST /v1/chat/completions  stream=true)"
if [ -z "${FIRST_MODEL:-}" ]; then
    warn "Skipping — no model discovered"
else
    STREAM_BODY=$(jq -n \
        --arg model "$FIRST_MODEL" \
        '{
            model: $model,
            max_tokens: 30,
            temperature: 0,
            stream: true,
            messages: [{role: "user", content: "Say hello in one sentence."}]
        }')
    STREAM_OUT=$(curl -sf --max-time 20 \
        -H "Content-Type: application/json" \
        -d "$STREAM_BODY" \
        "${BASE_URL}/v1/chat/completions" 2>/dev/null | head -5 || true)
    if echo "$STREAM_OUT" | grep -q "data:"; then
        pass "Streaming response received (first chunks):"
        echo "$STREAM_OUT" | head -3 | sed 's/^/  /'
    else
        warn "Streaming check inconclusive (may still work — check manually)"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
header "Summary"
if [ "$FAILURES" -eq 0 ]; then
    pass "All critical checks passed for ${BASE_URL}"
else
    fail "${FAILURES} critical check(s) failed"
fi

exit "$FAILURES"
