#!/usr/bin/env bash
# Christopher Gray  |  Version: 0.0.21  |  Update: 7/14/2026
# vLLM smoke test — auto-discovers every running vLLM instance (ports + models)
#                   and runs the full smoke test against each one.
# Includes 11 auto-graded model-quality tests (reasoning, math, summarization,
# instruction-following, code, factual, long-context, translation, sentiment,
# vision/OCR, audio/ASR) with a per-instance capability scorecard.
#
#
#Update Yourself:
#  wget --no-cache -O 'tester_vllm.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/tester_vllm.sh' && chmod u+x tester_vllm.sh
#
#
# Usage: ./tester_vllm.sh [HOST] [PORT]
#   No args     -> auto-discover ALL local vLLM instances and test each.
#   HOST        -> discover instances on that host (local discovery only).
#   HOST PORT   -> test only that specific host:port (skips discovery).
set -euo pipefail

HOST="${1:-localhost}"
PORT_ARG="${2:-}"            # non-empty only if user passed a port argument

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

# ── Helper: millisecond wall-clock timestamp (GNU date, with portable fallback)
now_ms() {
    local ts
    ts=$(date +%s%3N 2>/dev/null || true)
    if [[ "$ts" =~ ^[0-9]+$ ]]; then
        echo "$ts"
    elif command -v perl &>/dev/null; then
        perl -MTime::HiRes=time -e 'printf "%.0f\n", time() * 1000'
    else
        echo $(( $(date +%s) * 1000 ))
    fi
}

tokens_per_second() {
    local tokens="$1" elapsed_ms="$2"
    awk -v tok="$tokens" -v ms="$elapsed_ms" \
        'BEGIN { if (tok > 0 && ms > 0) printf "%.2f", tok * 1000 / ms; else printf "0.00" }'
}

# ── Helper: decode base64 from stdin -> binary on stdout (portable) ───────────
b64decode() {
    if command -v openssl &>/dev/null; then openssl base64 -d -A
    elif base64 --help 2>&1 | grep -q -- '--decode'; then base64 --decode
    else base64 -D; fi
}

# ── Helper: single-turn, non-streaming chat; echoes assistant text (deterministic)
#    Uses the caller's FIRST_MODEL and BASE_URL (bash dynamic scope).
chat_once() {
    local prompt="$1" maxtok="${2:-256}" body
    body=$(jq -n --arg m "$FIRST_MODEL" --arg p "$prompt" --argjson mt "$maxtok" \
        '{model:$m, max_tokens:$mt, temperature:0,
          messages:[{role:"user", content:$p}]}')
    http_post "${BASE_URL}/v1/chat/completions" "$body" 2>/dev/null \
        | jq -r '.choices[0].message.content // empty' 2>/dev/null
}

# ── Helper: benchmark generated completion tokens/sec for the active model ────
#    Uses the caller's FIRST_MODEL and BASE_URL (bash dynamic scope).
benchmark_chat_tps() {
    local runs="${1:-2}" maxtok="${2:-192}"
    local i body resp start_ms end_ms elapsed_ms prompt_tokens completion_tokens total_tokens tps
    local ok=0 total_completion=0 total_elapsed=0

    for i in $(seq 1 "$runs"); do
        body=$(jq -n \
            --arg m "$FIRST_MODEL" \
            --argjson mt "$maxtok" \
            '{
                model: $m,
                max_tokens: $mt,
                temperature: 0,
                messages: [
                    {role: "user", content: "Write a dense technical paragraph about local AI inference performance, batching, KV cache behavior, and latency. Continue until you naturally reach the token budget."}
                ]
            }')

        start_ms=$(now_ms)
        resp=$(http_post "${BASE_URL}/v1/chat/completions" "$body" || true)
        end_ms=$(now_ms)
        elapsed_ms=$((end_ms - start_ms))
        [ "$elapsed_ms" -le 0 ] && elapsed_ms=1

        if [ -z "$resp" ]; then
            warn "Throughput run ${i}/${runs}: no response"
            continue
        fi

        prompt_tokens=$(printf '%s' "$resp" | jq -r '.usage.prompt_tokens // empty' 2>/dev/null || true)
        completion_tokens=$(printf '%s' "$resp" | jq -r '.usage.completion_tokens // empty' 2>/dev/null || true)
        total_tokens=$(printf '%s' "$resp" | jq -r '.usage.total_tokens // empty' 2>/dev/null || true)

        if ! [[ "$completion_tokens" =~ ^[0-9]+$ ]] || [ "$completion_tokens" -eq 0 ]; then
            warn "Throughput run ${i}/${runs}: response did not include usage.completion_tokens"
            continue
        fi

        tps=$(tokens_per_second "$completion_tokens" "$elapsed_ms")
        printf "  Run %d/%d : %4d completion tokens in %.2fs  =>  %s tok/s" \
            "$i" "$runs" "$completion_tokens" "$(awk -v ms="$elapsed_ms" 'BEGIN { printf "%.2f", ms / 1000 }')" "$tps"
        [ -n "$prompt_tokens" ] && printf "  (prompt=%s total=%s)" "$prompt_tokens" "${total_tokens:-?}"
        printf "\n"

        ok=$((ok + 1))
        total_completion=$((total_completion + completion_tokens))
        total_elapsed=$((total_elapsed + elapsed_ms))
    done

    if [ "$ok" -gt 0 ]; then
        tps=$(tokens_per_second "$total_completion" "$total_elapsed")
        pass "Average generation throughput: ${tps} completion tokens/sec (${total_completion} tokens across ${ok} run(s))"
    else
        warn "Could not calculate throughput for ${FIRST_MODEL}"
    fi
}

# ── Helper: grade a response against a case-insensitive regex ─────────────────
#    grade LABEL RESPONSE PATTERN   — updates QUALITY_PASS / QUALITY_TOTAL.
grade() {
    local label="$1" resp="$2" pat="$3"
    QUALITY_TOTAL=$((QUALITY_TOTAL + 1))
    if [ -n "$resp" ] && printf '%s' "$resp" | grep -qiE "$pat"; then
        pass "${label}"
        QUALITY_PASS=$((QUALITY_PASS + 1))
        return 0
    fi
    warn "${label} — expected /${pat}/"
    [ -n "$resp" ] && echo "     got: $(printf '%s' "$resp" | tr '\n' ' ' | head -c 160)"
    return 1
}

# ── Embedded test media (self-contained; no external files or network) ────────
# 1-bit PNG containing the literal text "VLLM-OCR-7392"  (for the vision/OCR test)
OCR_PNG_B64="iVBORw0KGgoAAAANSUhEUgAAAWgAAABaAQAAAAC9W/FqAAACU0lEQVR42u3WQW7bRhjF8d+MBi135tILAyFyjqBiexIfoSeIJ0XXPYNOEtBFDpBeoKABL7wrUxQF0445XcixLEVJkHVFgAAJvvn48b3/N2CovuKITuqT+qQ+qf/v6qQ2iZusBjctblruk3ch4DokPxF5RfTSclBh8ye39xk9qFf5vuaa+20nzZ645pptbDCCxaYYFm8P+s4jFgbZhAkU02wsiNZKu7ekMFZm25OZyTSbj3kyfztOy2pmXUByMXvBxVEHp/PzuaRvCrGwLqbn/DB7XkRdnrsPwuFX6BLnAcHWg96YTd8f1N56M/axzB59Le3F48JjWeZA55cPn9GtMrSI2mHqj6X84+5yaShn+aB2Gn/GwPKkwNQrn6EqkB9Stc3zroWaRM045k8g90+CbOxcsXye2OsGrtF3rz/uJP6xz9cjbhtvKI0oTdefrN4+1JtazO0XZucSmRdzswUy7ruxT2PIjwMWGDtR3F9xvbtbVzehJkJZRYZeZN4Z+PeDyxkbcn4Kfc7H+u6J7eCWfng6gIEoSDueut3Tl3QjS+MdpCUS+fcjBiujSLudy7e8z0o6zLKwXrphSY2KxrMKV+4qDZF8EKCxcLcgbae4kwzmluTp5tMvv9He3TWp/JV2eb7W/t64SWXbyX2IXoUHppv3XRvv22Y7xYX2TeO2nbYU9PsBSnSBlrAc4BWPbru9rHtCYrt180DdZYQcsks9rh5edpk+bKB1XFfCyHdn43JWax3Pap1XdVrX4arOz2oNta7qYFXD6a/3pD6pT+qT+iuO/wARid8UZkrVCQAAAABJRU5ErkJggg=="
# Short mono MP3 of the spoken phrase "the quick brown fox"  (for the ASR test)
SPEECH_MP3_B64="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"

# ── Helper: listening TCP ports owned by a PID ────────────────────────────────
listen_ports_for_pid() {
    local pid="$1"
    if command -v lsof &>/dev/null; then
        lsof -Pan -p "$pid" -iTCP -sTCP:LISTEN 2>/dev/null \
            | awk 'NR>1 {n=split($9,a,":"); print a[n]}'
    elif command -v ss &>/dev/null; then
        ss -ltnpH 2>/dev/null \
            | grep -F "pid=${pid}," \
            | grep -oE ':[0-9]+ ' \
            | tr -d ': '
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# System Hardware  (printed once, host-wide)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# 0. vLLM Instance Discovery
#    Builds the TARGET_PORTS array of unique ports to test.
# ─────────────────────────────────────────────────────────────────────────────
header "0. vLLM Instance Discovery"

TARGET_PORTS=()

if [ -n "$PORT_ARG" ]; then
    # User pinned a specific port — honor it, skip auto-discovery.
    info "Explicit port supplied — testing only ${HOST}:${PORT_ARG}"
    TARGET_PORTS=("$PORT_ARG")
else
    VLLM_PROCS=$(ps aux 2>/dev/null | grep -E 'vllm serve|vllm[._]entrypoints[._]openai|[Vv]llm.*api_server' | grep -v grep || true)

    if [ -z "$VLLM_PROCS" ]; then
        warn "No running vLLM processes found on this host"
    else
        PROC_COUNT=$(echo "$VLLM_PROCS" | wc -l | tr -d ' ')
        info "Found ${PROC_COUNT} vLLM-related process(es); resolving ports..."
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            V_PID=$(echo "$line" | awk '{print $2}')
            V_HOST_BIND=$(echo "$line" | grep -oP '(?<=--host[= ])\S+' || true)
            V_MODEL_NAME=$(echo "$line" | grep -oP '(?<=--served-model-name[= ])\S+' || \
                           echo "$line" | grep -oP '(?<=vllm serve )\S+' || true)

            # Port resolution: prefer the explicit --port flag, then the actual
            # listening socket(s) for the PID, then the vLLM default (8000).
            PORTS_FOUND=()
            V_PORT_FLAG=$(echo "$line" | grep -oP '(?<=--port[= ])\d+' || true)
            if [ -n "$V_PORT_FLAG" ]; then
                PORTS_FOUND+=("$V_PORT_FLAG")
            else
                while IFS= read -r lp; do
                    [ -n "$lp" ] && PORTS_FOUND+=("$lp")
                done < <(listen_ports_for_pid "$V_PID")
                [ "${#PORTS_FOUND[@]}" -eq 0 ] && PORTS_FOUND+=("8000")
            fi

            [ -z "$V_HOST_BIND" ]  && V_HOST_BIND="0.0.0.0"
            [ -z "$V_MODEL_NAME" ] && V_MODEL_NAME="(from /v1/models)"

            for p in "${PORTS_FOUND[@]}"; do
                echo "  PID ${V_PID}  |  port=${p}  host=${V_HOST_BIND}  model=${V_MODEL_NAME}"
                TARGET_PORTS+=("$p")
            done
        done <<< "$VLLM_PROCS"
    fi
fi

# Dedup ports (multiple worker processes can share one server port).
# Portable across bash 3.2 (macOS) — no mapfile.
if [ "${#TARGET_PORTS[@]}" -gt 0 ]; then
    UNIQUE_PORTS=()
    while IFS= read -r p; do
        [ -n "$p" ] && UNIQUE_PORTS+=("$p")
    done < <(printf '%s\n' "${TARGET_PORTS[@]}" | sort -un)
    TARGET_PORTS=("${UNIQUE_PORTS[@]}")
fi

if [ "${#TARGET_PORTS[@]}" -eq 0 ]; then
    fail "No vLLM ports discovered — nothing to test."
    echo "  Hint: start vLLM, or pass an explicit target: $0 ${HOST} <port>"
    exit 1
fi

info "Will test ${#TARGET_PORTS[@]} instance(s) on ${HOST}: ports ${TARGET_PORTS[*]}"

# ─────────────────────────────────────────────────────────────────────────────
# Per-instance test suite.  Called once per discovered port.
# Increments the global FAILURES counter on critical failures.
# ─────────────────────────────────────────────────────────────────────────────
run_instance_tests() {
    local HOST="$1"
    local PORT="$2"
    local BASE_URL="http://${HOST}:${PORT}"
    local FIRST_MODEL=""
    local QUALITY_PASS=0 QUALITY_TOTAL=0   # capability scorecard (tests 10-20)

    echo
    echo -e "${BOLD}${CYAN}#############################################################${RESET}"
    echo -e "${BOLD}${CYAN}#  Instance target: ${BASE_URL}${RESET}"
    echo -e "${BOLD}${CYAN}#############################################################${RESET}"

    # ── 1. Reachability ───────────────────────────────────────────────────────
    header "1. Reachability"
    if curl -sf --max-time 5 "${BASE_URL}" &>/dev/null || \
       curl -sf --max-time 5 "${BASE_URL}/health" &>/dev/null || \
       curl -sf --max-time 5 "${BASE_URL}/v1/models" &>/dev/null; then
        pass "Host is reachable at ${BASE_URL}"
    else
        fail "Cannot reach ${BASE_URL}"
        echo "  Hint: check that vLLM is running and the host/port are correct."
        FAILURES=$((FAILURES + 1))
        return
    fi

    # ── 2. Health endpoint ────────────────────────────────────────────────────
    header "2. Health check  (GET /health)"
    local HEALTH
    HEALTH=$(http_get "${BASE_URL}/health" || true)
    if [ -n "$HEALTH" ]; then
        pass "Health endpoint responded: ${HEALTH}"
    else
        warn "/health returned empty or no response (may be unsupported on this version)"
    fi

    # ── 3. Model list ─────────────────────────────────────────────────────────
    header "3. Model list  (GET /v1/models)"
    local MODELS_JSON MODEL_COUNT
    MODELS_JSON=$(http_get "${BASE_URL}/v1/models" || true)
    if [ -z "$MODELS_JSON" ]; then
        fail "No response from /v1/models"
        FAILURES=$((FAILURES + 1))
    else
        MODEL_COUNT=$(echo "$MODELS_JSON" | jq '.data | length' 2>/dev/null || echo 0)
        if [ "$MODEL_COUNT" -eq 0 ]; then
            warn "Model list is empty"
            FAILURES=$((FAILURES + 1))
        else
            pass "Found ${MODEL_COUNT} model(s):"
            echo "$MODELS_JSON" | jq -r '.data[] | "  • \(.id)  (owned_by: \(.owned_by // "n/a"))"'
            # Pick the first model for subsequent tests (discovered, not hardcoded)
            FIRST_MODEL=$(echo "$MODELS_JSON" | jq -r '.data[0].id')
            info "Using model for tests: ${FIRST_MODEL}"
        fi
    fi

    # ── 4. Server info / version ──────────────────────────────────────────────
    header "4. Server info"
    local path RESP
    for path in "/version" "/v1/version" "/info"; do
        RESP=$(http_get "${BASE_URL}${path}" || true)
        if [ -n "$RESP" ]; then
            pass "${path}: ${RESP}"
        fi
    done

    # ── 5. OpenAI-compatible chat completion ──────────────────────────────────
    header "5. Chat completion  (POST /v1/chat/completions)"
    if [ -z "${FIRST_MODEL:-}" ]; then
        warn "Skipping — no model discovered"
    else
        local CHAT_BODY CHAT_RESP CHAT_TEXT USAGE
        CHAT_BODY=$(jq -n \
            --arg model "$FIRST_MODEL" \
            '{
                model: $model,
                max_tokens: 512,
                temperature: 0.1,
                messages: [
                    {role: "system", content: "You are a helpful assistant. Be concise."},
                    {role: "user",   content: "What model are you and what are your key capabilities? What is your training data cut off date. Answer in 3-4 sentences."}
                ]
            }')
        CHAT_RESP=$(http_post "${BASE_URL}/v1/chat/completions" "$CHAT_BODY" || true)
        if [ -z "$CHAT_RESP" ]; then
            fail "No response from /v1/chat/completions"
            FAILURES=$((FAILURES + 1))
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
                FAILURES=$((FAILURES + 1))
            fi
        fi
    fi

    # ── 5b. Generation throughput ─────────────────────────────────────────────
    header "5b. Generation throughput  (completion tokens/sec)"
    if [ -z "${FIRST_MODEL:-}" ]; then
        warn "Skipping — no model discovered"
    else
        info "Benchmarking model: ${FIRST_MODEL}"
        info "Metric: non-streaming completion_tokens / end-to-end request seconds"
        benchmark_chat_tps 2 192
    fi

    # ── 6. OpenAI-compatible text completion ──────────────────────────────────
    header "6. Text completion  (POST /v1/completions)"
    if [ -z "${FIRST_MODEL:-}" ]; then
        warn "Skipping — no model discovered"
    else
        local COMP_BODY COMP_RESP COMP_TEXT
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

    # ── 7. Embeddings endpoint ────────────────────────────────────────────────
    header "7. Embeddings  (POST /v1/embeddings)"
    if [ -z "${FIRST_MODEL:-}" ]; then
        warn "Skipping — no model discovered"
    else
        local EMB_BODY EMB_RESP EMB_LEN
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

    # ── 8. Sampling parameters / model introspection via chat ─────────────────
    header "8. Model self-description prompts"
    if [ -z "${FIRST_MODEL:-}" ]; then
        warn "Skipping — no model discovered"
    else
        local PROMPTS PROMPT BODY RESP TEXT
        PROMPTS=(
            "What is your max context window length in tokens?"
            "List any special capabilities you have, such as vision, code, tool use, or multilingual support."
            "What languages can you respond in?"
        )
        for PROMPT in "${PROMPTS[@]}"; do
            BODY=$(jq -n \
                --arg model "$FIRST_MODEL" \
                --arg prompt "$PROMPT" \
                '{
                    model: $model,
                    max_tokens: 512,
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

    # ── 9. Streaming check ────────────────────────────────────────────────────
    header "9. Streaming  (POST /v1/chat/completions  stream=true)"
    if [ -z "${FIRST_MODEL:-}" ]; then
        warn "Skipping — no model discovered"
    else
        local STREAM_BODY STREAM_OUT
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

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL QUALITY & CAPABILITY TESTS (10-20)
    #  Auto-graded against known answers. Tolerant of "thinking"/reasoning models:
    #  the expected answer is matched ANYWHERE in the output, with a generous
    #  token budget so a chain-of-thought preamble doesn't truncate the answer.
    # ══════════════════════════════════════════════════════════════════════════
    if [ -z "${FIRST_MODEL:-}" ]; then
        warn "Skipping capability tests 10-20 — no model discovered"
    else
        local QTOK=1024
        local R CLEAN JSON_ONLY SUM_SRC NEEDLE HAY i PCT
        local VBODY VRESP VTEXT ATMP AREQ ATEXT

        header "10. Reasoning  (multi-step logic)"
        R=$(chat_once "Alice is older than Bob. Carol is younger than Bob. Who is the oldest of the three? Reply with only the name." "$QTOK")
        grade "Logical ordering -> Alice" "$R" "alice"

        header "11. Math  (word problem)"
        R=$(chat_once "A shirt costs \$40. It is discounted 25%, then 10% sales tax is added to the discounted price. What is the final price in dollars? Reply with only the number." "$QTOK")
        grade "Arithmetic -> 33" "$R" "(^|[^0-9.])33([^0-9]|\$)"

        header "12. Text summarization"
        SUM_SRC="Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy stored in glucose. It takes place in the chloroplasts, uses carbon dioxide and water, and releases oxygen as a byproduct. This process is the foundation of most food chains on Earth."
        R=$(chat_once "Summarize the following text in one short sentence:\n\n${SUM_SRC}" "$QTOK")
        grade "Summary captures the core topic" "$R" "photosynthes"

        header "13. Instruction following  (strict JSON)"
        R=$(chat_once "Respond with ONLY minified JSON, no markdown and no code fences, of the exact form {\"city\":\"\",\"country\":\"\"} giving the capital of France." "$QTOK")
        CLEAN=$(printf '%s' "$R" | sed -E 's/```json//g; s/```//g')
        JSON_ONLY=$(printf '%s' "$CLEAN" | grep -oE '\{[^{}]*\}' | tail -1)
        QUALITY_TOTAL=$((QUALITY_TOTAL + 1))
        if [ -n "$JSON_ONLY" ] && printf '%s' "$JSON_ONLY" | jq -e '.city' >/dev/null 2>&1 \
             && printf '%s' "$JSON_ONLY" | jq -r '.city' | grep -qi 'paris'; then
            pass "Valid JSON, city=Paris: ${JSON_ONLY}"
            QUALITY_PASS=$((QUALITY_PASS + 1))
        else
            warn "Did not return valid JSON with city=Paris"
            [ -n "$R" ] && echo "     got: $(printf '%s' "$R" | tr '\n' ' ' | head -c 160)"
        fi

        header "14. Code generation"
        R=$(chat_once "Write a Python function named is_prime(n) that returns True if n is prime. Output only the code." "$QTOK")
        grade "Defines is_prime()" "$R" "def[[:space:]]+is_prime"

        header "15. Factual knowledge"
        R=$(chat_once "What is the chemical symbol for gold? Reply with only the symbol." "$QTOK")
        grade "Gold -> Au" "$R" "(^|[^A-Za-z])Au([^A-Za-z]|\$)"

        header "16. Long-context needle retrieval"
        NEEDLE="PLUM-4417"
        HAY=""
        for i in $(seq 1 60);   do HAY="${HAY}Log line ${i}: routine status nominal, nothing to report. "; done
        HAY="${HAY}NOTE: the vault access code is ${NEEDLE}. "
        for i in $(seq 61 120); do HAY="${HAY}Log line ${i}: routine status nominal, nothing to report. "; done
        R=$(chat_once "The following is a long log. Find the vault access code buried in it and reply with only the code.\n\n${HAY}" "$QTOK")
        grade "Recalled needle ${NEEDLE}" "$R" "PLUM[- ]?4417"

        header "17. Translation  (multilingual)"
        R=$(chat_once "Translate the phrase 'good morning' into French. Reply with only the translation." "$QTOK")
        grade "EN->FR 'bonjour'" "$R" "bonjour"

        header "18. Sentiment classification"
        R=$(chat_once "Classify the sentiment of this review as POSITIVE or NEGATIVE. Reply with one word.\n\nReview: I absolutely loved this movie, it was fantastic and moving!" "$QTOK")
        grade "Detected POSITIVE" "$R" "positive"

        header "19. Vision / OCR  (multimodal image input)"
        VBODY=$(jq -n --arg m "$FIRST_MODEL" --arg url "data:image/png;base64,${OCR_PNG_B64}" \
            '{model:$m, max_tokens:64, temperature:0,
              messages:[{role:"user", content:[
                  {type:"text", text:"What text is written in this image? Reply with only the exact text."},
                  {type:"image_url", image_url:{url:$url}}]}]}')
        VRESP=$(http_post "${BASE_URL}/v1/chat/completions" "$VBODY" || true)
        VTEXT=$(printf '%s' "$VRESP" | jq -r '.choices[0].message.content // empty' 2>/dev/null || true)
        if [ -z "$VRESP" ] || [ -z "$VTEXT" ]; then
            warn "Vision not supported by this model (no multimodal image input) — skipped"
        else
            grade "OCR read 'VLLM-OCR-7392'" "$VTEXT" "VLLM[- ]?OCR[- ]?7392|OCR[- ]?7392"
        fi

        header "20. Audio processing  (speech-to-text)"
        ATMP="${TMPDIR:-/tmp}/vllm_asr_$$_${PORT}.mp3"
        printf '%s' "$SPEECH_MP3_B64" | b64decode > "$ATMP" 2>/dev/null || true
        AREQ=$(curl -sf --max-time 60 \
            -F "file=@${ATMP};type=audio/mpeg" \
            -F "model=${FIRST_MODEL}" \
            "${BASE_URL}/v1/audio/transcriptions" 2>/dev/null || true)
        rm -f "$ATMP" 2>/dev/null || true
        if [ -z "$AREQ" ]; then
            warn "Audio/ASR not supported (no /v1/audio/transcriptions — needs a Whisper/ASR model) — skipped"
        else
            ATEXT=$(printf '%s' "$AREQ" | jq -r '.text // empty' 2>/dev/null || true)
            [ -z "$ATEXT" ] && ATEXT="$AREQ"
            grade "Transcribed 'the quick brown fox'" "$ATEXT" "quick|brown|fox"
        fi

        # ── Capability scorecard ──────────────────────────────────────────────
        header "Capability Scorecard — ${BASE_URL}"
        info "Model : ${FIRST_MODEL}"
        if [ "$QUALITY_TOTAL" -gt 0 ]; then
            PCT=$(( QUALITY_PASS * 100 / QUALITY_TOTAL ))
            info "Quality score : ${QUALITY_PASS}/${QUALITY_TOTAL} graded checks passed (${PCT}%)"
        else
            warn "No graded checks were run"
        fi
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Run the suite against every discovered instance.
# ─────────────────────────────────────────────────────────────────────────────
for p in "${TARGET_PORTS[@]}"; do
    run_instance_tests "$HOST" "$p"
done

# ── Summary ───────────────────────────────────────────────────────────────────
header "Summary"
info "Tested ${#TARGET_PORTS[@]} instance(s) on ${HOST}: ports ${TARGET_PORTS[*]}"
if [ "$FAILURES" -eq 0 ]; then
    pass "All critical checks passed across all instances"
else
    fail "${FAILURES} critical check(s) failed across all instances"
fi

exit "$FAILURES"
