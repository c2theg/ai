#!/usr/bin/env bash
# Christopher Gray  |  Version: 0.1.1  |  Update: 7/15/2026
# Ollama smoke test - IP/port based tester similar to tester_vllm.sh.
#
# Download: 
#   curl -sSL https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/tester_ollama.sh -o tester_ollama.sh
#
# Usage:
#   ./tester_ollama.sh                         # prompts; Enter defaults to 127.0.0.1
#   ./tester_ollama.sh 10.11.1.10              # uses port 11434
#   ./tester_ollama.sh 10.11.1.10 11434        # explicit host + port
#   ./tester_ollama.sh http://10.11.1.10       # full URL, default port 11434
#   ./tester_ollama.sh http://10.11.1.10:11435 # full URL with explicit port
#   ./tester_ollama.sh 10.11.1.10 11434 llama3 # explicit model
set -euo pipefail

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

prompt_target_if_needed() {
    if [ "$#" -gt 0 ] && [ -n "${1:-}" ]; then
        TARGET_RAW="$1"
        return 0
    fi

    echo ""
    echo -n "Enter Ollama IP/host or full URL [127.0.0.1]: "
    read -r TARGET_RAW
    TARGET_RAW="${TARGET_RAW:-127.0.0.1}"
}

build_base_url() {
    local target="$1" port="${2:-}"
    if [[ "$target" =~ ^https?:// ]]; then
        target="${target%/}"
        if [[ "$target" =~ ^(https?://)([^/:]+)(:([0-9]+))?(/.*)?$ ]]; then
            local scheme="${BASH_REMATCH[1]}"
            local host="${BASH_REMATCH[2]}"
            local url_port="${BASH_REMATCH[4]:-}"
            port="${port:-${url_port:-11434}}"
            target="${scheme}${host}:${port}"
        elif [ -z "$port" ]; then
            target="${target}:11434"
        else
            target="${target}:${port}"
        fi
        printf '%s' "$target"
        return 0
    fi
    port="${port:-11434}"
    printf 'http://%s:%s' "$target" "$port"
}

http_get() {
    local url="$1"
    curl -sf --max-time 10 "$url" 2>/dev/null
}

http_post() {
    local url="$1"
    local body="$2"
    curl -sf --max-time 120 \
        -H "Content-Type: application/json" \
        -d "$body" \
        "$url" 2>/dev/null
}

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

wall_tokens_per_second() {
    local tokens="$1" elapsed_ms="$2"
    awk -v tok="$tokens" -v ms="$elapsed_ms" \
        'BEGIN { if (tok > 0 && ms > 0) printf "%.2f", tok * 1000 / ms; else printf "0.00" }'
}

ollama_eval_tokens_per_second() {
    local tokens="$1" eval_duration_ns="$2"
    awk -v tok="$tokens" -v ns="$eval_duration_ns" \
        'BEGIN { if (tok > 0 && ns > 0) printf "%.2f", tok * 1000000000 / ns; else printf "0.00" }'
}

format_ns_seconds() {
    local ns="$1"
    awk -v ns="$ns" 'BEGIN { if (ns > 0) printf "%.2f", ns / 1000000000; else printf "0.00" }'
}

bytes_to_human() {
    local bytes="$1"
    awk -v b="${bytes:-0}" 'BEGIN {
        split("B KiB MiB GiB TiB", u, " ");
        i=1;
        while (b >= 1024 && i < 5) { b /= 1024; i++ }
        printf "%.1f %s", b, u[i]
    }'
}

chat_once() {
    local prompt="$1" maxtok="${2:-256}" body
    body=$(jq -n --arg m "$FIRST_MODEL" --arg p "$prompt" --argjson mt "$maxtok" \
        '{model:$m, stream:false, options:{temperature:0, num_predict:$mt},
          messages:[{role:"user", content:$p}]}')
    (http_post "${BASE_URL}/api/chat" "$body" || true) \
        | jq -r '.message.content // empty' 2>/dev/null
}

grade() {
    local label="$1" resp="$2" pat="$3"
    QUALITY_TOTAL=$((QUALITY_TOTAL + 1))
    if [ -n "$resp" ] && printf '%s' "$resp" | grep -qiE "$pat"; then
        pass "${label}"
        QUALITY_PASS=$((QUALITY_PASS + 1))
        return 0
    fi
    warn "${label} - expected /${pat}/"
    [ -n "$resp" ] && echo "     got: $(printf '%s' "$resp" | tr '\n' ' ' | head -c 160)"
    return 0
}

benchmark_chat_tps() {
    local runs="${1:-2}" maxtok="${2:-192}"
    local i body resp start_ms end_ms elapsed_ms
    local eval_count eval_duration total_duration prompt_eval_count eval_tps wall_tps
    local ok=0 total_eval=0 total_eval_ns=0 total_wall_ms=0

    for i in $(seq 1 "$runs"); do
        body=$(jq -n --arg m "$FIRST_MODEL" --argjson mt "$maxtok" \
            '{
                model: $m,
                stream: false,
                options: {temperature: 0, num_predict: $mt},
                messages: [
                    {role: "user", content: "Write a dense technical paragraph about local AI inference performance, batching, KV cache behavior, and latency. Continue until you naturally reach the token budget."}
                ]
            }')

        start_ms=$(now_ms)
        resp=$(http_post "${BASE_URL}/api/chat" "$body" || true)
        end_ms=$(now_ms)
        elapsed_ms=$((end_ms - start_ms))
        [ "$elapsed_ms" -le 0 ] && elapsed_ms=1

        if [ -z "$resp" ]; then
            warn "Throughput run ${i}/${runs}: no response"
            continue
        fi

        eval_count=$(printf '%s' "$resp" | jq -r '.eval_count // empty' 2>/dev/null || true)
        eval_duration=$(printf '%s' "$resp" | jq -r '.eval_duration // empty' 2>/dev/null || true)
        total_duration=$(printf '%s' "$resp" | jq -r '.total_duration // empty' 2>/dev/null || true)
        prompt_eval_count=$(printf '%s' "$resp" | jq -r '.prompt_eval_count // empty' 2>/dev/null || true)

        if ! [[ "$eval_count" =~ ^[0-9]+$ ]] || [ "$eval_count" -eq 0 ]; then
            warn "Throughput run ${i}/${runs}: response did not include eval_count"
            continue
        fi
        [[ "$eval_duration" =~ ^[0-9]+$ ]] || eval_duration=0

        eval_tps=$(ollama_eval_tokens_per_second "$eval_count" "$eval_duration")
        wall_tps=$(wall_tokens_per_second "$eval_count" "$elapsed_ms")
        printf "  Run %d/%d : %4d eval tokens in %.2fs eval / %.2fs wall  =>  %s tok/s eval, %s tok/s wall" \
            "$i" "$runs" "$eval_count" "$(format_ns_seconds "$eval_duration")" \
            "$(awk -v ms="$elapsed_ms" 'BEGIN { printf "%.2f", ms / 1000 }')" "$eval_tps" "$wall_tps"
        [ -n "$prompt_eval_count" ] && printf "  (prompt_eval=%s)" "$prompt_eval_count"
        [ -n "$total_duration" ] && printf "  total_duration=%ss" "$(format_ns_seconds "$total_duration")"
        printf "\n"

        ok=$((ok + 1))
        total_eval=$((total_eval + eval_count))
        total_eval_ns=$((total_eval_ns + eval_duration))
        total_wall_ms=$((total_wall_ms + elapsed_ms))
    done

    if [ "$ok" -gt 0 ]; then
        eval_tps=$(ollama_eval_tokens_per_second "$total_eval" "$total_eval_ns")
        wall_tps=$(wall_tokens_per_second "$total_eval" "$total_wall_ms")
        pass "Average generation throughput: ${eval_tps} eval tokens/sec (${wall_tps} wall tok/s, ${total_eval} tokens across ${ok} run(s))"
    else
        warn "Could not calculate throughput for ${FIRST_MODEL}"
    fi
}

print_system_hardware() {
    header "System Hardware"
    local os_type arch
    os_type=$(uname -s)
    arch=$(uname -m)
    info "Platform : ${os_type} / ${arch}"

    if [ "$os_type" = "Linux" ]; then
        local cpu_name cpu_cores ram_info
        cpu_name=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || \
                   lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2 | xargs || echo "unknown")
        cpu_cores=$(nproc 2>/dev/null || echo "?")
        ram_info=$(free -h 2>/dev/null | awk '/^Mem:/{printf "total=%s  used=%s  free=%s", $2, $3, $4}' || echo "unknown")
        info "CPU      : ${cpu_name}  (${cpu_cores} cores)"
        info "RAM      : ${ram_info}"
    elif [ "$os_type" = "Darwin" ]; then
        local cpu_name total_ram
        cpu_name=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
        total_ram=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.1f GB", $1/1073741824}' || echo "unknown")
        info "CPU      : ${cpu_name}"
        info "RAM      : ${total_ram}"
    fi

    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        local driver_ver cuda_ver gpu_count
        driver_ver=$(nvidia-smi 2>/dev/null | grep -oP 'Driver Version: \K[\d.]+' || echo "n/a")
        cuda_ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' || echo "n/a")
        gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 | xargs || echo "?")
        info "GPU      : NVIDIA  (driver=${driver_ver}  CUDA=${cuda_ver}  count=${gpu_count})"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null | \
            while IFS=',' read -r idx name mtot mused mfree util temp; do
                name=$(echo "$name" | xargs); mtot=$(echo "$mtot" | xargs)
                mused=$(echo "$mused" | xargs); mfree=$(echo "$mfree" | xargs)
                util=$(echo "$util" | xargs); temp=$(echo "$temp" | xargs)
                echo "  [GPU ${idx}] ${name}"
                echo "           VRAM : ${mtot} MiB total  |  ${mused} MiB used  |  ${mfree} MiB free"
                echo "           Util : ${util}%  |  Temp: ${temp} C"
            done
    elif [ "$os_type" = "Darwin" ]; then
        local gpu_info
        gpu_info=$(system_profiler SPDisplaysDataType 2>/dev/null | \
            grep -E 'Chipset Model|Total Number of Cores|VRAM|Metal' | \
            sed 's/^ *//' | paste -sd'  ' - || echo "unknown")
        info "GPU      : Apple Silicon / Metal - ${gpu_info}"
    else
        warn "GPU      : None detected locally, or testing a remote Ollama host"
    fi
}

run_ollama_tests() {
    local BASE_URL="$1"
    local MODEL_ARG="${2:-}"
    local FIRST_MODEL=""
    local QUALITY_PASS=0 QUALITY_TOTAL=0

    echo
    echo -e "${BOLD}${CYAN}#############################################################${RESET}"
    echo -e "${BOLD}${CYAN}#  Ollama target: ${BASE_URL}${RESET}"
    echo -e "${BOLD}${CYAN}#############################################################${RESET}"

    header "1. Reachability"
    if curl -sf --max-time 5 "${BASE_URL}/api/version" &>/dev/null || \
       curl -sf --max-time 5 "${BASE_URL}/api/tags" &>/dev/null; then
        pass "Ollama is reachable at ${BASE_URL}"
    else
        fail "Cannot reach Ollama at ${BASE_URL}"
        echo "  Hint: start Ollama, expose the port, or pass the correct target:"
        echo "        $0 <ip-or-host> [port] [model]"
        FAILURES=$((FAILURES + 1))
        return
    fi

    header "2. Server version"
    local VERSION_JSON
    VERSION_JSON=$(http_get "${BASE_URL}/api/version" || true)
    if [ -n "$VERSION_JSON" ]; then
        pass "Version: $(printf '%s' "$VERSION_JSON" | jq -r '.version // .' 2>/dev/null)"
    else
        warn "/api/version returned no response"
    fi

    header "3. Model list  (GET /api/tags)"
    local MODELS_JSON MODEL_COUNT
    MODELS_JSON=$(http_get "${BASE_URL}/api/tags" || true)
    if [ -z "$MODELS_JSON" ]; then
        fail "No response from /api/tags"
        FAILURES=$((FAILURES + 1))
        return
    fi

    MODEL_COUNT=$(printf '%s' "$MODELS_JSON" | jq '.models | length' 2>/dev/null || echo 0)
    if [ "$MODEL_COUNT" -eq 0 ]; then
        fail "Ollama model list is empty"
        echo "  Hint: pull a model first, e.g. ollama pull llama3.1"
        FAILURES=$((FAILURES + 1))
        return
    fi

    pass "Found ${MODEL_COUNT} local model(s):"
    printf '%s' "$MODELS_JSON" | jq -r '.models[] |
        "  - \(.name)  size=\((.size // 0) | tostring)  modified=\(.modified_at // "n/a")"' \
        | while IFS= read -r line; do
            size=$(printf '%s' "$line" | sed -n 's/.*size=\([0-9][0-9]*\).*/\1/p')
            if [ -n "$size" ]; then
                human=$(bytes_to_human "$size")
                printf '%s\n' "$(printf '%s' "$line" | sed "s/size=${size}/size=${human}/")"
            else
                printf '%s\n' "$line"
            fi
        done

    if [ -n "$MODEL_ARG" ]; then
        if printf '%s' "$MODELS_JSON" | jq -e --arg m "$MODEL_ARG" '.models[] | select(.name == $m)' >/dev/null 2>&1; then
            FIRST_MODEL="$MODEL_ARG"
        else
            fail "Requested model '${MODEL_ARG}' was not found in /api/tags"
            FAILURES=$((FAILURES + 1))
            return
        fi
    else
        FIRST_MODEL=$(printf '%s' "$MODELS_JSON" | jq -r '.models[0].name')
    fi
    info "Using model for tests: ${FIRST_MODEL}"

    header "4. Model details  (POST /api/show)"
    local SHOW_BODY SHOW_RESP
    SHOW_BODY=$(jq -n --arg m "$FIRST_MODEL" '{model:$m}')
    SHOW_RESP=$(http_post "${BASE_URL}/api/show" "$SHOW_BODY" || true)
    if [ -n "$SHOW_RESP" ]; then
        pass "Model metadata responded"
        printf '%s' "$SHOW_RESP" | jq -r '
            "  family      : \(.details.family // "n/a")",
            "  parameter   : \(.details.parameter_size // "n/a")",
            "  quantization: \(.details.quantization_level // "n/a")",
            "  format      : \(.details.format // "n/a")"
        ' 2>/dev/null || true
    else
        warn "/api/show returned no response"
    fi

    header "5. Chat completion  (POST /api/chat)"
    local CHAT_BODY CHAT_RESP CHAT_TEXT
    CHAT_BODY=$(jq -n --arg m "$FIRST_MODEL" \
        '{
            model: $m,
            stream: false,
            options: {temperature: 0.1, num_predict: 512},
            messages: [
                {role: "system", content: "You are a helpful assistant. Be concise."},
                {role: "user", content: "What model are you and what are your key capabilities? Answer in 3-4 sentences."}
            ]
        }')
    CHAT_RESP=$(http_post "${BASE_URL}/api/chat" "$CHAT_BODY" || true)
    CHAT_TEXT=$(printf '%s' "$CHAT_RESP" | jq -r '.message.content // empty' 2>/dev/null || true)
    if [ -n "$CHAT_TEXT" ]; then
        pass "Chat completion succeeded"
        echo "  Response : ${CHAT_TEXT}"
        printf '%s' "$CHAT_RESP" | jq -r '
            "  Tokens   : prompt_eval=\(.prompt_eval_count // "n/a") eval=\(.eval_count // "n/a")",
            "  Timing   : total=\((.total_duration // 0) / 1000000000)s eval=\((.eval_duration // 0) / 1000000000)s"
        ' 2>/dev/null || true
    else
        fail "No usable response from /api/chat"
        [ -n "$CHAT_RESP" ] && echo "  Raw: $(printf '%s' "$CHAT_RESP" | head -c 400)"
        FAILURES=$((FAILURES + 1))
    fi

    header "5b. Generation throughput"
    info "Metric: Ollama eval_count / eval_duration, plus wall-clock tokens/sec"
    benchmark_chat_tps 2 192

    header "6. Generate endpoint  (POST /api/generate)"
    local GEN_BODY GEN_RESP GEN_TEXT
    GEN_BODY=$(jq -n --arg m "$FIRST_MODEL" \
        '{model:$m, stream:false, prompt:"The capital of France is", options:{temperature:0, num_predict:20}}')
    GEN_RESP=$(http_post "${BASE_URL}/api/generate" "$GEN_BODY" || true)
    GEN_TEXT=$(printf '%s' "$GEN_RESP" | jq -r '.response // empty' 2>/dev/null || true)
    if [ -n "$GEN_TEXT" ]; then
        pass "Generate succeeded: \"The capital of France is${GEN_TEXT}\""
    else
        warn "/api/generate returned no usable text"
    fi

    header "7. Embeddings"
    local EMB_BODY EMB_RESP EMB_LEN
    EMB_BODY=$(jq -n --arg m "$FIRST_MODEL" '{model:$m, input:"Hello, world!"}')
    EMB_RESP=$(http_post "${BASE_URL}/api/embed" "$EMB_BODY" || true)
    EMB_LEN=$(printf '%s' "$EMB_RESP" | jq '.embeddings[0] | length' 2>/dev/null || echo 0)
    if [ "${EMB_LEN:-0}" -gt 0 ] 2>/dev/null; then
        pass "/api/embed returned vector length ${EMB_LEN}"
    else
        EMB_BODY=$(jq -n --arg m "$FIRST_MODEL" '{model:$m, prompt:"Hello, world!"}')
        EMB_RESP=$(http_post "${BASE_URL}/api/embeddings" "$EMB_BODY" || true)
        EMB_LEN=$(printf '%s' "$EMB_RESP" | jq '.embedding | length' 2>/dev/null || echo 0)
        if [ "${EMB_LEN:-0}" -gt 0 ] 2>/dev/null; then
            pass "/api/embeddings returned vector length ${EMB_LEN}"
        else
            warn "Embeddings not supported by this model or Ollama version"
        fi
    fi

    header "8. Model self-description prompts"
    local PROMPTS PROMPT R
    PROMPTS=(
        "What is your approximate context window length in tokens?"
        "List any special capabilities you have, such as code, tool use, vision, or multilingual support."
        "What languages can you respond in?"
    )
    for PROMPT in "${PROMPTS[@]}"; do
        R=$(chat_once "$PROMPT" 256 || true)
        if [ -n "$R" ]; then
            echo -e "  ${BOLD}Q:${RESET} ${PROMPT}"
            echo "  A: ${R}"
            echo
        else
            warn "No response for: ${PROMPT}"
        fi
    done

    header "9. Streaming check"
    local STREAM_BODY STREAM_OUT
    STREAM_BODY=$(jq -n --arg m "$FIRST_MODEL" \
        '{model:$m, stream:true, options:{temperature:0, num_predict:30},
          messages:[{role:"user", content:"Say hello in one sentence."}]}')
    STREAM_OUT=$(curl -sf --max-time 20 \
        -H "Content-Type: application/json" \
        -d "$STREAM_BODY" \
        "${BASE_URL}/api/chat" 2>/dev/null | head -5 || true)
    if printf '%s' "$STREAM_OUT" | jq -e 'select(.message.content != null or .done != null)' >/dev/null 2>&1; then
        pass "Streaming response received (first chunks):"
        printf '%s\n' "$STREAM_OUT" | head -3 | sed 's/^/  /'
    else
        warn "Streaming check inconclusive"
    fi

    local QTOK=1024 CLEAN JSON_ONLY SUM_SRC NEEDLE HAY i PCT

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

    header "Capability Scorecard - ${BASE_URL}"
    info "Model : ${FIRST_MODEL}"
    if [ "$QUALITY_TOTAL" -gt 0 ]; then
        PCT=$(( QUALITY_PASS * 100 / QUALITY_TOTAL ))
        info "Quality score : ${QUALITY_PASS}/${QUALITY_TOTAL} graded checks passed (${PCT}%)"
    else
        warn "No graded checks were run"
    fi
}

TARGET_RAW=""
prompt_target_if_needed "${1:-}"
PORT_ARG="${2:-}"
MODEL_ARG="${3:-}"

if [ -n "$PORT_ARG" ] && ! [[ "$PORT_ARG" =~ ^[0-9]+$ ]] && [ -z "$MODEL_ARG" ]; then
    MODEL_ARG="$PORT_ARG"
    PORT_ARG=""
fi

BASE_URL=$(build_base_url "$TARGET_RAW" "$PORT_ARG")

print_system_hardware
run_ollama_tests "$BASE_URL" "$MODEL_ARG"

header "Summary"
info "Tested Ollama target: ${BASE_URL}"
if [ "$FAILURES" -eq 0 ]; then
    pass "All critical checks passed"
else
    fail "${FAILURES} critical check(s) failed"
fi

exit "$FAILURES"
