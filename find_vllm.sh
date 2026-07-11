#!/usr/bin/env bash
#   By: Christopher Gray
#
#   Updated: 7/11/2026
#   Version: 0.0.6
#   wget --no-cache -O 'find_vllm.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/find_vllm.sh' && chmod u+x find_vllm.sh
#
# find_vllm.sh — Discover all vLLM instances on an Ubuntu 24.04+ host.
#
# For each running vLLM server it reports:
#   - PID, user, uptime, full command line
#   - Listening port(s)
#   - Model(s) served (queried live from the OpenAI-compatible /v1/models API)
#   - Served model name, max context length, served-model-id
#   - Key launch args (tensor-parallel, dtype, gpu-mem-util, quantization, etc.)
#   - GPU(s) used and current VRAM consumption (if nvidia-smi is present)
#   - Health/readiness status
#
# Detection is layered so it works whether vLLM runs bare-metal, in a venv,
# under systemd, in Docker, or via `vllm serve` / `python -m vllm...`.
#
# Usage:
#   ./find_vllm.sh                       # human-readable report
#   ./find_vllm.sh --json                # machine-readable JSON array
#   ./find_vllm.sh --host 1.2.3.4        # also probe a remote host's API
#   ./find_vllm.sh --all-gpu             # ALSO list every GPU compute process
#                                        #   (not just vLLM): PID, VRAM, user,
#                                        #   uptime and full command line, so
#                                        #   nothing hiding on the GPU is a mystery
#
#   # Send a live test prompt to every detected/reachable instance and print
#   # the model's reply (verifies the server actually generates, not just /health):
#   ./find_vllm.sh --test                       # uses a default test prompt
#   ./find_vllm.sh --prompt "Say hi in 5 words" # custom prompt (implies --test)
#   ./find_vllm.sh --test --max-tokens 128      # cap the reply length
#   ./find_vllm.sh --host 1.2.3.4 --test        # test a remote instance too
#
# No root required for discovery, but run with sudo to see processes/ports
# owned by other users.

set -uo pipefail

# Requires bash 4+ for associative arrays (GPU_BY_PID etc.). Ubuntu 24.04 ships
# bash 5; macOS ships 3.2 — fail with a clear hint instead of `declare -A` spew.
if [[ -z "${BASH_VERSINFO:-}" || "${BASH_VERSINFO[0]}" -lt 4 ]]; then
  echo "find_vllm.sh requires bash 4+ (found ${BASH_VERSION:-unknown}). On macOS: brew install bash." >&2
  exit 1
fi

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
JSON=0
REMOTE_HOST=""
CURL_TIMEOUT=3
ALLGPU=0
TEST=0
TEST_PROMPT="Reply with a single short sentence confirming you are online and name the model you are."
MAX_TOKENS=64
GEN_TIMEOUT=60

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) JSON=1; shift ;;
    --all-gpu|--all) ALLGPU=1; shift ;;
    --host) REMOTE_HOST="${2:-}"; shift 2 ;;
    --timeout) CURL_TIMEOUT="${2:-3}"; shift 2 ;;
    --test) TEST=1; shift ;;
    --prompt) TEST=1; TEST_PROMPT="${2:-}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:-64}"; shift 2 ;;
    --gen-timeout) GEN_TIMEOUT="${2:-60}"; shift 2 ;;
    -h|--help)
      sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------------
# Colors (disabled when not a TTY or in JSON mode)
# ----------------------------------------------------------------------------
if [[ -t 1 && $JSON -eq 0 ]]; then
  C_HDR=$'\033[1;36m'; C_KEY=$'\033[1;33m'; C_OK=$'\033[1;32m'
  C_WARN=$'\033[1;31m'; C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else
  C_HDR=""; C_KEY=""; C_OK=""; C_WARN=""; C_DIM=""; C_RST=""
fi

have() { command -v "$1" >/dev/null 2>&1; }

# ----------------------------------------------------------------------------
# Dependency hints (non-fatal)
# ----------------------------------------------------------------------------
HAVE_SS=0;     have ss        && HAVE_SS=1
HAVE_LSOF=0;   have lsof      && HAVE_LSOF=1
HAVE_CURL=0;   have curl      && HAVE_CURL=1
HAVE_NVSMI=0;  have nvidia-smi && HAVE_NVSMI=1
HAVE_JQ=0;     have jq        && HAVE_JQ=1
HAVE_DOCKER=0; have docker    && docker info >/dev/null 2>&1 && HAVE_DOCKER=1

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# Find PIDs that look like vLLM. Matches:
#   - `vllm serve ...` / the `vllm` entrypoint
#   - `python -m vllm.entrypoints...` (openai api server etc.)
#   - api_server / vllm.entrypoints in the command line
find_vllm_pids() {
  # -f matches against full command line; -a prints the command too, then we cut.
  # Anchor on "vllm" (covers `vllm serve` and vllm.entrypoints.openai.api_server)
  # or the openai api_server module path. A bare "api_server" is deliberately NOT
  # matched — it false-positives on unrelated servers (e.g. --api_server_url ...).
  pgrep -af 'vllm|openai\.api_server' 2>/dev/null \
    | grep -Ev 'find_vllm\.sh|grep|pgrep' \
    | awk '{print $1}' \
    | grep -Ev "^($$|$PPID)\$" \
    | sort -u
}

# Extract a flag value from a command line. Handles "--flag value" and
# "--flag=value". $1=cmdline, $2=flag (without --).
arg_val() {
  local cmd="$1" flag="$2" val=""
  # --flag=value
  val=$(grep -oE -- "--${flag}=[^[:space:]]+" <<<"$cmd" | head -1 | cut -d= -f2-)
  if [[ -z "$val" ]]; then
    # --flag value
    val=$(grep -oE -- "--${flag}[[:space:]]+[^[:space:]]+" <<<"$cmd" | head -1 \
          | awk '{print $2}')
  fi
  printf '%s' "$val"
}

# Discover listening TCP ports for a PID.
ports_for_pid() {
  local pid="$1" out=""
  if [[ $HAVE_SS -eq 1 ]]; then
    out=$(ss -ltnp 2>/dev/null | grep -E "pid=${pid}," \
          | grep -oE '[0-9.:*\[\]]+:[0-9]+ ' | grep -oE '[0-9]+ *$' | tr -d ' ' | sort -un)
  fi
  if [[ -z "$out" && $HAVE_LSOF -eq 1 ]]; then
    out=$(lsof -nP -iTCP -sTCP:LISTEN -a -p "$pid" 2>/dev/null \
          | awk 'NR>1{print $9}' | sed -E 's/.*:([0-9]+)$/\1/' | sort -un)
  fi
  printf '%s' "$out"
}

# Query the OpenAI-compatible API on a port. Sets globals MODEL_INFO_*.
# Returns 0 if reachable.
probe_api() {
  local host="$1" port="$2"
  [[ $HAVE_CURL -eq 1 ]] || return 1
  local base="http://${host}:${port}"
  local models health
  models=$(curl -fsS --max-time "$CURL_TIMEOUT" "${base}/v1/models" 2>/dev/null) || return 1
  health=$(curl -fsS --max-time "$CURL_TIMEOUT" -o /dev/null -w '%{http_code}' \
           "${base}/health" 2>/dev/null || echo "n/a")
  API_MODELS_JSON="$models"
  API_HEALTH="$health"
  return 0
}

# Pretty-print model list from API JSON.
parse_models() {
  local json="$1"
  if [[ $HAVE_JQ -eq 1 ]]; then
    echo "$json" | jq -r '
      .data[]? |
      "    - id: \(.id)\n" +
      "      max_model_len: \(.max_model_len // "?")\n" +
      "      root: \(.root // "?")"' 2>/dev/null
  else
    # crude fallback without jq
    echo "$json" | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]+"' \
      | sed -E 's/.*"id"[^"]*"([^"]+)"/    - id: \1/'
  fi
}

# Pull the first served model id out of a /v1/models JSON blob.
first_model_id() {
  local json="$1"
  if [[ $HAVE_JQ -eq 1 ]]; then
    echo "$json" | jq -r '.data[0].id // empty' 2>/dev/null
  else
    echo "$json" | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 \
      | sed -E 's/.*"id"[^"]*"([^"]+)"/\1/'
  fi
}

# Send a test prompt to a vLLM OpenAI-compatible endpoint and print the reply.
# $1=host $2=port $3=model-id (may be empty -> derived from /v1/models)
# Tries /v1/chat/completions first, falls back to /v1/completions.
send_test_prompt() {
  local host="$1" port="$2" model="$3"
  [[ $HAVE_CURL -eq 1 ]] || { echo "    ${C_WARN}(curl missing — cannot send test prompt)${C_RST}"; return 1; }
  local base="http://${host}:${port}"

  # Resolve a model id if we weren't handed one.
  if [[ -z "$model" ]]; then
    local mj; mj=$(curl -fsS --max-time "$CURL_TIMEOUT" "${base}/v1/models" 2>/dev/null)
    model=$(first_model_id "$mj")
  fi
  [[ -z "$model" ]] && model="default"

  # Build JSON payload (jq for safe escaping; manual fallback otherwise).
  local chat_payload comp_payload
  if [[ $HAVE_JQ -eq 1 ]]; then
    chat_payload=$(jq -nc --arg m "$model" --arg p "$TEST_PROMPT" --argjson mt "$MAX_TOKENS" \
      '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0.2,stream:false}')
    comp_payload=$(jq -nc --arg m "$model" --arg p "$TEST_PROMPT" --argjson mt "$MAX_TOKENS" \
      '{model:$m,prompt:$p,max_tokens:$mt,temperature:0.2,stream:false}')
  else
    local esc; esc=$(printf '%s' "$TEST_PROMPT" | sed 's/\\/\\\\/g; s/"/\\"/g')
    chat_payload="{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"${esc}\"}],\"max_tokens\":${MAX_TOKENS},\"temperature\":0.2,\"stream\":false}"
    comp_payload="{\"model\":\"${model}\",\"prompt\":\"${esc}\",\"max_tokens\":${MAX_TOKENS},\"temperature\":0.2,\"stream\":false}"
  fi

  echo "    ${C_DIM}prompt:${C_RST} ${TEST_PROMPT}"
  local t0 t1 elapsed resp reply endpoint
  t0=$(date +%s.%N 2>/dev/null || date +%s)

  # 1) chat completions
  resp=$(curl -fsS --max-time "$GEN_TIMEOUT" -H 'Content-Type: application/json' \
         -d "$chat_payload" "${base}/v1/chat/completions" 2>/dev/null)
  if [[ -n "$resp" ]]; then
    endpoint="/v1/chat/completions"
    if [[ $HAVE_JQ -eq 1 ]]; then reply=$(echo "$resp" | jq -r '.choices[0].message.content // empty' 2>/dev/null); fi
  fi

  # 2) fall back to legacy completions
  if [[ -z "${reply:-}" ]]; then
    resp=$(curl -fsS --max-time "$GEN_TIMEOUT" -H 'Content-Type: application/json' \
           -d "$comp_payload" "${base}/v1/completions" 2>/dev/null)
    if [[ -n "$resp" ]]; then
      endpoint="/v1/completions"
      if [[ $HAVE_JQ -eq 1 ]]; then reply=$(echo "$resp" | jq -r '.choices[0].text // empty' 2>/dev/null); fi
    fi
  fi

  t1=$(date +%s.%N 2>/dev/null || date +%s)
  elapsed=$(awk "BEGIN{printf \"%.2f\", ${t1}-${t0}}" 2>/dev/null)

  if [[ -z "$resp" ]]; then
    echo "    ${C_WARN}no response (request failed or timed out after ${GEN_TIMEOUT}s)${C_RST}"
    return 1
  fi
  if [[ -z "${reply:-}" ]]; then
    # jq unavailable or schema unexpected — show raw payload trimmed.
    echo "    ${C_WARN}could not parse reply; raw response:${C_RST}"
    echo "      ${resp:0:600}"
    return 1
  fi

  echo "    ${C_KEY}via:${C_RST} ${endpoint}  ${C_KEY}model:${C_RST} ${model}  ${C_KEY}latency:${C_RST} ${elapsed}s"
  echo "    ${C_OK}reply:${C_RST} ${reply}"
  return 0
}

# ----------------------------------------------------------------------------
# GPU snapshot (once)
# ----------------------------------------------------------------------------
declare -A GPU_BY_PID   # pid -> "gpuidx:memMiB; ..."
declare -A ALLGPU_MEM   # pid -> total VRAM in MiB (summed across GPUs)
declare -A ALLGPU_GPUS  # pid -> "gpu0,gpu1,"
if [[ $HAVE_NVSMI -eq 1 ]]; then
  # query compute apps: pid, used mem, gpu uuid -> map uuid to index
  declare -A UUID_IDX
  while IFS=, read -r idx uuid; do
    UUID_IDX["$(echo "$uuid" | xargs)"]="$(echo "$idx" | xargs)"
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)

  while IFS=, read -r pid mem uuid; do
    pid="$(echo "$pid" | xargs)"; mem="$(echo "$mem" | xargs)"
    uuid="$(echo "$uuid" | xargs)"
    [[ -z "$pid" || "$pid" == "[N/A]" ]] && continue
    idx="${UUID_IDX[$uuid]:-?}"
    GPU_BY_PID["$pid"]="${GPU_BY_PID[$pid]:-}gpu${idx}:${mem}; "
    # Aggregate for the --all-gpu report. On unified-memory parts (e.g. GB10)
    # used_memory may be [N/A]; treat non-numeric as 0 for summing/sorting.
    memnum="$mem"; [[ "$memnum" =~ ^[0-9]+$ ]] || memnum=0
    ALLGPU_MEM["$pid"]=$(( ${ALLGPU_MEM["$pid"]:-0} + memnum ))
    ALLGPU_GPUS["$pid"]="${ALLGPU_GPUS[$pid]:-}gpu${idx},"
  done < <(nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid \
            --format=csv,noheader,nounits 2>/dev/null)
fi

# Print every descendant PID of $1 (children, grandchildren, ...), depth-first.
# vLLM v1 spawns its EngineCore/workers more than one level down, so a single
# pgrep -P is not enough to attribute their VRAM to the server process.
descendants() {
  local pid="$1" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    echo "$child"
    descendants "$child"
  done
}

# Find the GPU usage for a process tree (parent pid + all descendants).
gpu_for_pid() {
  local pid="$1" acc="" child
  acc="${GPU_BY_PID[$pid]:-}"
  for child in $(descendants "$pid"); do
    acc="${acc}${GPU_BY_PID[$child]:-}"
  done
  printf '%s' "${acc%%; }"
}

# --all-gpu: human-readable table of EVERY GPU compute process, biggest first.
print_all_gpu() {
  if [[ $HAVE_NVSMI -eq 0 ]]; then
    echo "${C_WARN}--all-gpu: nvidia-smi not found — cannot list GPU processes${C_RST}"
    return
  fi
  echo "${C_HDR}=== All GPU compute processes (largest VRAM first) ===${C_RST}"
  if [[ ${#ALLGPU_MEM[@]} -eq 0 ]]; then
    echo "  ${C_DIM}(none reported by nvidia-smi — run with sudo to see other users' processes)${C_RST}"
    echo
    return
  fi
  printf "  ${C_KEY}%-8s %-11s %-10s %-9s %-14s %s${C_RST}\n" \
    "PID" "VRAM(MiB)" "GPU(s)" "USER" "UPTIME" "COMMAND"
  local pid mem gpus user etime cmd tag
  # Sort PIDs by aggregated VRAM, descending.
  for pid in $(for p in "${!ALLGPU_MEM[@]}"; do echo "${ALLGPU_MEM[$p]} $p"; done \
               | sort -rn | awk '{print $2}'); do
    mem="${ALLGPU_MEM[$pid]}"
    gpus="${ALLGPU_GPUS[$pid]%,}"
    user=$(ps -o user=  -p "$pid" 2>/dev/null | xargs)
    etime=$(ps -o etime= -p "$pid" 2>/dev/null | xargs)
    cmd=$(ps -o args=  -p "$pid" 2>/dev/null)
    [[ -z "$cmd" ]] && cmd="(process not visible — try sudo)"
    tag=""; grep -qiE 'vllm|openai\.api_server' <<<"$cmd" && tag=" ${C_OK}[vllm]${C_RST}"
    printf "  %-8s %-11s %-10s %-9s %-14s %s%s\n" \
      "$pid" "$mem" "${gpus:-?}" "${user:-?}" "${etime:-?}" "${cmd:0:90}" "$tag"
  done
  echo
}

# --all-gpu in JSON mode: array of {pid,vram_mib,gpus,user,uptime,is_vllm,cmdline}.
allgpu_json() {
  local arr="[]" pid mem gpus user etime cmd isv obj
  for pid in "${!ALLGPU_MEM[@]}"; do
    mem="${ALLGPU_MEM[$pid]}"
    gpus="${ALLGPU_GPUS[$pid]%,}"
    user=$(ps -o user=  -p "$pid" 2>/dev/null | xargs)
    etime=$(ps -o etime= -p "$pid" 2>/dev/null | xargs)
    cmd=$(ps -o args=  -p "$pid" 2>/dev/null)
    isv=false; grep -qiE 'vllm|openai\.api_server' <<<"$cmd" && isv=true
    obj=$(jq -n --arg pid "$pid" --argjson mem "${mem:-0}" --arg gpus "$gpus" \
      --arg user "$user" --arg etime "$etime" --arg cmd "$cmd" --argjson isv "$isv" \
      '{pid:$pid,vram_mib:$mem,gpus:($gpus|split(",")|map(select(length>0))),
        user:$user,uptime:$etime,is_vllm:$isv,cmdline:$cmd}')
    arr=$(jq -c ". + [${obj}]" <<<"$arr")
  done
  jq 'sort_by(-.vram_mib)' <<<"$arr"
}

# ----------------------------------------------------------------------------
# Collect
# ----------------------------------------------------------------------------
PIDS=$(find_vllm_pids)
RESULTS_JSON="[]"

emit_json_obj() {
  # Build one JSON object from globals. Requires jq for safe escaping.
  if [[ $HAVE_JQ -eq 1 ]]; then
    jq -n \
      --arg pid "$1" --arg user "$2" --arg etime "$3" --arg cmd "$4" \
      --arg ports "$5" --arg model "$6" --arg tp "$7" --arg dtype "$8" \
      --arg gpumem "$9" --arg quant "${10}" --arg maxlen "${11}" \
      --arg gpu "${12}" --arg health "${13}" --arg apimodels "${14}" \
      '{pid:$pid,user:$user,uptime:$etime,ports:($ports|split(" ")|map(select(length>0))),
        served_model:$model,tensor_parallel:$tp,dtype:$dtype,gpu_mem_util:$gpumem,
        quantization:$quant,max_model_len:$maxlen,gpus:$gpu,health:$health,
        cmdline:$cmd,api_models:($apimodels|try fromjson catch null)}'
  fi
}

print_instance() {
  local pid="$1"
  local user etime cmd
  cmd=$(ps -o args= -p "$pid" 2>/dev/null)
  # PID may have exited between discovery and now (or briefly matched our own
  # invocation) — skip ghosts with no command line rather than print blanks.
  [[ -z "${cmd// }" ]] && return
  user=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
  etime=$(ps -o etime= -p "$pid" 2>/dev/null | xargs)

  local ports; ports=$(ports_for_pid "$pid" | tr '\n' ' ' | xargs)
  # vLLM default port is 8000 if none parsed from ss/lsof
  local arg_port; arg_port=$(arg_val "$cmd" "port")
  [[ -z "$ports" && -n "$arg_port" ]] && ports="$arg_port"
  [[ -z "$ports" ]] && ports="$(arg_val "$cmd" port)"

  local model tp dtype gpumem quant maxlen host gpu
  model=$(arg_val "$cmd" "model");                 [[ -z "$model" ]] && model="(default)"
  # served-model-name overrides display name if present
  local served; served=$(arg_val "$cmd" "served-model-name"); [[ -n "$served" ]] && model="$served ($model)"
  tp=$(arg_val "$cmd" "tensor-parallel-size");     [[ -z "$tp" ]] && tp="1"
  dtype=$(arg_val "$cmd" "dtype");                 [[ -z "$dtype" ]] && dtype="auto"
  gpumem=$(arg_val "$cmd" "gpu-memory-utilization");[[ -z "$gpumem" ]] && gpumem="0.9(default)"
  quant=$(arg_val "$cmd" "quantization");          [[ -z "$quant" ]] && quant="none"
  maxlen=$(arg_val "$cmd" "max-model-len");         [[ -z "$maxlen" ]] && maxlen="(model default)"
  host=$(arg_val "$cmd" "host");                   [[ -z "$host" ]] && host="0.0.0.0"
  gpu=$(gpu_for_pid "$pid");                        [[ -z "$gpu" ]] && gpu="(none/unknown)"

  # Probe the live API on the first port
  local probe_port api_health="n/a" api_models_json=""
  probe_port=$(echo "$ports" | awk '{print $1}')
  local probe_host="127.0.0.1"
  [[ "$host" != "0.0.0.0" && "$host" != "::" && -n "$host" ]] && probe_host="$host"
  if [[ -n "$probe_port" ]] && probe_api "$probe_host" "$probe_port"; then
    api_health="$API_HEALTH"
    api_models_json="$API_MODELS_JSON"
  fi

  if [[ $JSON -eq 1 ]]; then
    local obj
    obj=$(emit_json_obj "$pid" "$user" "$etime" "$cmd" "$ports" "$model" "$tp" \
          "$dtype" "$gpumem" "$quant" "$maxlen" "$gpu" "$api_health" "$api_models_json")
    RESULTS_JSON=$(jq -c ". + [${obj}]" <<<"$RESULTS_JSON")
    return
  fi

  echo "${C_HDR}=== vLLM instance — PID ${pid} ===${C_RST}"
  printf "  ${C_KEY}%-18s${C_RST} %s\n" "User:"          "$user"
  printf "  ${C_KEY}%-18s${C_RST} %s\n" "Uptime:"        "$etime"
  printf "  ${C_KEY}%-18s${C_RST} %s\n" "Listen host:"   "$host"
  printf "  ${C_KEY}%-18s${C_RST} %s\n" "Port(s):"       "${ports:-(none detected)}"
  printf "  ${C_KEY}%-18s${C_RST} %s\n" "Model:"         "$model"
  printf "  ${C_KEY}%-18s${C_RST} %s\n" "Tensor parallel:" "$tp"
  printf "  ${C_KEY}%-18s${C_RST} %s\n" "Dtype:"         "$dtype"
  printf "  ${C_KEY}%-18s${C_RST} %s\n" "GPU mem util:"  "$gpumem"
  printf "  ${C_KEY}%-18s${C_RST} %s\n" "Quantization:"  "$quant"
  printf "  ${C_KEY}%-18s${C_RST} %s\n" "Max model len:" "$maxlen"
  printf "  ${C_KEY}%-18s${C_RST} %s\n" "GPU usage:"     "$gpu"

  if [[ "$api_health" == "200" ]]; then
    printf "  ${C_KEY}%-18s${C_RST} ${C_OK}%s${C_RST}\n" "API health:" "OK (200)"
  else
    printf "  ${C_KEY}%-18s${C_RST} ${C_WARN}%s${C_RST}\n" "API health:" "${api_health}"
  fi

  if [[ -n "$api_models_json" ]]; then
    echo "  ${C_KEY}Live model list:${C_RST}"
    parse_models "$api_models_json"
  fi

  # Optional live test prompt (--test / --prompt).
  if [[ $TEST -eq 1 ]]; then
    echo "  ${C_KEY}Test prompt:${C_RST}"
    if [[ -n "$probe_port" && "$api_health" == "200" ]] || [[ -n "$probe_port" && -n "$api_models_json" ]]; then
      send_test_prompt "$probe_host" "$probe_port" "$(first_model_id "$api_models_json")"
    else
      echo "    ${C_WARN}skipped — no reachable API port for this instance${C_RST}"
    fi
  fi

  echo "  ${C_DIM}cmd: ${cmd}${C_RST}"
  echo
}

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
if [[ $JSON -eq 0 ]]; then
  echo "${C_HDR}vLLM discovery on $(hostname) — $(date '+%Y-%m-%d %H:%M:%S')${C_RST}"
  [[ $HAVE_SS -eq 0 && $HAVE_LSOF -eq 0 ]] && \
    echo "${C_WARN}warning: neither 'ss' nor 'lsof' found — port detection limited${C_RST}"
  [[ $HAVE_NVSMI -eq 0 ]] && \
    echo "${C_DIM}note: nvidia-smi not found — GPU/VRAM details unavailable${C_RST}"
  [[ $HAVE_CURL -eq 0 ]] && \
    echo "${C_DIM}note: curl not found — live API model query unavailable${C_RST}"
  [[ $TEST -eq 1 ]] && \
    echo "${C_DIM}test mode: sending a generation prompt to each reachable instance (max_tokens=${MAX_TOKENS}, gen-timeout=${GEN_TIMEOUT}s)${C_RST}"
  echo
fi
# --test output is human-readable; it is suppressed in --json mode.
[[ $TEST -eq 1 && $JSON -eq 1 ]] && \
  echo "warning: --test has no effect with --json (run without --json to see replies)" >&2

if [[ -z "$PIDS" ]]; then
  if [[ $JSON -eq 1 ]]; then echo "[]"; else
    echo "${C_WARN}No vLLM processes found.${C_RST}"
    echo "${C_DIM}(If vLLM runs in Docker, run this inside the container, or check below.)${C_RST}"
  fi
else
  while read -r pid; do
    [[ -n "$pid" ]] && print_instance "$pid"
  done <<<"$PIDS"
fi

# ----------------------------------------------------------------------------
# All GPU compute processes (--all-gpu) — not just vLLM
# ----------------------------------------------------------------------------
if [[ $ALLGPU -eq 1 && $JSON -eq 0 ]]; then
  print_all_gpu
fi

# ----------------------------------------------------------------------------
# Docker containers running vLLM (bonus)
# ----------------------------------------------------------------------------
if [[ $HAVE_DOCKER -eq 1 && $JSON -eq 0 ]]; then
  dock=$(docker ps --format '{{.ID}}\t{{.Image}}\t{{.Ports}}\t{{.Names}}' 2>/dev/null \
         | grep -iE 'vllm' || true)
  if [[ -n "$dock" ]]; then
    echo "${C_HDR}=== vLLM Docker containers ===${C_RST}"
    printf "  %-14s %-30s %-25s %s\n" "CONTAINER" "IMAGE" "PORTS" "NAME"
    while IFS=$'\t' read -r id img ports name; do
      printf "  %-14s %-30s %-25s %s\n" "$id" "$img" "$ports" "$name"
    done <<<"$dock"
    echo
  fi
fi

# ----------------------------------------------------------------------------
# Optional remote API probe
# ----------------------------------------------------------------------------
if [[ -n "$REMOTE_HOST" && $HAVE_CURL -eq 1 ]]; then
  echo "${C_HDR}=== Remote probe: ${REMOTE_HOST} ===${C_RST}"
  for p in 8000 8001 8002 8080 9000; do
    if probe_api "$REMOTE_HOST" "$p"; then
      echo "  ${C_OK}Port ${p}: reachable (health=${API_HEALTH})${C_RST}"
      parse_models "$API_MODELS_JSON"
      if [[ $TEST -eq 1 ]]; then
        echo "  ${C_KEY}Test prompt:${C_RST}"
        send_test_prompt "$REMOTE_HOST" "$p" "$(first_model_id "$API_MODELS_JSON")"
      fi
    fi
  done
  echo
fi

if [[ $JSON -eq 1 ]]; then
  if [[ $ALLGPU -eq 1 && $HAVE_JQ -eq 1 ]]; then
    # Combine: preserve the vLLM array under "vllm", add "all_gpu".
    jq -n --argjson v "$RESULTS_JSON" --argjson g "$(allgpu_json)" \
      '{vllm:$v, all_gpu:$g}'
  else
    echo "$RESULTS_JSON" | { [[ $HAVE_JQ -eq 1 ]] && jq . || cat; }
  fi
fi
