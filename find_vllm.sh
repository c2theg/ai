#!/usr/bin/env bash
#   By: Christopher Gray
#
#   Updated: 6/18/2026
#   Version: 0.0.4
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
#   ./find_vllm.sh                # human-readable report
#   ./find_vllm.sh --json         # machine-readable JSON array
#   ./find_vllm.sh --host 1.2.3.4 # also probe a remote host's API (best-effort)
#
# No root required for discovery, but run with sudo to see processes/ports
# owned by other users.

set -uo pipefail

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
JSON=0
REMOTE_HOST=""
CURL_TIMEOUT=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) JSON=1; shift ;;
    --host) REMOTE_HOST="${2:-}"; shift 2 ;;
    --timeout) CURL_TIMEOUT="${2:-3}"; shift 2 ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
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
  pgrep -af 'vllm|vllm\.entrypoints|api_server' 2>/dev/null \
    | grep -Ev 'find_vllm\.sh|grep|pgrep' \
    | awk '{print $1}' | sort -u
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

# ----------------------------------------------------------------------------
# GPU snapshot (once)
# ----------------------------------------------------------------------------
declare -A GPU_BY_PID  # pid -> "gpuidx:memMiB; ..."
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
  done < <(nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid \
            --format=csv,noheader,nounits 2>/dev/null)
fi

# Find the GPU usage for a process tree (parent pid + children).
gpu_for_pid() {
  local pid="$1" acc=""
  acc="${GPU_BY_PID[$pid]:-}"
  # include child PIDs (vLLM spawns workers)
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    acc="${acc}${GPU_BY_PID[$child]:-}"
  done
  printf '%s' "${acc%%; }"
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
  user=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
  etime=$(ps -o etime= -p "$pid" 2>/dev/null | xargs)
  cmd=$(ps -o args= -p "$pid" 2>/dev/null)

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
  echo
fi

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
    fi
  done
  echo
fi

if [[ $JSON -eq 1 ]]; then
  echo "$RESULTS_JSON" | { [[ $HAVE_JQ -eq 1 ]] && jq . || cat; }
fi
