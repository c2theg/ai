#!/usr/bin/env bash
# Christopher Gray  |  vllm_killer.sh  |  Version: 1.1.1  |  7/11/2026
# Stop every vLLM instance (and the sleep watchdog) to free all memory and start
#
# Download and install:
#   wget --no-cache -O 'vllm_killer.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/vllm_killer.sh' && chmod u+x vllm_killer.sh
#
#
# from scratch. Companion to install_ai_spark_vllm.sh.
#
# Usage:
#   ./vllm_killer.sh            — kill all vLLM processes + sleep watchdog
#   ./vllm_killer.sh --docker   — also stop OpenWebUI / Qdrant / SearXNG containers
#   ./vllm_killer.sh -d         — same as --docker
#   ./vllm_killer.sh --dry-run  — show what WOULD be killed, kill nothing
#   ./vllm_killer.sh --help
#
# It sends SIGTERM first for a clean shutdown, waits, then SIGKILL (-9) to any
# stragglers (a wedged CUDA process can ignore signals until its kernels finish).

set -uo pipefail

KILL_DOCKER=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --docker|-d)  KILL_DOCKER=1 ;;
        --dry-run|-n) DRY_RUN=1 ;;
        -h|--help)
            sed -n '/^# Usage:/,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "⚠️  Unknown argument: $arg (see --help)" ;;
    esac
done

# Process-name patterns that identify a vLLM process (server, engine core,
# workers) or the watchdog this stack spawns.
PATTERNS=(
    "vllm serve"
    "vllm.entrypoints"
    "VLLM::EngineCore"
    "vllm.engine"
    "vllm.v1.engine"
    "-m vllm"
    "sleep_watchdog.sh"
)

# Echo the PIDs matching any pattern (deduped, one per line). Excludes this
# script itself so a pattern can never match the killer.
_matching_pids() {
    local pat pids=""
    for pat in "${PATTERNS[@]}"; do
        pids="$pids $(pgrep -f "$pat" 2>/dev/null)"
    done
    # Dedupe, drop our own PID and our parent shell.
    printf '%s\n' $pids | sort -un | grep -vxE "$$|$PPID" 2>/dev/null
}

# Read _matching_pids into the named array (bash 3.2-safe; no mapfile).
_load_pids() {
    local _name="$1" _line
    eval "$_name=()"
    while IFS= read -r _line; do
        [ -n "$_line" ] && eval "$_name+=(\"\$_line\")"
    done < <(_matching_pids)
}

# Ports to probe for a live model — covers the installer's catalog ports and the
# sequential BASE_PORT (8010+) block. Closed localhost ports refuse instantly, so
# probing the whole range stays fast.
PROBE_PORTS=$(seq 8000 8040)

# Print the models currently answering on any probed port (served id from the
# OpenAI-compatible /v1/models endpoint). Echoes how many were found on stdout's
# last line is avoided — callers read the printed list; returns 0 if any live.
_show_live_models() {
    local p resp id any=0
    for p in $PROBE_PORTS; do
        resp=$(curl -sf --max-time 1 "http://localhost:${p}/v1/models" 2>/dev/null) || continue
        if command -v jq >/dev/null 2>&1; then
            id=$(echo "$resp" | jq -r '.data[].id' 2>/dev/null | paste -sd ',' -)
        else
            id=$(echo "$resp" | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' \
                 | sed -E 's/.*"([^"]*)"$/\1/' | paste -sd ',' -)
        fi
        [ -z "$id" ] && id="(up — model id unknown)"
        printf "    port %-6s : %s\n" "$p" "$id"
        any=1
    done
    [ "$any" = "0" ] && echo "    (none responding)"
}

# Print a memory line ("<free> GB free of <total> GB total") from /proc/meminfo.
_mem_line() {
    [ -r /proc/meminfo ] || { echo "    (/proc/meminfo unavailable — non-Linux host)"; return; }
    local mt ma
    mt=$(awk '/^MemTotal:/{print $2}'     /proc/meminfo)
    ma=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    printf "    %d GB free of %d GB total\n" "$(( ma / 1024 / 1024 ))" "$(( mt / 1024 / 1024 ))"
}

echo ""
echo "  ══ vLLM KILLER ══════════════════════════════════════════════════"
echo "  ── BEFORE ───────────────────────────────────────────────────────"

# ---- Memory before ----
echo "  Memory:"
_mem_line

# ---- Models live in vLLM before (API probe) ----
echo "  Models live in vLLM:"
_show_live_models

# ---- Show what will be killed ----
echo "  Processes:"
_load_pids PIDS
if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "  No vLLM processes are running — nothing to kill."
else
    echo "  Found ${#PIDS[@]} vLLM-related process(es):"
    for pid in "${PIDS[@]}"; do
        # Query rss and args separately so whitespace can't shift the columns.
        # RSS in MB ≈ GPU memory on the GB10's unified pool.
        rss_kb=$(ps -p "$pid" -o rss=   2>/dev/null | tr -d ' ')
        args=$(ps -p "$pid" -o args=    2>/dev/null | cut -c1-90)
        [ -n "$args" ] && printf "    pid %-8s ~%6d MB  %s\n" "$pid" "$(( ${rss_kb:-0} / 1024 ))" "$args"
    done
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "  --dry-run: no processes were killed."
    [ "$KILL_DOCKER" -eq 1 ] && echo "  --dry-run: would also stop containers open-webui / searxng / qdrant."
    echo "  ════════════════════════════════════════════════════════════════"
    exit 0
fi

# ---- Kill: SIGTERM, wait, then SIGKILL any survivors ----
if [ "${#PIDS[@]}" -gt 0 ]; then
    echo "  → Sending SIGTERM (graceful)…"
    kill -15 "${PIDS[@]}" 2>/dev/null || true
    sleep 3

    _load_pids SURVIVORS
    if [ "${#SURVIVORS[@]}" -gt 0 ]; then
        echo "  → ${#SURVIVORS[@]} still alive — sending SIGKILL (-9)…"
        kill -9 "${SURVIVORS[@]}" 2>/dev/null || true
        sleep 2
    fi

    # Final sweep by pattern in case new children appeared.
    for pat in "${PATTERNS[@]}"; do
        pkill -9 -f "$pat" 2>/dev/null || true
    done
    sleep 1
fi

# ---- Optional: stop the stack's docker containers ----
if [ "$KILL_DOCKER" -eq 1 ]; then
    if command -v docker >/dev/null 2>&1; then
        echo "  → Stopping containers: open-webui, searxng, qdrant…"
        docker stop open-webui searxng qdrant 2>/dev/null || true
        docker rm   open-webui searxng qdrant 2>/dev/null || true
    else
        echo "  ⚠️  docker not found — skipping container shutdown."
    fi
fi

# ---- AFTER ----
echo ""
echo "  ── AFTER ────────────────────────────────────────────────────────"

# ---- Verify processes ----
_load_pids LEFT
echo "  Processes:"
if [ "${#LEFT[@]}" -eq 0 ]; then
    echo "    ✅ All vLLM processes stopped."
else
    echo "    ⚠️  ${#LEFT[@]} still present (may be finishing CUDA cleanup):"
    for pid in "${LEFT[@]}"; do
        ps -p "$pid" -o pid=,args= 2>/dev/null | sed 's/^/       /'
    done
    echo "       Re-run in a few seconds, or inspect: ps -p <pid> -o pid,args"
fi

# ---- Models live in vLLM after (should be none) ----
echo "  Models live in vLLM:"
_show_live_models

# ---- Memory after ----
echo "  Memory:"
_mem_line

# ---- GPU (best effort; unified memory reports N/A for per-GPU memory) ----
if command -v nvidia-smi >/dev/null 2>&1; then
    gutil=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
    [[ "$gutil" =~ ^[0-9]+$ ]] && printf "  GPU compute   : %d%% utilization\n" "$gutil"
fi

echo "  ════════════════════════════════════════════════════════════════"
