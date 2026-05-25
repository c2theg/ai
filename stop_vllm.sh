#!/usr/bin/env bash
# Christopher Gray  |  Version: 0.0.7  |  Update: 5/25/2026
# Stop all running vLLM instances on this host
# Usage: ./stop_vllm.sh [-f]   (-f = force, no confirmation prompt)
#
#
# Update Yourself:
#    wget --no-cache -O 'stop_vllm.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/stop_vllm.sh' && chmod u+x stop_vllm.sh
#
set -euo pipefail

FORCE=0
[ "${1:-}" = "-f" ] && FORCE=1

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

header "vLLM Stop Utility"

# Find top-level vLLM serve processes (not engine core children)
VLLM_PROCS=$(ps aux 2>/dev/null | grep -E 'vllm serve|vllm\.entrypoints\.openai' | grep -v grep || true)

if [ -z "$VLLM_PROCS" ]; then
    warn "No running vLLM processes found — nothing to stop"
    exit 0
fi

PROC_COUNT=$(echo "$VLLM_PROCS" | wc -l | tr -d ' ')
info "Found ${PROC_COUNT} vLLM instance(s):"
echo

PIDS=()
while IFS= read -r line; do
    V_PID=$(echo "$line" | awk '{print $2}')
    V_PORT=$(echo "$line" | grep -oP '(?<=--port )\d+' || true)
    V_MODEL=$(echo "$line" | grep -oP '(?<=--served-model-name )\S+' || true)
    [ -z "$V_MODEL" ] && V_MODEL=$(echo "$line" | grep -oP '(?<=vllm serve )\S+' || true)
    [ -z "$V_PORT" ]  && V_PORT="unknown"
    [ -z "$V_MODEL" ] && V_MODEL="unknown"
    echo "  PID ${V_PID}  |  port=${V_PORT}  model=${V_MODEL}"
    PIDS+=("$V_PID")
done <<< "$VLLM_PROCS"

echo

if [ "$FORCE" -eq 0 ]; then
    read -r -p "Stop all ${PROC_COUNT} instance(s)? [y/N] " CONFIRM
    [[ "$CONFIRM" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }
fi

for PID in "${PIDS[@]}"; do
    if ! kill -0 "$PID" 2>/dev/null; then
        warn "PID ${PID} already gone"
        continue
    fi

    info "Sending SIGTERM to PID ${PID}..."
    kill -TERM "$PID" 2>/dev/null || true

    # Wait up to 15s for graceful shutdown
    for i in $(seq 1 15); do
        sleep 1
        if ! kill -0 "$PID" 2>/dev/null; then
            pass "PID ${PID} stopped (after ${i}s)"
            break
        fi
        if [ "$i" -eq 15 ]; then
            warn "PID ${PID} did not exit after 15s — sending SIGKILL"
            kill -KILL "$PID" 2>/dev/null || true
            sleep 1
            if ! kill -0 "$PID" 2>/dev/null; then
                pass "PID ${PID} killed"
            else
                fail "PID ${PID} could not be killed"
            fi
        fi
    done
done

# Clean up any orphaned EngineCore workers
ORPHANS=$(ps aux 2>/dev/null | grep -E 'VLLM::EngineCor' | grep -v grep | awk '{print $2}' || true)
if [ -n "$ORPHANS" ]; then
    warn "Cleaning up orphaned EngineCore workers..."
    echo "$ORPHANS" | xargs kill -KILL 2>/dev/null || true
    pass "Orphans removed"
fi

echo
pass "Done. Run tester_vllm.sh to verify nothing is listening."
