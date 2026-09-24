#!/usr/bin/env bash
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


Version:  0.0.5
Last Updated:  9/24/2026


"
# Downloads unsloth GGUF models (Hugging Face) for llama.cpp onto an Ubuntu box.
# Robust: resumes partials, skips already-complete files, retries on network
# errors -- safe to run by hand or from cron.
#
# Target: /home/ubuntu/models           (override: MODELS_DIR env or 1st arg)
# Source: Hugging Face (unsloth GGUF repos, public -- no token needed)
#
# Original request (note: #1 was a /blob/ HTML link, corrected to /resolve/ below):
#   https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q4_K_XL.gguf    (fixed: blob -> resolve)
#   https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
#   https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/resolve/main/qwen-image-2.1-Q3_K_XL.gguf
#   https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf   (requested as "DD-Q4_K_XL"; assumed typo for UD)
#
# Usage:
#    ./download_llama_cpp_models.sh [MODELS_DIR]
#
# Environment:
#   MODELS_DIR   Where to save files   (default: /home/ubuntu/models, or 1st arg)
#   FORCE=1      Re-download even if the file already exists (default: skip existing)
#   HF_TOKEN     Optional Hugging Face token (only needed for gated/large limits)

set -euo pipefail

# ============================================================
# llama.cpp - model downloader
# unsloth GGUF from Hugging Face
# ============================================================

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

# MODELS_DIR priority: 1st argument > $MODELS_DIR env > default
#MODELS_DIR="${1:-${MODELS_DIR:-/home/ubuntu/models}}"
MODELS_DIR="${1:-${MODELS_DIR:-/opt/ai/models}}"

FORCE="${FORCE:-0}"

# Progress bar only when attached to a terminal (keep cron output quiet).
SHOW_PROGRESS=0
if [[ -t 1 ]]; then
  SHOW_PROGRESS=1
fi

# URL list. Filenames derive from each URL's path.
URLS=(
    "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_XL.gguf"
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
    "https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/resolve/main/qwen-image-2.1-Q3_K_XL.gguf"
    "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf"
)

echo
echo "=============================================="
echo " llama.cpp - model downloader"
echo " unsloth GGUF from Hugging Face"
echo "=============================================="
echo "Target directory  : $MODELS_DIR"
echo "Force re-download : $FORCE"
echo "Models            : ${#URLS[@]}"
echo

# -------------------------------------------------------------
# Check that we're running as the expected user (interactive only)
# -------------------------------------------------------------

if [[ "$(id -un)" != "ubuntu" ]] && [[ -t 0 ]]; then
    echo "WARNING: This script was written for the 'ubuntu' user."
    echo "Current user: $(id -un)"
    echo
    read -rp "Continue anyway? [y/N] " answer
    [[ "$answer" =~ ^[Yy]$ ]] || exit 1
    echo
fi

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

# Human-readable byte count (arg = bytes, prints e.g. "18.3 G").
human() {
    awk -v b="${1:-0}" 'BEGIN{
        split("B K M G T P", u, " ");
        i = 1;
        while (b >= 1024 && i < 6) { b = b / 1024; i++; }
        printf("%.*f %s", (i > 1 ? 1 : 0), b, u[i]);
    }'
}

# Total size (bytes) of a remote resource, or 0 if unknown.
# Parses the Content-Length header of the redirect-resolved response.
content_length() {
    local size=0
    size=$(curl -sIL --connect-timeout 30 --max-time 60 "$1" 2>/dev/null \
              | awk -F: 'tolower($1) ~ /content-length/ { v = $2 } END { gsub(/\r/, "", v); print v + 0 }')
    [[ "$size" =~ ^[0-9]+$ ]] || size=0
    printf '%s' "$size"
}

# -------------------------------------------------------------
# Prepare target directory
# -------------------------------------------------------------

echo "[1/2] Preparing target directory..."
mkdir -p "$MODELS_DIR"
echo "       $MODELS_DIR"
echo

# -------------------------------------------------------------
# Download each model (resume + skip + retry)
# -------------------------------------------------------------

echo "[2/2] Downloading models..."
echo

FAILED=0
COUNT=0
SKIPPED=0

# Format seconds as H:MM:SS or M:SS.
fmt_time() {
    local s="${1:-0}"
    if (( s >= 3600 )); then
        printf '%d:%02d:%02d' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
    else
        printf '%d:%02d' $((s / 60)) $((s % 60))
    fi
}

# Draw a live progress bar for a background curl until it exits.
#   $1 = curl pid   $2 = .part file   $3 = expected total bytes (0 = unknown)
# Shows: [######----] 42%  7.6 G / 18.3 G  52.4 M/s  ETA 3:25
progress_monitor() {
    local pid="$1" file="$2" total="$3"
    local width=30 cur prev tick=1 rate=0 pct filled bar eta line
    prev=$(stat -c%s "$file" 2>/dev/null || echo 0)

    while kill -0 "$pid" 2>/dev/null; do
        sleep "$tick"
        cur=$(stat -c%s "$file" 2>/dev/null || echo 0)
        # Smooth the instantaneous rate a little so the display doesn't jitter.
        rate=$(( (rate * 2 + (cur - prev) / tick) / 3 ))
        prev="$cur"

        if [[ "$total" -gt 0 ]]; then
            pct=$(( cur * 100 / total ))
            (( pct > 100 )) && pct=100
            filled=$(( pct * width / 100 ))
            printf -v bar '%*s' "$filled" ''
            bar="${bar// /█}"
            printf -v line '%*s' $((width - filled)) ''
            bar="${bar}${line// /░}"
            if [[ "$rate" -gt 0 ]]; then
                eta="ETA $(fmt_time $(( (total - cur) / rate )))"
            else
                eta="ETA --:--"
            fi
            printf '\r\033[K   %s %3d%%  %s / %s  %s/s  %s' \
                "$bar" "$pct" "$(human "$cur")" "$(human "$total")" "$(human "$rate")" "$eta"
        else
            printf '\r\033[K   %s  %s/s' "$(human "$cur")" "$(human "$rate")"
        fi
    done
    printf '\r\033[K'
}

# Run one curl attempt (resuming the .part file), with a progress bar on a tty.
run_curl() {
    local url="$1" part="$2" expected="$3"
    curl -fsSL -C - \
         --connect-timeout 30 \
         --retry 5 --retry-delay 10 --retry-all-errors \
         ${HF_TOKEN:+-H "Authorization: Bearer ${HF_TOKEN}"} \
         -o "$part" "$url" &
    local pid=$!
    if [[ "$SHOW_PROGRESS" == "1" ]]; then
        progress_monitor "$pid" "$part" "$expected"
    fi
    wait "$pid"
}

download_one() {
    local url="$1" base="$2"
    local part="$MODELS_DIR/$base.part"
    local dest="$MODELS_DIR/$base"
    local expected="${3:-0}"

    echo "   [get ] downloading:      $base"

    local rc=0
    local cur=0 last=0 attempts=0
    while :; do
        cur=$(stat -c%s "$part" 2>/dev/null || echo 0)

        # Stop when we have everything we expect.
        if [[ "$expected" -gt 0 && "$cur" -ge "$expected" ]]; then
            break
        fi

        last="$cur"
        attempts=$((attempts + 1))

        # Resume what we have and continue to the end. --retry handles
        # transient drops inside one call; the outer loop handles repeated
        # drops by resuming from where the .part file left off.
        if run_curl "$url" "$part" "$expected"; then
            # Unknown expected size: a clean curl exit means we're done.
            if [[ "$expected" -eq 0 ]]; then
                if [[ -s "$part" ]]; then
                    break
                fi
                echo "   [warn] $base returned empty; retrying"
            fi
        else
            rc=$?
            echo "   [warn] network error on $base (curl exit $rc); will resume"
        fi

        # If we know the length, make sure we actually reached it.
        if [[ "$expected" -gt 0 ]]; then
            cur=$(stat -c%s "$part" 2>/dev/null || echo 0)
            if [[ "$cur" -ge "$expected" ]]; then
                break
            fi
        fi

        # Bail out after a bounded number of stalled attempts.
        if [[ $attempts -ge 12 ]]; then
            if [[ "$expected" -gt 0 ]]; then
                echo "   [fail] $base - stalled at $(human "$cur") of $(human "$expected")"
            else
                echo "   [fail] $base - no complete download after $attempts attempts"
            fi
            return 1
        fi
        sleep 3
    done

    # Verify final size when we knew it.
    if [[ "$expected" -gt 0 ]]; then
        local sz
        sz=$(stat -c%s "$part" 2>/dev/null || echo 0)
        if [[ "$sz" -ne "$expected" ]]; then
            echo "   [fail] $base - size mismatch: got $(human "$sz"), want $(human "$expected")"
            echo "         partial left at: $part"
            return 1
        fi
    fi

    # Move into final place, leaving a clean .part behind to resume from.
    mv -f "$part" "$dest"
    local final_size
    final_size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    echo "   [done] $base -> $dest    ($(human "$final_size"))"
    return 0
}

for url in "${URLS[@]}"; do
    base="$(basename "$url")"

    # Skip files that already exist. Downloads go to <name>.part and are only
    # renamed once complete, so a present .gguf is always a finished file.
    if [[ "$FORCE" != "1" && -s "$MODELS_DIR/$base" ]]; then
        echo "   [skip] already exists:   $base    ($(human "$(stat -c%s "$MODELS_DIR/$base")"))"
        SKIPPED=$((SKIPPED + 1))
        echo
        continue
    fi

    expected=$(content_length "$url")
    if download_one "$url" "$base" "$expected"; then
        COUNT=$((COUNT + 1))
    else
        FAILED=$((FAILED + 1))
    fi
    echo
done

# -------------------------------------------------------------
# Summary
# -------------------------------------------------------------

echo "=============================================="
echo " Download summary"
echo "=============================================="
echo "  target : $MODELS_DIR"
echo "  done   : $COUNT      skipped : $SKIPPED      failed : $FAILED"
echo
echo "GGUF models on disk:"
shopt -s nullglob
ggufs=("$MODELS_DIR"/*.gguf)
if [[ ${#ggufs[@]} -eq 0 ]]; then
    echo "  (none yet)"
else
    for f in "${ggufs[@]}"; do
        printf "  %8s   %s\n" "$(human "$(stat -c%s "$f")")" "$(basename -- "$f")"
    done
fi
echo
echo "=============================================="
if [[ $FAILED -gt 0 ]]; then
    echo " Completed with $FAILED failure(s). Re-run to resume:"
    echo "    $0 ${MODELS_DIR}"
    echo "=============================================="
    exit 1
fi
echo " All models downloaded."
echo "=============================================="
exit 0
