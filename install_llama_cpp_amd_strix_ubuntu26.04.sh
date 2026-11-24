#!/usr/bin/env bash
set -euo pipefail

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


Version:  0.1.0
Last Updated:  11/23/2026



Example: 
    llama-server -m /opt/ai/models/Qwen3.5-9B-UD-Q4_K_XL.gguf --alias qwen-9b --host 0.0.0.0 --port 10022 -ngl 999 -c 32768 --jinja
    

# no thinking: 
    llama-server -m /opt/ai/models/Qwen3.5-9B-UD-Q4_K_XL.gguf --alias qwen-9b --host 0.0.0.0 --port 10022 -ngl 999 -c 32768 --jinja --temp 0.4 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 --reasoning-budget 0


# Start / stop as services (model stays in memory only while running):
    sudo systemctl start  llama-qwen-27b        # port 10020
    sudo systemctl start  llama-qwen-35b-a3b    # port 10021
    sudo systemctl start  llama-qwen-9b         # port 10022
    sudo systemctl start  llama-inkling-mini    # port 10023 (Inkling 2 Mini, text + images)
    sudo systemctl stop   llama-qwen-9b
    sudo systemctl enable llama-qwen-9b         # also start at boot
    journalctl -u llama-qwen-9b -f              # logs

# Stop instances:
    llama-stop              # stop ALL llama-server instances (services + manual)
    llama-stop qwen-9b      # stop just the one with that alias

# See memory use per instance (PID, port, alias, VRAM/GTT, RAM):
    llama-status

# Start a model at boot with cron instead of systemd (crontab -e, as user ubuntu).
# cron has a minimal PATH, so use full paths; sleep gives the GPU/network time to come up.
    @reboot sleep 30 && /usr/local/bin/llama-server -m /opt/ai/models/Qwen3.5-9B-UD-Q4_K_XL.gguf --alias qwen-9b --host 0.0.0.0 --port 10022 -ngl 999 -c 32768 --jinja >> /home/ubuntu/llama-qwen-9b.log 2>&1
# Check it after a reboot:  llama-status   /   tail -f ~/llama-qwen-9b.log
# Use cron OR the systemd unit for a given model, not both (they'd fight over the port).

# Inkling 2 Mini (downloaded by this script, ~16 GB). Skip the download with SKIP_INKLING=1.
# Recommended sampling is temp 1.0 with no truncation; --mmproj enables image input.
    llama-server -m /opt/ai/models/Inkling-2-Mini-UD-IQ4_XS.gguf --mmproj /opt/ai/models/mmproj/Inkling-2-Mini-mmproj-F16.gguf --alias inkling-mini --host 0.0.0.0 --port 10023 -ngl 999 -c 32768 --jinja --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0

"
# ============================================================
# llama.cpp installer for AMD Strix Halo / Ubuntu 26.04
#
# Uses Vulkan/RADV for GPU acceleration.
#
# Models:
#   /opt/ai/models/*.gguf
#   /opt/ai/models/mmproj/*.gguf   (vision projectors, kept out of the model pickers)
#
# Installation:
#   /home/ubuntu/llama.cpp
#
# Commands created:
#   llama
#   llama-server
# ============================================================

MODEL_DIR="/opt/ai/models"
LLAMA_DIR="/home/ubuntu/llama.cpp"

# Skip the slow parts (apt upgrade, dependencies, git pull, rebuild) when
# llama-server is already built. Force a full update/rebuild with:
#   UPDATE=1 ./install_llama_cpp_amd_strix_ubuntu26.04.sh
UPDATE="${UPDATE:-0}"
ALREADY_BUILT=0
if [[ "$UPDATE" != "1" && -x "$LLAMA_DIR/build/bin/llama-server" ]]; then
    ALREADY_BUILT=1
fi

# Inkling 2 Mini needs llama.cpp b12611 or newer (PR #32704, merged
# 2026-11-21). Older builds load the model but give wrong output, so
# force a rebuild if the existing checkout predates that commit. The
# clone is shallow, so compare the HEAD commit date instead of build number.
INKLING_MIN_COMMIT_TS=1795259439   # 2026-11-21T11:10:39Z
if [[ "$ALREADY_BUILT" == "1" ]]; then
    head_ts=$(git -C "$LLAMA_DIR" log -1 --format=%ct 2>/dev/null || echo 0)
    if (( head_ts < INKLING_MIN_COMMIT_TS )); then
        echo "Existing llama.cpp build is too old for Inkling 2 Mini - forcing update/rebuild."
        ALREADY_BUILT=0
    fi
fi

echo
echo "=============================================="
echo " llama.cpp - AMD Strix Halo installer"
echo " Ubuntu 26.04 / Vulkan"
echo "=============================================="
echo

# ------------------------------------------------------------
# Check that we're running as the expected user
# ------------------------------------------------------------

if [[ "$(id -un)" != "ubuntu" ]]; then
    echo "WARNING: This script was written for the 'ubuntu' user."
    echo "Current user: $(id -un)"
    echo
    read -rp "Continue anyway? [y/N] " answer
    [[ "$answer" =~ ^[Yy]$ ]] || exit 1
fi

# ------------------------------------------------------------
# Update system
# ------------------------------------------------------------

echo "[1/10] Updating Ubuntu..."
if [[ "$ALREADY_BUILT" == "1" ]]; then
    echo "llama.cpp already built at $LLAMA_DIR - skipping (UPDATE=1 to force)."
else
    sudo apt-get update
    sudo apt-get -y upgrade
fi

# ------------------------------------------------------------
# Install dependencies
# ------------------------------------------------------------

echo
echo "[2/10] Installing build dependencies..."

if [[ "$ALREADY_BUILT" == "1" ]]; then
    echo "Skipping (already built)."
else
    sudo apt-get install -y \
        git \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        libvulkan-dev \
        vulkan-tools \
        mesa-vulkan-drivers \
        glslc \
        spirv-headers \
        libssl-dev \
        libcurl4-openssl-dev \
        curl \
        ca-certificates
fi

# ------------------------------------------------------------
# Verify Vulkan
# ------------------------------------------------------------

echo
echo "[3/10] Checking Vulkan..."

if ! command -v vulkaninfo >/dev/null 2>&1; then
    echo "ERROR: vulkaninfo was not installed."
    exit 1
fi

echo
echo "Vulkan device summary:"
vulkaninfo --summary || {
    echo
    echo "WARNING: vulkaninfo returned an error."
    echo "Your AMD GPU/Vulkan driver may need attention."
    echo
}

# ------------------------------------------------------------
# Create model directory
# ------------------------------------------------------------

echo
echo "[4/10] Creating model directory..."

mkdir -p "$MODEL_DIR"

# ------------------------------------------------------------
# Clone/update llama.cpp
# ------------------------------------------------------------

echo
echo "[5/10] Getting latest llama.cpp..."

if [[ "$ALREADY_BUILT" == "1" ]]; then
    echo "Skipping (already built)."
elif [[ -d "$LLAMA_DIR/.git" ]]; then
    echo "Existing llama.cpp installation found."
    echo "Updating it..."

    git -C "$LLAMA_DIR" fetch --depth=1 origin
    git -C "$LLAMA_DIR" reset --hard origin/master
else
    git clone --depth=1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

# ------------------------------------------------------------
# Build llama.cpp with Vulkan
# ------------------------------------------------------------

echo
echo "[6/10] Building llama.cpp with Vulkan..."

if [[ "$ALREADY_BUILT" == "1" ]]; then
    echo "Skipping (already built)."
else
    cd "$LLAMA_DIR"

    rm -rf build

    cmake -S . -B build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_VULKAN=ON

    cmake --build build --parallel
fi

# ------------------------------------------------------------
# Install convenient symlinks
# ------------------------------------------------------------

echo
echo "[7/10] Installing commands..."

sudo ln -sf "$LLAMA_DIR/build/bin/llama-cli" /usr/local/bin/llama
sudo ln -sf "$LLAMA_DIR/build/bin/llama-server" /usr/local/bin/llama-server

# Optional useful tools
if [[ -f "$LLAMA_DIR/build/bin/llama-bench" ]]; then
    sudo ln -sf "$LLAMA_DIR/build/bin/llama-bench" /usr/local/bin/llama-bench
fi

# ------------------------------------------------------------
# Create helper launcher
# ------------------------------------------------------------

cat > "$HOME/run-llama.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/opt/ai/models"

mapfile -t MODELS < <(find "$MODEL_DIR" -maxdepth 1 -type f -iname "*.gguf" -printf "%f\n" | sort)

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo
    echo "No GGUF models found in:"
    echo "  $MODEL_DIR"
    echo
    echo "Copy a .gguf model there and run this script again."
    exit 1
fi

echo
echo "Available GGUF models:"
echo

for i in "${!MODELS[@]}"; do
    printf "  %2d) %s\n" "$((i + 1))" "${MODELS[$i]}"
done

echo
read -rp "Select model [1-${#MODELS[@]}]: " choice

if ! [[ "$choice" =~ ^[0-9]+$ ]] ||
   (( choice < 1 || choice > ${#MODELS[@]} )); then
    echo "Invalid selection."
    exit 1
fi

MODEL="${MODEL_DIR}/${MODELS[$((choice - 1))]}"

echo
echo "Starting:"
echo "  $MODEL"
echo
echo "GPU acceleration: Vulkan"
echo

exec llama \
    -m "$MODEL" \
    -ngl 999 \
    -c 32768 \
    -t "$(nproc)" \
    -n -1 \
    --temp 0.2 \
    --top-p 0.9 \
    --repeat-penalty 1.1 \
    -cnv
EOF

chmod +x "$HOME/run-llama.sh"

# ------------------------------------------------------------
# Create server helper
# ------------------------------------------------------------

cat > "$HOME/llama-server.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/opt/ai/models"

mapfile -t MODELS < <(find "$MODEL_DIR" -maxdepth 1 -type f -iname "*.gguf" -printf "%f\n" | sort)

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "No GGUF models found in $MODEL_DIR"
    exit 1
fi

echo
echo "Available GGUF models:"
echo

for i in "${!MODELS[@]}"; do
    printf "  %2d) %s\n" "$((i + 1))" "${MODELS[$i]}"
done

echo
read -rp "Select model [1-${#MODELS[@]}]: " choice

if ! [[ "$choice" =~ ^[0-9]+$ ]] ||
   (( choice < 1 || choice > ${#MODELS[@]} )); then
    echo "Invalid selection."
    exit 1
fi

MODEL="${MODEL_DIR}/${MODELS[$((choice - 1))]}"

echo
echo "Starting llama-server:"
echo "  Model: $MODEL"
echo "  URL:   http://0.0.0.0:10020  (LAN / Tailscale only, enforced by ufw)"
echo

exec llama-server \
    -m "$MODEL" \
    --alias primary \
    -ngl 999 \
    -c 32768 \
    -t "$(nproc)" \
    --host 0.0.0.0 \
    --port 10020 \
    --jinja
EOF

chmod +x "$HOME/llama-server.sh"

# ------------------------------------------------------------
# systemd units (one per model) + llama-stop / llama-status helpers
#
# Units are installed but NOT enabled or started, so nothing loads into
# memory until you run: sudo systemctl start llama-qwen-9b
# ------------------------------------------------------------

echo
echo "Installing systemd units and helper commands..."

# make_unit <name> <gguf file> <alias> <port> [extra llama-server args]
make_unit() {
    local name="$1" file="$2" alias="$3" port="$4" extra="${5:-}"
    sudo tee "/etc/systemd/system/llama-${name}.service" >/dev/null <<UNIT
[Unit]
Description=llama.cpp server - ${alias} (port ${port})
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
ExecStart=/usr/local/bin/llama-server -m ${MODEL_DIR}/${file} --alias ${alias} --host 0.0.0.0 --port ${port} -ngl 999 -c 32768 --jinja${extra:+ ${extra}}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
}

make_unit qwen-27b     Qwen3.8-27B-UD-Q4_K_XL.gguf      qwen-27b     10020
make_unit qwen-35b-a3b Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf  qwen-35b-a3b 10021
make_unit qwen-9b      Qwen3.5-9B-UD-Q4_K_XL.gguf       qwen-9b      10022
make_unit inkling-mini Inkling-2-Mini-UD-IQ4_XS.gguf    inkling-mini 10023 \
    "--mmproj ${MODEL_DIR}/mmproj/Inkling-2-Mini-mmproj-F16.gguf --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0"
sudo systemctl daemon-reload

# llama-stop [alias]  -- stop all instances, or the one with that alias
sudo tee /usr/local/bin/llama-stop >/dev/null <<'EOF'
#!/usr/bin/env bash
# Usage: llama-stop          stop every llama-server (systemd units + manual)
#        llama-stop <alias>  stop only the instance started with --alias <alias>
set -uo pipefail

if [[ $# -eq 0 || "$1" == "all" ]]; then
    units=$(systemctl list-units 'llama-*.service' --no-legend --plain | awk '{print $1}')
    # shellcheck disable=SC2086
    [[ -n "$units" ]] && sudo systemctl stop $units && echo "Stopped: $units"
    pkill -x llama-server && echo "Stopped manually started llama-server processes."
else
    sudo systemctl stop "llama-$1.service" 2>/dev/null && echo "Stopped llama-$1.service"
    pkill -f -- "--alias $1( |,)" && echo "Stopped manual instance with alias $1."
fi
exit 0
EOF
sudo chmod +x /usr/local/bin/llama-stop

# llama-status  -- per-instance GPU + RAM usage
sudo tee /usr/local/bin/llama-status >/dev/null <<'EOF'
#!/usr/bin/env bash
# Per-process memory of running llama-server instances.
#   VRAM = dedicated GPU carve-out, GTT = GPU-visible system RAM (where most
#   of the model lives on a Strix Halo APU), RSS = normal process RAM.
# Run as the same user as the servers (or root) so /proc/<pid>/fdinfo is readable.

printf "%-8s %-7s %-16s %10s %10s %10s\n" PID PORT ALIAS "VRAM MiB" "GTT MiB" "RSS MiB"
for pid in $(pgrep -x llama-server); do
    args=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    port=$(sed -nE 's/.*--port ([0-9]+).*/\1/p' <<<"$args")
    name=$(sed -nE 's/.*--alias ([^ ]+).*/\1/p' <<<"$args")
    read -r vram gtt < <(cat /proc/"$pid"/fdinfo/* 2>/dev/null | awk '
        /^drm-client-id:/ { seen = ($2 in ids); ids[$2] = 1 }
        /^drm-memory-vram:/ { if (!seen) v += $2 }
        /^drm-memory-gtt:/  { if (!seen) g += $2 }
        END { printf "%d %d\n", v / 1024, g / 1024 }')
    rss=$(awk '/^VmRSS:/ { printf "%d", $2 / 1024 }' "/proc/$pid/status")
    printf "%-8s %-7s %-16s %10s %10s %10s\n" "$pid" "${port:-?}" "${name:-?}" "${vram:-0}" "${gtt:-0}" "${rss:-0}"
done

echo
echo "Whole GPU (all processes):"
for d in /sys/class/drm/card*/device; do
    [[ -r "$d/mem_info_vram_used" ]] || continue
    awk -v d="$d" '
        FILENAME ~ /vram_used/ { vu = $1 } FILENAME ~ /vram_total/ { vt = $1 }
        FILENAME ~ /gtt_used/  { gu = $1 } FILENAME ~ /gtt_total/  { gt = $1 }
        END { printf "  %s\n    VRAM %d / %d MiB\n    GTT  %d / %d MiB\n", d, vu/1048576, vt/1048576, gu/1048576, gt/1048576 }
    ' "$d/mem_info_vram_used" "$d/mem_info_vram_total" "$d/mem_info_gtt_used" "$d/mem_info_gtt_total"
done
EOF
sudo chmod +x /usr/local/bin/llama-status

# ------------------------------------------------------------
# Firewall: allow RFC1918 LAN + Tailscale, block everyone else
#
# llama-server has no auth by default, so binding to 0.0.0.0 is only
# safe if the firewall limits who can reach it.
#   10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16  RFC1918 private LAN
#   100.64.0.0/10                               Tailscale (CGNAT range)
# Rules are scoped to the llama-server port range only and never change
# ufw's default policy or enable it, so SSH access can't be locked out.
# ------------------------------------------------------------

echo
echo "[8/10] Configuring firewall for LAN + Tailscale access..."

LLAMA_PORTS="10020:10039"
ALLOWED_CIDRS=(10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 100.64.0.0/10)

if command -v ufw >/dev/null 2>&1; then
    # Allow rules first: ufw matches top-down, so these must precede the deny.
    for cidr in "${ALLOWED_CIDRS[@]}"; do
        sudo ufw allow from "$cidr" to any port "$LLAMA_PORTS" proto tcp \
            comment "llama-server $cidr" >/dev/null
    done
    # Tailscale interface (also covers IPv6 tailnet addresses).
    sudo ufw allow in on tailscale0 to any port "$LLAMA_PORTS" proto tcp \
        comment "llama-server tailscale0" >/dev/null
    # Everyone else is denied on these ports.
    sudo ufw deny "$LLAMA_PORTS"/tcp comment "llama-server deny public" >/dev/null

    if sudo ufw status | grep -q "^Status: active"; then
        echo "ufw is active; rules applied for ports $LLAMA_PORTS."
    else
        echo "WARNING: ufw is installed but INACTIVE, so these rules are not enforced yet."
        echo "llama-server on 0.0.0.0 is reachable from any network this host is on."
        echo "Enable it once you've confirmed SSH is allowed:"
        echo "  sudo ufw allow OpenSSH && sudo ufw enable"
    fi
else
    echo "WARNING: ufw not found; no firewall rules were added."
    echo "llama-server on 0.0.0.0 is reachable from any network this host is on."
    echo "Install ufw (sudo apt-get install ufw) and re-run this script."
fi

# ------------------------------------------------------------
# Download Inkling 2 Mini (UD-IQ4_XS) + vision projector
#
# Uses the Mainline_Support files: same tensors as the repo-root GGUFs but
# in the inkling2_loop layout mainline llama.cpp reads (the root files need
# Unsloth's fork). They ship as .gguf_file and are renamed to .gguf.
# Downloads resume if interrupted. Set HF_TOKEN to use an HF token,
# or SKIP_INKLING=1 to skip this step.
# ------------------------------------------------------------

echo
echo "[9/10] Downloading Inkling 2 Mini (UD-IQ4_XS)..."

INKLING_REPO="https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main"
INKLING_MODEL="$MODEL_DIR/Inkling-2-Mini-UD-IQ4_XS.gguf"
INKLING_MMPROJ="$MODEL_DIR/mmproj/Inkling-2-Mini-mmproj-F16.gguf"

# hf_download <repo path> <destination>
hf_download() {
    local src="$1" dest="$2" part="$2.part"
    local auth=()
    [[ -n "${HF_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")

    if [[ -s "$dest" ]]; then
        echo "Already present: $dest"
        return 0
    fi
    echo "Downloading $src"
    echo "  -> $dest"
    curl -fL --retry 5 --retry-delay 5 -C - "${auth[@]}" -o "$part" "$INKLING_REPO/$src"
    mv "$part" "$dest"
}

if [[ "${SKIP_INKLING:-0}" == "1" ]]; then
    echo "Skipping (SKIP_INKLING=1)."
else
    mkdir -p "$MODEL_DIR/mmproj"

    if [[ ! -s "$INKLING_MODEL" ]]; then
        # ~15.7 GB model + ~0.6 GB mmproj; ask for 17 GB free.
        need_kb=$((17 * 1024 * 1024))
        free_kb=$(df -Pk "$MODEL_DIR" | awk 'NR==2 {print $4}')
        if (( free_kb < need_kb )); then
            echo "ERROR: need ~17 GB free in $MODEL_DIR, only $((free_kb / 1024 / 1024)) GB available."
            exit 1
        fi
    fi

    hf_download "Mainline_Support/Inkling-2-Mini-UD-IQ4_XS.gguf_file" "$INKLING_MODEL"
    hf_download "mmproj-F16.gguf" "$INKLING_MMPROJ"
fi

# ------------------------------------------------------------
# Finished
# ------------------------------------------------------------

echo
echo "[10/10] Installation complete!"
echo
echo "=============================================="
echo " llama.cpp is installed"
echo "=============================================="
echo
echo "Installation:"
echo "  $LLAMA_DIR"
echo
echo "Models:"
echo "  $MODEL_DIR"
echo
echo "llama.cpp version:"
llama --version || true
echo
echo "Vulkan devices:"
vulkaninfo --summary 2>/dev/null || true
echo
echo "----------------------------------------------"
echo "Put GGUF models in:"
echo "  $MODEL_DIR"
echo
echo "Then run:"
echo "  ~/run-llama.sh"
echo
echo "Or start the OpenAI-compatible server:"
echo "  ~/llama-server.sh"
echo
echo "Server (reachable from RFC1918 LAN + Tailscale):"
echo "  http://$(hostname -I | awk '{print $1}'):10020"
command -v tailscale >/dev/null 2>&1 && echo "  http://$(tailscale ip -4 2>/dev/null | head -1):10020  (Tailscale)" || true
echo
echo "----------------------------------------------"
echo "Examples: run a server on a port with a model alias"
echo "----------------------------------------------"
echo
echo "Qwen3.8-27B on port 10020, alias \"qwen-27b\":"
echo "  llama-server -m $MODEL_DIR/Qwen3.8-27B-UD-Q4_K_XL.gguf --alias qwen-27b --host 0.0.0.0 --port 10020 -ngl 999 -c 32768 --jinja"
echo
echo "Qwen3.5-35B-A3B on port 10021, alias \"qwen-35b-a3b\":"
echo "  llama-server -m $MODEL_DIR/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --alias qwen-35b-a3b --host 0.0.0.0 --port 10021 -ngl 999 -c 32768 --jinja"
echo
echo "Qwen3.5-9B on port 10022, alias \"qwen-9b\":"
echo "  llama-server -m $MODEL_DIR/Qwen3.5-9B-UD-Q4_K_XL.gguf --alias qwen-9b --host 0.0.0.0 --port 10022 -ngl 999 -c 32768 --jinja"
echo
echo "Both at once, on all interfaces (LAN/Tailscale), several aliases on one:"
echo "  llama-server -m $MODEL_DIR/Qwen3.8-27B-UD-Q4_K_XL.gguf --alias qwen-27b,primary --host 0.0.0.0 --port 10020 -ngl 999 -c 32768 --jinja &"
echo "  llama-server -m $MODEL_DIR/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --alias qwen-35b-a3b --host 0.0.0.0 --port 10021 -ngl 999 -c 32768 --jinja &"
echo
echo "Inkling 2 Mini on port 10023, alias \"inkling-mini\" (text + images):"
echo "  sudo systemctl start llama-inkling-mini"
echo "  # or manually:"
echo "  llama-server -m $MODEL_DIR/Inkling-2-Mini-UD-IQ4_XS.gguf --mmproj $MODEL_DIR/mmproj/Inkling-2-Mini-mmproj-F16.gguf --alias inkling-mini --host 0.0.0.0 --port 10023 -ngl 999 -c 32768 --jinja --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0"
echo
echo "Note: qwen-image-2.1-Q3_K_XL.gguf is an image-generation model. llama-server"
echo "cannot serve it; use stable-diffusion.cpp or ComfyUI instead."
echo
echo "Call it using the alias as the model name:"
echo "  curl http://127.0.0.1:10020/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
echo
echo "List the model names the server exposes:"
echo "  curl http://127.0.0.1:10020/v1/models"
echo "----------------------------------------------"
echo
