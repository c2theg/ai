#!/usr/bin/env bash
set -Eeuo pipefail

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


Version:  0.0.6
Last Updated:  10/25/2026



Usage:
    sudo ./install_llama_cpp_ubuntu24.04.sh

    Run it again at any time to update: GPU drivers/tools, CUDA, llama.cpp
    (rebuilt only when upstream changed) and the downloaded models.

# Start / stop / status (model servers run as systemd units llama@<name>):
    llama-start  qwen3-embed-4b      # start one (or: llama-start all)
    llama-stop   qwen3-embed-4b      # stop one  (or: llama-stop  -> stops ALL)
    llama-status                     # state, port, health, RAM/GPU memory per server
    llama-logs   qwen3-embed-4b      # follow logs
    llama-autostart qwen3-embed-4b on|off   # start at boot or not

# Model configs (one per server, edit then: llama-start <name> again):
    /etc/llama.cpp/models/<name>.conf
    /etc/llama.cpp/global.conf       # host / extra args applied to every server

# Environment (all optional):
    BACKEND=cpu|cuda|vulkan    Override GPU auto-detection
    MODELS=all|none|name,name  Skip the model questions (names listed below)
    <NAME>_FILE=file.gguf      Force a quant, e.g. QWEN38_27B_FILE=Qwen3.8-27B-UD-Q4_K_S.gguf
    HF_TOKEN=hf_...            Hugging Face token (only for rate limits / gated repos)
    FORCE_REBUILD=1            Rebuild llama.cpp even if it is up to date

"
# ============================================================
# llama.cpp installer - generic Ubuntu 24.04
#
# Detects the hardware and builds llama.cpp for it:
#   NVIDIA GPU         -> CUDA     (driver, CUDA toolkit, nvtop, nvidia-smi)
#   AMD / Intel GPU    -> Vulkan   (Mesa RADV/ANV, nvtop, radeontop / intel_gpu_top)
#   No usable GPU      -> CPU only
#   AMD XDNA / Intel NPU are detected and reported, but llama.cpp can't use them.
#
# Then offers to download + install + autostart:
#   qwen3-embed-4b     Qwen/Qwen3-Embedding-4B-GGUF          (embeddings, port 10030)
#   qwen36-35b-a3b     unsloth/Qwen3.6-35B-A3B-GGUF          (chat MoE,   port 10021)
#   nemotron-bolt-e4b  nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB (chat,     port 10022)
#   qwen38-27b         unsloth/Qwen3.8-27B-GGUF              (chat,       port 10020)
#   gemma4-26b-a4b     unsloth/gemma-4-26B-A4B-it-qat-GGUF   (chat MoE,   port 10023)
#
# Paths:
#   Source:   /opt/llama.cpp
#   Binaries: /usr/local/bin/llama-*
#   Models:   /opt/ai/models/*.gguf
#   Configs:  /etc/llama.cpp/
#   Runs as:  system user "llama"
# ============================================================

LLAMA_DIR="/opt/llama.cpp"
LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
MODEL_DIR="/opt/ai/models"
CONF_ROOT="/etc/llama.cpp"
CONF_DIR="$CONF_ROOT/models"
SERVICE_USER="llama"
SERVICE_HOME="/var/lib/llama"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

export DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

# ask_yn "question" Y|N  -> returns 0 for yes. Reads from the terminal even
# when the script itself is piped in; falls back to the default without one.
ask_yn() {
    local prompt="$1" default="$2" answer hint="[y/N]"
    [[ "$default" == "Y" ]] && hint="[Y/n]"
    if { : </dev/tty; } 2>/dev/null; then
        read -rp "$prompt $hint " answer </dev/tty || answer=""
    else
        answer=""
    fi
    answer="${answer:-$default}"
    [[ "$answer" =~ ^[Yy] ]]
}

# apt_optional pkg...  -- install packages that may not exist on every release
apt_optional() {
    local p
    for p in "$@"; do
        if apt-cache show "$p" >/dev/null 2>&1; then
            apt-get install -y "$p" || echo "WARNING: could not install $p"
        else
            echo "NOTE: package $p not available, skipping."
        fi
    done
}

# upgrade_installed regex  -- upgrade every installed package whose name matches
upgrade_installed() {
    local pkgs
    pkgs=$(dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' 2>/dev/null |
        awk -v re="$1" '$1 == "ii" && $2 ~ re { print $2 }')
    # shellcheck disable=SC2086
    [[ -n "$pkgs" ]] && apt-get install -y --only-upgrade $pkgs
    return 0
}

hf_curl() {
    if [[ -n "${HF_TOKEN:-}" ]]; then
        curl -H "Authorization: Bearer $HF_TOKEN" "$@"
    else
        curl "$@"
    fi
}

# wait_healthy <port> <seconds>  -- wait for llama-server to finish loading
wait_healthy() {
    local port="$1" timeout="$2" i
    for ((i = 0; i < timeout; i += 3)); do
        curl -fs -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
        sleep 3
    done
    return 1
}

echo
echo "=============================================="
echo " llama.cpp installer - Ubuntu 24.04"
echo " CPU / NVIDIA CUDA / AMD + Intel Vulkan"
echo "=============================================="
echo

# ------------------------------------------------------------
# Must run as root
# ------------------------------------------------------------

if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: Run this script with sudo:"
    echo
    echo "  sudo $0"
    echo
    exit 1
fi

# ------------------------------------------------------------
# System information
# ------------------------------------------------------------

echo "[1/10] System information"
echo "------------------------------------------------------------"

# shellcheck disable=SC1091
. /etc/os-release
echo "OS:      $PRETTY_NAME"
echo "Kernel:  $(uname -r)"
echo "Arch:    $(uname -m)"
echo "CPU:     $(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1) ($(nproc) threads)"

if [[ "${VERSION_ID:-}" != "24.04" ]]; then
    echo
    echo "WARNING: This script was written for Ubuntu 24.04 (found ${VERSION_ID:-unknown})."
    ask_yn "Continue anyway?" N || exit 1
fi

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64)  CUDA_REPO_ARCH="x86_64" ;;
    aarch64) CUDA_REPO_ARCH="sbsa" ;;
    *)       CUDA_REPO_ARCH="" ;;
esac

# ------------------------------------------------------------
# Base packages (installing again upgrades them to the latest)
# ------------------------------------------------------------

echo
echo "[2/10] Installing / updating base packages"
echo "------------------------------------------------------------"

apt-get update
apt-get install -y software-properties-common
add-apt-repository -y universe >/dev/null
apt-get update

apt-get install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    pkg-config \
    curl \
    wget \
    jq \
    pciutils \
    libssl-dev \
    libcurl4-openssl-dev \
    ca-certificates \
    gnupg \
    htop \
    btop \
    lm-sensors

# ------------------------------------------------------------
# Detect accelerators
# ------------------------------------------------------------

echo
echo "[3/10] Detecting GPUs / NPUs"
echo "------------------------------------------------------------"

# Display controllers: VGA [0300], 3D [0302], Display [0380]
GPU_LINES=$(lspci -nn | grep -E '\[03(00|02|80)\]' || true)
HAS_NVIDIA=0; HAS_AMD_GPU=0; HAS_INTEL_GPU=0
grep -q '\[10de:' <<<"$GPU_LINES" && HAS_NVIDIA=1
grep -q '\[1002:' <<<"$GPU_LINES" && HAS_AMD_GPU=1
grep -q '\[8086:' <<<"$GPU_LINES" && HAS_INTEL_GPU=1

# NPUs: AMD XDNA (Ryzen AI) and Intel NPU (Core Ultra)
NPU_LINES=$(lspci -nn | grep -Ei '\[1022:(1502|17f0)\]|\[8086:(7d1d|643e|ad1d|b03e|fd3e)\]' || true)

echo "${GPU_LINES:-No GPUs found.}"
[[ -n "$NPU_LINES" ]] && echo "$NPU_LINES"
echo

# NVIDIA wins when present: building Vulkan as well would make llama.cpp
# split models across the dGPU and a (much slower) iGPU.
if [[ -n "${BACKEND:-}" ]]; then
    echo "BACKEND=$BACKEND set in the environment."
elif [[ "$HAS_NVIDIA" == "1" ]]; then
    BACKEND="cuda"
elif [[ "$HAS_AMD_GPU" == "1" || "$HAS_INTEL_GPU" == "1" ]]; then
    BACKEND="vulkan"
else
    BACKEND="cpu"
fi

case "$BACKEND" in
    cuda|vulkan|cpu) ;;
    *) echo "ERROR: BACKEND must be cpu, cuda or vulkan (got '$BACKEND')."; exit 1 ;;
esac

echo "Backend: $BACKEND"
if [[ "$BACKEND" == "vulkan" && "$HAS_INTEL_GPU" == "1" && "$HAS_AMD_GPU" == "0" ]]; then
    echo "NOTE: Only an Intel GPU was found. Small integrated GPUs can be slower"
    echo "      than the CPU; re-run with BACKEND=cpu if that's the case here."
fi
if [[ -n "$NPU_LINES" ]]; then
    echo "NOTE: An NPU was detected. llama.cpp does not use NPUs; models run on"
    echo "      the $BACKEND backend."
fi

# ------------------------------------------------------------
# Platform packages
# ------------------------------------------------------------

echo
echo "[4/10] Installing / updating $BACKEND platform packages"
echo "------------------------------------------------------------"

if [[ "$BACKEND" == "cuda" ]]; then

    # Driver: install the recommended one the first time, then keep the
    # installed branch up to date on later runs.
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
        echo "No working NVIDIA driver found. Installing the recommended driver..."
        apt-get install -y ubuntu-drivers-common
        ubuntu-drivers devices || true
        ubuntu-drivers install
        echo
        echo "=============================================="
        echo " NVIDIA driver installed. Reboot, then run this"
        echo " script again to finish:"
        echo
        echo "   sudo reboot"
        echo "   sudo $0"
        echo "=============================================="
        exit 0
    fi
    upgrade_installed '^(nvidia-|libnvidia-)'

    # CUDA toolkit from NVIDIA's repo (toolkit only, never the "cuda"
    # meta-package, which would also replace the driver).
    if [[ -z "$CUDA_REPO_ARCH" ]]; then
        echo "ERROR: No CUDA repository for architecture $ARCH."
        exit 1
    fi
    if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
        CUDA_KEYRING="$(mktemp --suffix=.deb)"
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${CUDA_REPO_ARCH}/cuda-keyring_1.1-1_all.deb" \
            -O "$CUDA_KEYRING"
        dpkg -i "$CUDA_KEYRING"
        rm -f "$CUDA_KEYRING"
        apt-get update
    fi
    apt-get install -y cuda-toolkit nvtop
    upgrade_installed '^(cuda-|libcublas|libcudnn|libnccl)'

    CUDA_HOME="/usr/local/cuda"
    cat >/etc/profile.d/cuda.sh <<EOF
export CUDA_HOME="$CUDA_HOME"
export PATH="\${CUDA_HOME}/bin:\${PATH}"
EOF
    chmod 644 /etc/profile.d/cuda.sh
    export PATH="$CUDA_HOME/bin:$PATH"
    ldconfig

    # Keep the driver loaded between requests (faster model starts).
    systemctl enable --now nvidia-persistenced >/dev/null 2>&1 || true

    echo
    nvidia-smi
    nvcc --version | tail -2

elif [[ "$BACKEND" == "vulkan" ]]; then

    apt-get install -y \
        libvulkan-dev \
        vulkan-tools \
        mesa-vulkan-drivers \
        glslc \
        spirv-headers \
        nvtop
    [[ "$HAS_AMD_GPU" == "1" ]] && apt_optional radeontop rocm-smi
    [[ "$HAS_INTEL_GPU" == "1" ]] && apt_optional intel-gpu-tools
    upgrade_installed '^(mesa-|libvulkan|libdrm|linux-firmware)'

    # Newer AMD/Intel GPUs (e.g. Strix Halo, Arc B-series) need a newer kernel
    # than 24.04's original 6.8. The HWE kernel tracks newer releases.
    KERNEL_MAJOR_MINOR=$(uname -r | awk -F. '{ printf "%d%02d", $1, $2 }')
    if dpkg -s linux-generic-hwe-24.04 >/dev/null 2>&1; then
        apt-get install -y linux-generic-hwe-24.04
    elif (( KERNEL_MAJOR_MINOR < 611 )); then
        echo
        echo "Running kernel $(uname -r). Recent AMD/Intel GPUs work best with the"
        echo "24.04 HWE kernel (needs a reboot)."
        if ask_yn "Install linux-generic-hwe-24.04?" Y; then
            apt-get install -y linux-generic-hwe-24.04
        fi
    fi

    echo
    echo "Vulkan devices:"
    vulkaninfo --summary 2>/dev/null | sed -n '/^Devices:/,$p' || {
        echo "WARNING: vulkaninfo returned an error. The GPU driver may need attention"
        echo "         (a reboot after a kernel/firmware update often fixes this)."
    }

else

    echo "CPU only - nothing extra to install."

fi

# ------------------------------------------------------------
# Service user + directories
# ------------------------------------------------------------

echo
echo "[5/10] Creating service user and directories"
echo "------------------------------------------------------------"

if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    useradd --system --home-dir "$SERVICE_HOME" --create-home \
        --shell /usr/sbin/nologin "$SERVICE_USER"
fi
# /dev/dri and /dev/kfd access for AMD/Intel GPUs
for g in video render; do
    getent group "$g" >/dev/null && usermod -aG "$g" "$SERVICE_USER"
done

mkdir -p "$MODEL_DIR" "$CONF_DIR" "$SERVICE_HOME"
chown "$SERVICE_USER:$SERVICE_USER" "$SERVICE_HOME"
chmod 755 "$MODEL_DIR"

# ------------------------------------------------------------
# Clone / update llama.cpp
# ------------------------------------------------------------

echo
echo "[6/10] Getting latest llama.cpp"
echo "------------------------------------------------------------"

if [[ -d "$LLAMA_DIR/.git" ]]; then
    git -C "$LLAMA_DIR" fetch --depth=1 origin master
    git -C "$LLAMA_DIR" reset --hard origin/master
else
    git clone --depth=1 "$LLAMA_REPO" "$LLAMA_DIR"
fi
LLAMA_COMMIT="$(git -C "$LLAMA_DIR" rev-parse --short HEAD)"
echo "llama.cpp commit: $LLAMA_COMMIT"

# ------------------------------------------------------------
# Build llama.cpp (only when the source or backend changed)
# ------------------------------------------------------------

echo
echo "[7/10] Building llama.cpp ($BACKEND)"
echo "------------------------------------------------------------"

# Static binaries so a rebuild can't break servers that are running.
CMAKE_ARGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=OFF
    -DLLAMA_BUILD_TESTS=OFF
)

# llama.cpp's AVX code assumes AVX2 + FMA + F16C come with it. Older x86 CPUs
# that have AVX but not the rest (Sandy/Ivy Bridge Xeons, early AMD FX) fail
# to compile with -march=native, so build those for SSE 4.2 instead.
CPU_FLAGS=" $(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2) "
if [[ "$ARCH" == "x86_64" && "$CPU_FLAGS" == *" avx "* ]] &&
   [[ "$CPU_FLAGS" != *" avx2 "* || "$CPU_FLAGS" != *" fma "* || "$CPU_FLAGS" != *" f16c "* ]]; then
    echo "CPU has AVX without AVX2/FMA/F16C - building for SSE 4.2 (llama.cpp's AVX-only path doesn't compile)."
    CMAKE_ARGS+=(-DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF
                 -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF)
else
    CMAKE_ARGS+=(-DGGML_NATIVE=ON)
fi

case "$BACKEND" in
    cuda)   CMAKE_ARGS+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
                         -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc") ;;
    vulkan) CMAKE_ARGS+=(-DGGML_VULKAN=ON) ;;
esac

STAMP_FILE="$LLAMA_DIR/.install-stamp"
NEW_STAMP="$LLAMA_COMMIT ${CMAKE_ARGS[*]}"
OLD_STAMP="$(cat "$STAMP_FILE" 2>/dev/null || true)"
REBUILT=0

if [[ "$FORCE_REBUILD" == "1" || "$NEW_STAMP" != "$OLD_STAMP" || ! -x /usr/local/bin/llama-server ]]; then
    # Different backend/flags -> start from a clean build directory.
    if [[ "${OLD_STAMP#* }" != "${CMAKE_ARGS[*]}" ]]; then
        rm -rf "$LLAMA_DIR/build"
    fi

    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" "${CMAKE_ARGS[@]}"
    cmake --build "$LLAMA_DIR/build" --parallel "$(nproc)"

    # Copy then rename, so binaries in use are replaced, not overwritten.
    for f in "$LLAMA_DIR"/build/bin/llama-*; do
        [[ -f "$f" && -x "$f" ]] || continue
        name="$(basename "$f")"
        install -m 755 "$f" "/usr/local/bin/.$name.new"
        mv -f "/usr/local/bin/.$name.new" "/usr/local/bin/$name"
    done
    ln -sf /usr/local/bin/llama-cli /usr/local/bin/llama

    echo "$NEW_STAMP" >"$STAMP_FILE"
    REBUILT=1
else
    echo "Already up to date ($LLAMA_COMMIT) - skipping build (FORCE_REBUILD=1 to force)."
fi

echo
llama-server --version 2>&1 | tail -n 3 || true

# ------------------------------------------------------------
# systemd template unit + start/stop/status commands
# ------------------------------------------------------------

echo
echo "[8/10] Installing systemd unit and llama-* commands"
echo "------------------------------------------------------------"

if [[ ! -f "$CONF_ROOT/global.conf" ]]; then
    cat >"$CONF_ROOT/global.conf" <<'EOF'
# Applied to every llama@<name> server.
# LLAMA_HOST   interface to listen on (0.0.0.0 = all; firewall limits it to LAN + Tailscale)
# GLOBAL_ARGS  extra llama-server arguments, e.g. require an API key:
#              GLOBAL_ARGS="--api-key-file /etc/llama.cpp/api-keys"
LLAMA_HOST=0.0.0.0
GLOBAL_ARGS=""
EOF
fi

cat >/etc/systemd/system/llama@.service <<EOF
[Unit]
Description=llama.cpp server (%i)
After=network-online.target
Wants=network-online.target
ConditionPathExists=$CONF_DIR/%i.conf

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
Environment=HOME=$SERVICE_HOME
Environment=LLAMA_HOST=0.0.0.0
EnvironmentFile=-$CONF_ROOT/global.conf
EnvironmentFile=$CONF_DIR/%i.conf
ExecStart=/usr/local/bin/llama-server -m \${MODEL} --alias \${ALIAS} --host \${LLAMA_HOST} --port \${PORT} \$ARGS \$GLOBAL_ARGS
Restart=on-failure
RestartSec=10
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload

# llama-start <name>|all
cat >/usr/local/bin/llama-start <<'EOF'
#!/usr/bin/env bash
# Usage: llama-start <name> [name...]   start model server(s)
#        llama-start all                 start every configured model
# Configs: /etc/llama.cpp/models/<name>.conf  (unit: llama@<name>.service)
set -uo pipefail
CONF_DIR=/etc/llama.cpp/models
SUDO=""; [[ $EUID -ne 0 ]] && SUDO=sudo

configured() { for f in "$CONF_DIR"/*.conf; do [[ -e "$f" ]] && basename "$f" .conf; done; }

if [[ $# -eq 0 ]]; then
    echo "Usage: llama-start <name>|all"
    echo "Configured models:"
    configured | sed 's/^/  /'
    exit 1
fi

names=("$@")
[[ "$1" == "all" ]] && mapfile -t names < <(configured)

for n in "${names[@]}"; do
    conf="$CONF_DIR/$n.conf"
    if [[ ! -f "$conf" ]]; then
        echo "No config for '$n' ($conf)"
        continue
    fi
    port=$(sed -n 's/^PORT=//p' "$conf" | tr -d '"')
    # restart so edited configs take effect
    $SUDO systemctl restart "llama@$n.service" &&
        echo "Started llama@$n  ->  http://127.0.0.1:$port   (loading: llama-status / llama-logs $n)"
done
EOF

# llama-stop [name|all]
cat >/usr/local/bin/llama-stop <<'EOF'
#!/usr/bin/env bash
# Usage: llama-stop          stop every llama-server (systemd units + manual)
#        llama-stop <name>   stop only that model server
set -uo pipefail
SUDO=""; [[ $EUID -ne 0 ]] && SUDO=sudo

if [[ $# -eq 0 || "$1" == "all" ]]; then
    units=$(systemctl list-units 'llama@*.service' --no-legend --plain | awk '{print $1}')
    # shellcheck disable=SC2086
    [[ -n "$units" ]] && $SUDO systemctl stop $units && echo "Stopped: $units"
    $SUDO pkill -x llama-server && echo "Stopped manually started llama-server processes."
else
    for n in "$@"; do
        $SUDO systemctl stop "llama@$n.service" 2>/dev/null && echo "Stopped llama@$n"
        $SUDO pkill -f -- "--alias $n( |,|$)" && echo "Stopped manual instance with alias $n."
    done
fi
exit 0
EOF

# llama-autostart <name> on|off
cat >/usr/local/bin/llama-autostart <<'EOF'
#!/usr/bin/env bash
# Usage: llama-autostart <name> on|off   start (or not) at boot
set -uo pipefail
SUDO=""; [[ $EUID -ne 0 ]] && SUDO=sudo
if [[ $# -ne 2 ]]; then echo "Usage: llama-autostart <name> on|off"; exit 1; fi
case "$2" in
    on)  $SUDO systemctl enable  "llama@$1.service" ;;
    off) $SUDO systemctl disable "llama@$1.service" ;;
    *)   echo "Usage: llama-autostart <name> on|off"; exit 1 ;;
esac
EOF

# llama-logs <name>
cat >/usr/local/bin/llama-logs <<'EOF'
#!/usr/bin/env bash
# Usage: llama-logs <name>   follow a model server's log
if [[ $# -ne 1 ]]; then echo "Usage: llama-logs <name>"; exit 1; fi
exec journalctl -u "llama@$1.service" -n 100 -f
EOF

# llama-status
cat >/usr/local/bin/llama-status <<'EOF'
#!/usr/bin/env bash
# State, port, health and memory of every llama-server.
#   RSS = process RAM. GPU = NVIDIA memory, or AMD VRAM + GTT (GPU-visible
#   system RAM, where most of the model lives on an APU). AMD/Intel per-process
#   GPU memory needs root (sudo llama-status).
set -uo pipefail
CONF_DIR=/etc/llama.cpp/models

declare -A NV
if command -v nvidia-smi >/dev/null 2>&1; then
    while IFS=', ' read -r pid mem; do
        [[ "$pid" =~ ^[0-9]+$ ]] && NV[$pid]=$(( ${NV[$pid]:-0} + mem ))
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
fi

gpu_mib() {
    local pid="$1"
    if [[ -n "${NV[$pid]:-}" ]]; then echo "${NV[$pid]}"; return; fi
    cat /proc/"$pid"/fdinfo/* 2>/dev/null | awk '
        /^drm-client-id:/   { seen = ($2 in ids); ids[$2] = 1 }
        /^drm-memory-(vram|gtt):/ { if (!seen) t += $2 }
        END { printf "%d\n", t / 1024 }'
}

rss_mib() { awk '/^VmRSS:/ { printf "%d", $2 / 1024 }' "/proc/$1/status" 2>/dev/null; }

health() {
    local body
    body=$(curl -s -m 2 "http://127.0.0.1:$1/health" 2>/dev/null) || { echo "-"; return; }
    case "$body" in
        *'"ok"'*)  echo "ok" ;;
        *oading*)  echo "loading" ;;
        *)         echo "error" ;;
    esac
}

fmt="%-20s %-10s %-9s %-6s %-8s %-8s %9s %9s\n"
printf "$fmt" NAME STATE AUTOSTART PORT HEALTH PID "RSS MiB" "GPU MiB"
declare -A SHOWN
for conf in "$CONF_DIR"/*.conf; do
    [[ -e "$conf" ]] || continue
    n=$(basename "$conf" .conf)
    port=$(sed -n 's/^PORT=//p' "$conf" | tr -d '"')
    state=$(systemctl is-active "llama@$n.service" 2>/dev/null)
    boot=$(systemctl is-enabled "llama@$n.service" 2>/dev/null)
    pid=$(systemctl show -p MainPID --value "llama@$n.service" 2>/dev/null)
    if [[ "$pid" =~ ^[1-9][0-9]*$ ]]; then
        SHOWN[$pid]=1
        printf "$fmt" "$n" "$state" "${boot:-?}" "$port" "$(health "$port")" "$pid" "$(rss_mib "$pid")" "$(gpu_mib "$pid")"
    else
        printf "$fmt" "$n" "$state" "${boot:-?}" "$port" "-" "-" "-" "-"
    fi
done

# llama-server processes started by hand (not through systemd)
for pid in $(pgrep -x llama-server); do
    [[ -n "${SHOWN[$pid]:-}" ]] && continue
    args=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    port=$(sed -nE 's/.*--port ([0-9]+).*/\1/p' <<<"$args")
    name=$(sed -nE 's/.*(--alias|-a) ([^ ]+).*/\2/p' <<<"$args")
    printf "$fmt" "${name:-?} (manual)" "active" "-" "${port:-8080}" "$(health "${port:-8080}")" "$pid" "$(rss_mib "$pid")" "$(gpu_mib "$pid")"
done

echo
echo "System memory:"
free -m | awk 'NR == 2 { printf "  RAM  %d / %d MiB used\n", $3, $2 }'

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "NVIDIA GPUs:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader 2>/dev/null | sed 's/^/  /'
fi

for d in /sys/class/drm/card*/device; do
    [[ -r "$d/mem_info_vram_used" ]] || continue
    echo
    echo "AMD GPU $(basename "$(readlink -f "$d")"):"
    awk '
        FILENAME ~ /vram_used/ { vu = $1 } FILENAME ~ /vram_total/ { vt = $1 }
        FILENAME ~ /gtt_used/  { gu = $1 } FILENAME ~ /gtt_total/  { gt = $1 }
        END { printf "  VRAM %d / %d MiB\n  GTT  %d / %d MiB\n", vu/1048576, vt/1048576, gu/1048576, gt/1048576 }
    ' "$d/mem_info_vram_used" "$d/mem_info_vram_total" "$d/mem_info_gtt_used" "$d/mem_info_gtt_total"
done
EOF

chmod 755 /usr/local/bin/llama-start /usr/local/bin/llama-stop /usr/local/bin/llama-status \
          /usr/local/bin/llama-logs /usr/local/bin/llama-autostart

echo "Installed: llama-start  llama-stop  llama-status  llama-logs  llama-autostart"

# ------------------------------------------------------------
# Models: download + install + autostart
# ------------------------------------------------------------

echo
echo "[9/10] Models"
echo "------------------------------------------------------------"

MODEL_KEYS=(qwen3-embed-4b qwen36-35b-a3b nemotron-bolt-e4b qwen38-27b gemma4-26b-a4b)
declare -A M_REPO M_DESC M_PORT M_ARGS M_CANDS M_OVERHEAD M_DEFAULT

M_REPO[qwen3-embed-4b]="Qwen/Qwen3-Embedding-4B-GGUF"
M_DESC[qwen3-embed-4b]="Qwen3-Embedding-4B - embeddings for RAG"
M_PORT[qwen3-embed-4b]=10030
M_ARGS[qwen3-embed-4b]="--embeddings --pooling last -c 8192 -b 8192 -ub 8192"
M_OVERHEAD[qwen3-embed-4b]=1024
M_DEFAULT[qwen3-embed-4b]=Y

M_REPO[qwen36-35b-a3b]="unsloth/Qwen3.6-35B-A3B-GGUF"
M_DESC[qwen36-35b-a3b]="Qwen3.6-35B-A3B - MoE chat/agent model, fast for its size"
M_PORT[qwen36-35b-a3b]=10021
M_ARGS[qwen36-35b-a3b]="-c 32768 --jinja"
M_OVERHEAD[qwen36-35b-a3b]=3072
M_DEFAULT[qwen36-35b-a3b]=N

M_REPO[nemotron-bolt-e4b]="nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB"
M_DESC[nemotron-bolt-e4b]="Nemotron-3.5-Bolt-E4B - small, fast chat model"
M_PORT[nemotron-bolt-e4b]=10022
M_ARGS[nemotron-bolt-e4b]="-c 32768 --jinja"
M_OVERHEAD[nemotron-bolt-e4b]=2048
M_DEFAULT[nemotron-bolt-e4b]=N

M_REPO[qwen38-27b]="unsloth/Qwen3.8-27B-GGUF"
M_DESC[qwen38-27b]="Qwen3.8-27B - dense chat/agent model, best quality, slowest"
M_PORT[qwen38-27b]=10020
M_ARGS[qwen38-27b]="-c 32768 --jinja"
M_OVERHEAD[qwen38-27b]=4096
M_DEFAULT[qwen38-27b]=N

M_REPO[gemma4-26b-a4b]="unsloth/gemma-4-26B-A4B-it-qat-GGUF"
M_DESC[gemma4-26b-a4b]="Gemma-4-26B-A4B (QAT) - MoE chat model from Google"
M_PORT[gemma4-26b-a4b]=10023
M_ARGS[gemma4-26b-a4b]="-c 32768 --jinja"
M_OVERHEAD[gemma4-26b-a4b]=3072
M_DEFAULT[gemma4-26b-a4b]=N

# Candidate files, best first: "file:bytes". The first one that fits in the
# remaining memory is used. Override with <NAME>_FILE (see header).
M_CANDS[qwen3-embed-4b]="Qwen3-Embedding-4B-Q8_0.gguf:4279660224 Qwen3-Embedding-4B-Q4_K_M.gguf:2496703776"
M_CANDS[qwen36-35b-a3b]="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf:22360456160 Qwen3.6-35B-A3B-UD-IQ4_XS.gguf:17730509792 Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf:16845511648 Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf:13211155424"
M_CANDS[nemotron-bolt-e4b]="Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf:6180446208"
M_CANDS[qwen38-27b]="Qwen3.8-27B-UD-Q4_K_XL.gguf:17559178144 Qwen3.8-27B-UD-Q4_K_S.gguf:15358213024 Qwen3.8-27B-UD-IQ4_XS.gguf:14252845984 Qwen3.8-27B-UD-Q3_K_XL.gguf:13146393504"
M_CANDS[gemma4-26b-a4b]="gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf:14249047104"

# Memory available to models: system RAM + dedicated GPU memory
# (on an AMD APU the VRAM carve-out is not part of MemTotal), minus 4 GiB for the OS.
RAM_MIB=$(awk '/^MemTotal:/ { printf "%d", $2 / 1024 }' /proc/meminfo)
VRAM_MIB=0
if [[ "$BACKEND" == "cuda" ]]; then
    VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{ s += $1 } END { printf "%d", s }')
elif [[ "$BACKEND" == "vulkan" ]]; then
    VRAM_MIB=$(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | awk '{ s += $1 } END { printf "%d", s / 1048576 }')
fi
REMAINING_MIB=$(( RAM_MIB + VRAM_MIB - 4096 ))
echo "Memory for models: ${RAM_MIB} MiB RAM + ${VRAM_MIB} MiB GPU (minus 4 GiB for the OS)"
echo

# pick_file <key>  -> prints the chosen file name and its size in MiB
pick_file() {
    local key="$1" var cand file bytes mib need smallest=""
    var="$(tr 'a-z-' 'A-Z_' <<<"$key")_FILE"
    if [[ -n "${!var:-}" ]]; then
        echo "${!var} 0"
        return
    fi
    for cand in ${M_CANDS[$key]}; do
        file="${cand%%:*}"; bytes="${cand##*:}"
        mib=$(( bytes / 1048576 ))
        need=$(( mib + ${M_OVERHEAD[$key]} ))
        if (( need <= REMAINING_MIB )); then
            echo "$file $mib"
            return
        fi
        smallest="$file $mib"
    done
    echo "$smallest"
}

# download_model <repo> <file>  -- returns 0 if the file is new or changed
download_model() {
    local repo="$1" file="$2"
    local url="https://huggingface.co/$repo/resolve/main/$file"
    local dest="$MODEL_DIR/$file" remote_sha local_sha

    # HF publishes the LFS SHA-256 in x-linked-etag on the (unfollowed) redirect.
    remote_sha=$(hf_curl -sI "$url" | awk 'tolower($1) == "x-linked-etag:" { gsub(/[\r"]/, "", $2); print $2 }' | tail -1)

    if [[ -f "$dest" ]]; then
        if [[ -z "$remote_sha" ]]; then
            echo "  Can't reach Hugging Face; keeping existing $file."
            return 1
        fi
        if [[ ! -f "$dest.sha256" ]]; then
            echo "  Hashing existing $file (one time)..."
            sha256sum "$dest" | awk '{ print $1 }' >"$dest.sha256"
        fi
        if [[ "$(cat "$dest.sha256")" == "$remote_sha" ]]; then
            echo "  $file is up to date."
            return 1
        fi
        echo "  A newer $file is on Hugging Face - downloading."
    fi

    echo "  Downloading $url"
    # .part is kept on failure so the next run resumes where this one stopped.
    if ! hf_curl -fL -C - --retry 10 --retry-delay 5 --retry-all-errors \
        -o "$dest.part" "$url"; then
        echo "  ERROR: download of $file failed."
        return 2
    fi

    if [[ -n "$remote_sha" ]]; then
        echo "  Verifying SHA-256..."
        local_sha=$(sha256sum "$dest.part" | awk '{ print $1 }')
        if [[ "$local_sha" != "$remote_sha" ]]; then
            echo "  ERROR: checksum mismatch for $file; removing the partial download."
            rm -f "$dest.part"
            return 2
        fi
        echo "$remote_sha" >"$dest.sha256"
    fi
    # rename over the old file: running servers keep the old copy until restarted
    mv -f "$dest.part" "$dest"
    chmod 644 "$dest"
    return 0
}

if [[ -n "${MODELS:-}" ]]; then
    echo "MODELS=$MODELS set in the environment - not asking."
fi

SELECTED=()
for key in "${MODEL_KEYS[@]}"; do
    read -r file mib < <(pick_file "$key")
    default="${M_DEFAULT[$key]}"
    [[ -f "$CONF_DIR/$key.conf" ]] && default=Y

    echo "$key  (port ${M_PORT[$key]})"
    echo "  ${M_DESC[$key]}"
    echo "  https://huggingface.co/${M_REPO[$key]}"
    if (( mib > 0 )); then
        echo "  File: $file  (~$(( mib / 1024 )) GiB)"
        if (( mib + ${M_OVERHEAD[$key]} > REMAINING_MIB )); then
            echo "  WARNING: this will not fit in the memory left (~$(( REMAINING_MIB / 1024 )) GiB)."
            default=N
        fi
    else
        echo "  File: $file  (from the environment)"
    fi

    if [[ -n "${MODELS:-}" ]]; then
        want=1
        case ",$MODELS," in
            *,all,*|*",$key,"*) ;;
            *) want=0 ;;
        esac
    else
        want=0
        ask_yn "  Download, install and autostart $key?" "$default" && want=1
    fi
    echo

    if [[ "$want" == "1" ]]; then
        SELECTED+=("$key:$file")
        REMAINING_MIB=$(( REMAINING_MIB - mib - ${M_OVERHEAD[$key]} ))
    fi
done

STARTED=()
for entry in "${SELECTED[@]}"; do
    key="${entry%%:*}"; file="${entry#*:}"
    echo "== $key"

    changed=0
    rc=0
    download_model "${M_REPO[$key]}" "$file" || rc=$?
    if (( rc == 2 )); then
        echo "  Skipping $key (download failed). Re-run the script to retry."
        continue
    fi
    (( rc == 0 )) && changed=1
    if [[ ! -f "$MODEL_DIR/$file" ]]; then
        echo "  Skipping $key ($MODEL_DIR/$file missing)."
        continue
    fi

    conf="$CONF_DIR/$key.conf"
    new_conf="# llama.cpp server config, used by llama@$key.service
# Edit, then apply with: llama-start $key
MODEL=$MODEL_DIR/$file
ALIAS=$key
PORT=${M_PORT[$key]}
ARGS=\"${M_ARGS[$key]}\""
    # Keep hand edits: only rewrite when the model file changed.
    if [[ ! -f "$conf" ]] || ! grep -qx "MODEL=$MODEL_DIR/$file" "$conf"; then
        echo "$new_conf" >"$conf"
        changed=1
    fi

    # Load chat models after the embedding server, so llama.cpp's
    # automatic GPU fitting sees the memory the embedder already uses.
    if [[ "$key" != "qwen3-embed-4b" ]]; then
        mkdir -p "/etc/systemd/system/llama@$key.service.d"
        cat >"/etc/systemd/system/llama@$key.service.d/order.conf" <<'EOF'
[Unit]
After=llama@qwen3-embed-4b.service
EOF
    fi
    systemctl daemon-reload
    systemctl enable "llama@$key.service" >/dev/null 2>&1

    if [[ "$changed" == "1" || "$REBUILT" == "1" ]] || ! systemctl is-active --quiet "llama@$key.service"; then
        echo "  Starting llama@$key ..."
        systemctl restart "llama@$key.service"
        if wait_healthy "${M_PORT[$key]}" 600; then
            echo "  Ready on port ${M_PORT[$key]}."
        else
            echo "  WARNING: not healthy yet - check: llama-logs $key"
        fi
    else
        echo "  Already running and up to date."
    fi
    STARTED+=("$key")
    echo
done

# Restart other running servers so they pick up a rebuilt llama-server.
if [[ "$REBUILT" == "1" ]]; then
    for unit in $(systemctl list-units 'llama@*.service' --state=active --no-legend --plain | awk '{print $1}'); do
        key="${unit#llama@}"; key="${key%.service}"
        [[ " ${STARTED[*]} " == *" $key "* ]] && continue
        echo "Restarting $unit (llama.cpp was updated)..."
        systemctl restart "$unit"
    done
fi

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
echo "[10/10] Configuring firewall for LAN + Tailscale access"
echo "------------------------------------------------------------"

LLAMA_PORTS="10020:10039"
ALLOWED_CIDRS=(10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 100.64.0.0/10)

if command -v ufw >/dev/null 2>&1; then
    # Allow rules first: ufw matches top-down, so these must precede the deny.
    for cidr in "${ALLOWED_CIDRS[@]}"; do
        ufw allow from "$cidr" to any port "$LLAMA_PORTS" proto tcp \
            comment "llama-server $cidr" >/dev/null
    done
    # Tailscale interface (also covers IPv6 tailnet addresses).
    ufw allow in on tailscale0 to any port "$LLAMA_PORTS" proto tcp \
        comment "llama-server tailscale0" >/dev/null
    # Everyone else is denied on these ports.
    ufw deny "$LLAMA_PORTS"/tcp comment "llama-server deny public" >/dev/null

    if ufw status | grep -q "^Status: active"; then
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
# Finished
# ------------------------------------------------------------

HOST_IP="$(hostname -I | awk '{print $1}')"

echo
echo "=============================================="
echo " llama.cpp is installed ($BACKEND, $LLAMA_COMMIT)"
echo "=============================================="
echo
echo "Source:   $LLAMA_DIR"
echo "Models:   $MODEL_DIR"
echo "Configs:  $CONF_DIR"
echo
llama-status || true
echo
echo "----------------------------------------------"
echo "Commands"
echo "----------------------------------------------"
echo "  llama-start <name>|all        start / restart (applies config edits)"
echo "  llama-stop  [name]            stop one, or all when no name is given"
echo "  llama-status                  state, health, memory"
echo "  llama-logs  <name>            follow the log"
echo "  llama-autostart <name> on|off start at boot or not"
if [[ "$BACKEND" == "cuda" ]]; then
    echo "  nvidia-smi / nvtop            GPU usage"
elif [[ "$BACKEND" == "vulkan" ]]; then
    echo "  nvtop / radeontop / intel_gpu_top   GPU usage"
fi
echo
echo "Add your own model: copy a .conf in $CONF_DIR, change MODEL/ALIAS/PORT"
echo "(use 10020-10039 for LAN access), then: llama-start <name>"
echo
echo "----------------------------------------------"
echo "API (OpenAI-compatible; model name = alias)"
echo "----------------------------------------------"
echo "Embeddings:"
echo "  curl http://$HOST_IP:10030/v1/embeddings \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen3-embed-4b\",\"input\":\"Hello world\"}'"
echo "  (Qwen3-Embedding: prefix queries, not documents, with"
echo "   \"Instruct: <task>\\nQuery: <text>\" for best retrieval.)"
echo
echo "Chat:"
echo "  curl http://$HOST_IP:10020/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen38-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
echo "  Turn thinking off per request: \"chat_template_kwargs\":{\"enable_thinking\":false}"
echo
echo "List what a server exposes:"
echo "  curl http://$HOST_IP:10030/v1/models"
command -v tailscale >/dev/null 2>&1 && echo "Tailscale IP: $(tailscale ip -4 2>/dev/null | head -1)" || true

if [[ -f /var/run/reboot-required ]]; then
    echo
    echo "*** A reboot is required to finish driver/kernel updates: sudo reboot ***"
fi
echo
