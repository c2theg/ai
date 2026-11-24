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


Version:  0.0.9
Last Updated:  11/24/2026

Supported:  Ubuntu 24.04 LTS and Ubuntu 26.04 LTS

Install (-4 = IPv4, so a host with broken IPv6 doesn't hang):
    wget -4 https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_llama_cpp_ubuntu24.04.sh && chmod +x install_llama_cpp_ubuntu24.04.sh

Usage:
    sudo ./install_llama_cpp_ubuntu24.04.sh

    Run it again at any time to update: GPU drivers/tools, CUDA, llama.cpp
    (rebuilt only when upstream changed) and the downloaded models.
    Installed models are updated in place (running ones are restarted, stopped
    ones stay stopped) and you're asked whether each should autostart at boot;
    only models that aren't installed yet are offered for download.

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
    MODELS=all|none|name,name  Which NEW models to install, without asking (names
                               listed below). Installed models are always checked
                               for updates and keep their autostart setting.
    <NAME>_FILE=file.gguf      Force a quant, e.g. QWEN38_27B_FILE=Qwen3.8-27B-UD-Q4_K_S.gguf
    HF_TOKEN=hf_...            Hugging Face token (only for rate limits / gated repos)
    FORCE_REBUILD=1            Rebuild llama.cpp even if it is up to date
    CUDA_REPO=ubuntu2404       Use another NVIDIA CUDA repo than this release's
                               (e.g. when NVIDIA has no repo for it yet)
    LUNARG_SDK=auto|1|0        Vulkan: use the LunarG SDK's glslc + headers when the
                               system ones are too old for all llama.cpp features
    MESA_PPA=1                 Vulkan/AMD on 24.04: upgrade an old Mesa (< 25) from
                               ppa:kisak/kisak-mesa without asking
    IPV4_ONLY=auto|1|0         Use IPv4 only for apt / git / curl / wget. auto (default)
                               turns it on when IPv6 is configured but doesn't work

    llama.cpp always tracks the newest upstream master commit; each run
    verifies that against GitHub and rebuilds when the commit or the
    compiler / CUDA / glslc / Vulkan header versions changed.

"
# ============================================================
# llama.cpp installer - generic Ubuntu 24.04 / 26.04
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
#   inkling2-mini      unsloth/Inkling-2-Mini-GGUF           (chat MoE + images, port 10024)
#   gemma4-e4b         unsloth/gemma-4-E4B-it-qat-GGUF       (chat + images,     port 10025)
#
# Paths:
#   Source:   /opt/llama.cpp
#   Binaries: /usr/local/bin/llama-*
#   Models:   /opt/ai/models/*.gguf
#             /opt/ai/models/mmproj/<name>-*.gguf  (vision projectors)
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
echo " llama.cpp installer - Ubuntu 24.04 / 26.04"
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

SUPPORTED_RELEASES="24.04 26.04"
if [[ "${ID:-}" != "ubuntu" || " $SUPPORTED_RELEASES " != *" ${VERSION_ID:-} "* ]]; then
    echo
    echo "WARNING: This script supports Ubuntu $SUPPORTED_RELEASES (found ${PRETTY_NAME:-unknown})."
    ask_yn "Continue anyway?" N || exit 1
fi

# ------------------------------------------------------------
# IPv4-only mode
#
# A host with an IPv6 address but no working IPv6 route makes apt, git,
# curl and wget hang trying the IPv6 addresses of GitHub / Hugging Face /
# the Ubuntu mirrors. IPV4_ONLY=auto (default) tests IPv6 and, if it
# doesn't work, makes this run use IPv4:
#   - apt:        /etc/apt/apt.conf.d/99llama-force-ipv4 (Acquire::ForceIPv4)
#   - git etc.:   a "prefer IPv4" precedence line in /etc/gai.conf
#   - curl/wget:  called with -4
# The apt + gai.conf changes are marked, and removed again on a later run
# once IPv6 works (or with IPV4_ONLY=0).
# ------------------------------------------------------------

IPV4_ONLY="${IPV4_ONLY:-auto}"
APT_IPV4_CONF="/etc/apt/apt.conf.d/99llama-force-ipv4"
GAI_MARK="# llama.cpp installer: prefer IPv4 (IPV4_ONLY)"

# ipv6_works  -- an IPv6 default route exists and an HTTPS connection over it succeeds
ipv6_works() {
    ip -6 route show default 2>/dev/null | grep -q . || return 1
    if command -v curl >/dev/null 2>&1; then
        curl -6 -sS -m 6 -o /dev/null https://raw.githubusercontent.com/ 2>/dev/null
    elif command -v wget >/dev/null 2>&1; then
        # any HTTP response (even an error status) means the connection worked
        wget -6 -q -T 6 -t 1 -O /dev/null https://raw.githubusercontent.com/ 2>/dev/null
        (( $? != 4 ))
    else
        return 0
    fi
}

USE_IPV4=0
case "$IPV4_ONLY" in
    1) USE_IPV4=1 ;;
    0) USE_IPV4=0 ;;
    *) if ip -6 route show default 2>/dev/null | grep -q . && ! ipv6_works; then
           USE_IPV4=1
       fi ;;
esac

echo
if (( USE_IPV4 )); then
    echo "Network: IPv4 only (IPV4_ONLY=$IPV4_ONLY) - apt, git, curl and wget will use IPv4."
    echo 'Acquire::ForceIPv4 "true";  // added by the llama.cpp installer (IPV4_ONLY)' >"$APT_IPV4_CONF"
    if ! grep -qF "$GAI_MARK" /etc/gai.conf 2>/dev/null; then
        printf '%s\nprecedence ::ffff:0:0/96  100\n' "$GAI_MARK" >>/etc/gai.conf
    fi
    curl() { command curl -4 "$@"; }
    wget() { command wget -4 "$@"; }
else
    echo "Network: IPv4 + IPv6."
    # Undo an IPv4-only setup from an earlier run.
    rm -f "$APT_IPV4_CONF"
    if grep -qF "$GAI_MARK" /etc/gai.conf 2>/dev/null; then
        # drop the marker line and the precedence line after it; cat > keeps perms
        gai_new="$(awk -v m="$GAI_MARK" '$0 == m { skip = 1; next } skip { skip = 0; next } { print }' /etc/gai.conf)"
        printf '%s\n' "$gai_new" >/etc/gai.conf
        echo "Removed the IPv4 preference from /etc/gai.conf (IPv6 works again)."
    fi
fi

# Release-specific names, e.g. 26.04 -> ubuntu2604 / linux-generic-hwe-26.04
UBUNTU_VER="${VERSION_ID:-24.04}"
CUDA_REPO="${CUDA_REPO:-ubuntu${UBUNTU_VER//./}}"
HWE_PKG="linux-generic-hwe-${UBUNTU_VER}"

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
    gnupg

# Nice-to-have monitoring tools (names can differ between releases)
apt_optional htop btop lm-sensors

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
    # Use NVIDIA's repo for this release (ubuntu2404 / ubuntu2604). The keyring
    # package creates /etc/apt/sources.list.d/cuda-<repo>-<arch>.list, so a
    # missing list file means it's a first install or the OS was upgraded
    # (e.g. 24.04 -> 26.04) and the repo has to be switched over.
    CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/${CUDA_REPO_ARCH}"
    CUDA_LIST="/etc/apt/sources.list.d/cuda-${CUDA_REPO}-${CUDA_REPO_ARCH}.list"
    CUDA_FROM_NVIDIA=1
    if [[ ! -f "$CUDA_LIST" ]]; then
        CUDA_KEYRING="$(mktemp --suffix=.deb)"
        if wget -q "$CUDA_REPO_URL/cuda-keyring_1.1-1_all.deb" -O "$CUDA_KEYRING"; then
            # Drop repo lists left over from another Ubuntu release.
            for f in /etc/apt/sources.list.d/cuda-ubuntu*.list; do
                [[ -e "$f" && "$f" != "$CUDA_LIST" ]] && rm -f "$f" && echo "Removed old CUDA repo: $f"
            done
            dpkg -i "$CUDA_KEYRING"
            apt-get update
        else
            CUDA_FROM_NVIDIA=0
        fi
        rm -f "$CUDA_KEYRING"
    fi

    if [[ "$CUDA_FROM_NVIDIA" == "1" ]]; then
        apt-get install -y cuda-toolkit nvtop
        upgrade_installed '^(cuda-|libcublas|libcudnn|libnccl)'
        CUDA_HOME="/usr/local/cuda"
    else
        # No NVIDIA repo for this release yet: fall back to Ubuntu's own
        # packaged toolkit (may be older, so newest GPUs may be unsupported).
        echo "WARNING: NVIDIA has no CUDA repo at $CUDA_REPO_URL."
        echo "         Using Ubuntu's nvidia-cuda-toolkit instead. To use NVIDIA's"
        echo "         previous-LTS repo, re-run with: sudo CUDA_REPO=ubuntu2404 $0"
        apt-get install -y nvidia-cuda-toolkit nvtop
        CUDA_HOME="/usr"
        [[ -x /usr/lib/nvidia-cuda-toolkit/bin/nvcc ]] && CUDA_HOME="/usr/lib/nvidia-cuda-toolkit"
    fi
    if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
        echo "ERROR: nvcc not found in $CUDA_HOME/bin after installing the CUDA toolkit."
        exit 1
    fi

    # Newer releases ship a newer GCC than nvcc may accept as host compiler
    # (26.04 has GCC 15). Test it, and fall back to an older g++ if needed.
    CUDA_HOST_CXX=""
    cuda_host_ok() {
        local d rc=0
        d="$(mktemp -d)"
        printf '__global__ void k() {}\nint main() { k<<<1, 1>>>(); return 0; }\n' >"$d/t.cu"
        "$CUDA_HOME/bin/nvcc" ${1:+-ccbin "$1"} -c "$d/t.cu" -o "$d/t.o" >/dev/null 2>&1 || rc=1
        rm -rf "$d"
        return $rc
    }
    if ! cuda_host_ok ""; then
        echo "nvcc does not accept the default g++ ($(g++ -dumpversion)); trying an older one..."
        for v in 14 13 12; do
            apt_optional "g++-$v" >/dev/null
            if command -v "g++-$v" >/dev/null 2>&1 && cuda_host_ok "g++-$v"; then
                CUDA_HOST_CXX="$(command -v "g++-$v")"
                echo "Using $CUDA_HOST_CXX as the CUDA host compiler."
                break
            fi
        done
        if [[ -z "$CUDA_HOST_CXX" ]]; then
            echo "ERROR: nvcc can't compile with any installed g++. Install a CUDA"
            echo "       toolkit that supports g++ $(g++ -dumpversion), or a supported g++."
            exit 1
        fi
    fi

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

    # Newer AMD/Intel GPUs (e.g. Strix Halo, Arc B-series) need kernel 6.11+.
    # 24.04 shipped 6.8, so offer its HWE kernel; 26.04's stock kernel is
    # already new enough. Keep an installed HWE kernel up to date either way.
    KERNEL_MAJOR_MINOR=$(uname -r | awk -F. '{ printf "%d%02d", $1, $2 }')
    if dpkg -s "$HWE_PKG" >/dev/null 2>&1; then
        apt-get install -y "$HWE_PKG"
    elif (( KERNEL_MAJOR_MINOR < 611 )) && apt-cache show "$HWE_PKG" >/dev/null 2>&1; then
        echo
        echo "Running kernel $(uname -r). Recent AMD/Intel GPUs work best with the"
        echo "$UBUNTU_VER HWE kernel (needs a reboot)."
        if ask_yn "Install $HWE_PKG?" Y; then
            apt-get install -y "$HWE_PKG"
        fi
    elif (( KERNEL_MAJOR_MINOR < 611 )); then
        echo "NOTE: kernel $(uname -r) is older than 6.11; recent AMD/Intel GPUs may not work well."
    fi

    echo
    echo "Vulkan devices:"
    VK_SUMMARY="$(vulkaninfo --summary 2>/dev/null || true)"
    if [[ -n "$VK_SUMMARY" ]]; then
        sed -n '/^Devices:/,$p' <<<"$VK_SUMMARY"
    else
        echo "WARNING: vulkaninfo returned an error. The GPU driver may need attention"
        echo "         (a reboot after a kernel/firmware update often fixes this)."
    fi

    # One line per Vulkan device: "<index> <type> <vendorID> <driverVersion>"
    VK_DEVS="$(awk '
        /^GPU[0-9]+:/            { if (idx != "") print idx, type, vendor, drv; idx = substr($1, 4, length($1) - 4); type = vendor = drv = "-" }
        /deviceType[ \t]*=/      { type = $NF }
        /vendorID[ \t]*=/        { vendor = $NF }
        /driverVersion[ \t]*=/   { drv = $NF }
        END                      { if (idx != "") print idx, type, vendor, drv }
    ' <<<"$VK_SUMMARY")"

    # llama.cpp uses ALL discrete + integrated GPUs by default, so a desktop
    # Radeon card would share the model with the (much slower) iGPU in the
    # CPU. When both kinds exist, use only the discrete GPU(s).
    # Override in /etc/llama.cpp/global.conf with GGML_VK_VISIBLE_DEVICES=...
    VK_DISCRETE="$(awk '$2 ~ /DISCRETE/   { printf "%s%s", s, $1; s = "," }' <<<"$VK_DEVS")"
    VK_INTEGRATED="$(awk '$2 ~ /INTEGRATED/ { printf "%s%s", s, $1; s = "," }' <<<"$VK_DEVS")"
    VK_VISIBLE=""
    if [[ -n "$VK_DISCRETE" && -n "$VK_INTEGRATED" ]]; then
        VK_VISIBLE="$VK_DISCRETE"
        echo
        echo "Discrete + integrated GPUs found: llama.cpp will use only discrete Vulkan device(s) $VK_VISIBLE."
    fi

    if [[ "$HAS_AMD_GPU" == "1" ]]; then
        # Mesa RADV version (RADV is the AMD Vulkan driver llama.cpp uses).
        # RDNA3/4 matrix cores and Strix Halo need a recent Mesa (25.0+).
        RADV_VER="$(awk '$3 == "0x1002" { print $4; exit }' <<<"$VK_DEVS")"
        echo
        echo "Mesa RADV driver: ${RADV_VER:-not found}"
        if [[ "$RADV_VER" =~ ^([0-9]+)\. ]] && (( BASH_REMATCH[1] < 25 )); then
            echo "Mesa $RADV_VER is old for llama.cpp on AMD (slower, newer GPUs may be missing features)."
            if [[ "$UBUNTU_VER" == "24.04" ]]; then
                echo "The kisak-mesa PPA has current stable Mesa for 24.04 (third-party, but widely used)."
                if [[ "${MESA_PPA:-0}" == "1" ]] || ask_yn "Add ppa:kisak/kisak-mesa and upgrade Mesa?" N; then
                    add-apt-repository -y ppa:kisak/kisak-mesa
                    apt-get update
                    apt-get install -y mesa-vulkan-drivers libvulkan1
                    upgrade_installed '^(mesa-|libgl1-mesa|libegl-mesa|libglx-mesa|libdrm)'
                    echo "Mesa upgraded: $(vulkaninfo --summary 2>/dev/null | awk '/driverVersion/ { print $NF; exit }')"
                fi
            fi
        fi

        # AMD APUs (Strix Halo, Phoenix, ...): the GPU uses system RAM through
        # GTT, which the kernel caps at ~50% of RAM by default. Raise the cap so
        # big models fit (kernel parameter, takes effect after a reboot).
        if grep -q 'INTEGRATED' <<<"$(awk '$3 == "0x1002"' <<<"$VK_DEVS")" && [[ -z "$VK_DISCRETE" ]]; then
            APU_RAM_MIB=$(awk '/^MemTotal:/ { printf "%d", $2 / 1024 }' /proc/meminfo)
            if (( APU_RAM_MIB >= 65536 )); then
                GTT_TARGET_MIB=$(( APU_RAM_MIB - 8192 ))
            else
                GTT_TARGET_MIB=$(( APU_RAM_MIB * 3 / 4 ))
            fi
            GTT_NOW_MIB=$(cat /sys/class/drm/card*/device/mem_info_gtt_total 2>/dev/null |
                awk '{ if ($1 > m) m = $1 } END { printf "%d", m / 1048576 }')
            echo "AMD APU: GPU can address ${GTT_NOW_MIB} MiB of system RAM (GTT)."
            GTT_CFG="/etc/default/grub.d/90-llama-amd-gtt.cfg"
            if (( GTT_NOW_MIB < GTT_TARGET_MIB * 9 / 10 )) && [[ ! -f "$GTT_CFG" ]]; then
                GTT_PAGES=$(( GTT_TARGET_MIB * 256 ))   # 4 KiB pages
                if [[ -d /etc/default/grub.d ]] && command -v update-grub >/dev/null 2>&1; then
                    echo "Raising it to ${GTT_TARGET_MIB} MiB lets much larger models run on the GPU."
                    if ask_yn "Set ttm.pages_limit=$GTT_PAGES on the kernel command line (needs a reboot)?" Y; then
                        cat >"$GTT_CFG" <<EOF
# Added by the llama.cpp installer: let the AMD iGPU use ~${GTT_TARGET_MIB} MiB of RAM (GTT).
GRUB_CMDLINE_LINUX_DEFAULT="\$GRUB_CMDLINE_LINUX_DEFAULT ttm.pages_limit=$GTT_PAGES ttm.page_pool_size=$GTT_PAGES"
EOF
                        update-grub
                        touch /var/run/reboot-required
                    fi
                else
                    echo "To let larger models run on the GPU, add this kernel parameter and reboot:"
                    echo "  ttm.pages_limit=$GTT_PAGES ttm.page_pool_size=$GTT_PAGES"
                fi
            elif [[ -f "$GTT_CFG" ]]; then
                echo "GTT limit is set in $GTT_CFG (reboot if it isn't active yet)."
            fi
        fi
    fi

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

# Always the newest master commit (newer than the latest bNNNN release tag).
# A shallow fetch + hard reset to FETCH_HEAD can't get stuck on an old
# commit, a stale remote-tracking ref, local edits, or a moved repo URL.
UPSTREAM_SHA="$(git ls-remote "$LLAMA_REPO" refs/heads/master 2>/dev/null | awk '{ print $1 }' || true)"
if [[ -d "$LLAMA_DIR/.git" ]]; then
    git -C "$LLAMA_DIR" remote set-url origin "$LLAMA_REPO"
    if git -C "$LLAMA_DIR" fetch --depth=1 origin master; then
        git -C "$LLAMA_DIR" reset --hard FETCH_HEAD
    elif [[ -x /usr/local/bin/llama-server ]]; then
        echo "WARNING: could not reach GitHub - keeping the current llama.cpp build."
    else
        echo "ERROR: could not fetch llama.cpp and no build is installed."; exit 1
    fi
else
    rm -rf "$LLAMA_DIR"
    git clone --depth=1 --branch master "$LLAMA_REPO" "$LLAMA_DIR"
fi
LLAMA_SHA="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
LLAMA_COMMIT="${LLAMA_SHA:0:7}"
echo "llama.cpp commit: $LLAMA_COMMIT  ($(git -C "$LLAMA_DIR" log -1 --format='%cd' --date=iso))"
if [[ -n "$UPSTREAM_SHA" && "$UPSTREAM_SHA" == "$LLAMA_SHA" ]]; then
    echo "Verified: matches the latest upstream master."
elif [[ -n "$UPSTREAM_SHA" ]]; then
    echo "WARNING: upstream master is ${UPSTREAM_SHA:0:7}, but the source is at $LLAMA_COMMIT."
fi

# The shallow clone has no history, so llama.cpp would report "version: 1".
# Ask GitHub for the real commit count (= the bNNNN build number).
LLAMA_BUILD_NUMBER="$(curl -fsI -m 10 "https://api.github.com/repos/ggml-org/llama.cpp/commits?sha=$LLAMA_SHA&per_page=1" 2>/dev/null |
    sed -nE 's/^link:.*[?&]page=([0-9]+)>; rel="last".*/\1/Ip' | tr -d '\r' || true)"
[[ "$LLAMA_BUILD_NUMBER" =~ ^[0-9]+$ ]] && echo "llama.cpp build:  b$LLAMA_BUILD_NUMBER" || LLAMA_BUILD_NUMBER=""

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
                         -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc")
            [[ -n "$CUDA_HOST_CXX" ]] && CMAKE_ARGS+=(-DCMAKE_CUDA_HOST_COMPILER="$CUDA_HOST_CXX") ;;
    vulkan) CMAKE_ARGS+=(-DGGML_VULKAN=ON) ;;
esac

# ---- Vulkan: make sure every shader feature llama.cpp has gets built ----
# llama.cpp silently drops features (coopmat matrix cores on RDNA3/4,
# integer dot, bf16, fp8/fp4) when glslc or the Vulkan headers are too old.
# Ubuntu 24.04's are; then the newest LunarG Vulkan SDK is used for glslc +
# headers only (the system Vulkan loader and Mesa RADV still run it).
#   LUNARG_SDK=auto (default) | 1 (always) | 0 (never)
VK_SDK_ROOT="/opt/vulkan-sdk"
LUNARG_SDK="${LUNARG_SDK:-auto}"
if [[ "$BACKEND" == "vulkan" ]]; then
    VK_MISSING=()
    VK_TESTS="$LLAMA_DIR/ggml/src/ggml-vulkan/vulkan-shaders/feature-tests"
    for t in "$VK_TESTS"/*.comp; do
        [[ -e "$t" ]] || continue
        [[ "$(basename "$t")" == coopmat2* ]] && continue   # NVIDIA-only
        glslc -o /dev/null -fshader-stage=compute --target-env=vulkan1.3 "$t" >/dev/null 2>&1 ||
            VK_MISSING+=("$(basename "$t" .comp)")
    done
    grep -q 'VK_KHR_shader_bfloat16' /usr/include/vulkan/vulkan_core.h 2>/dev/null ||
        VK_MISSING+=("vulkan-headers")

    if [[ "$LUNARG_SDK" == "1" || ( "$LUNARG_SDK" == "auto" && ${#VK_MISSING[@]} -gt 0 ) ]]; then
        echo "System glslc / Vulkan headers lack: ${VK_MISSING[*]:-nothing (LUNARG_SDK=1)}"
        if [[ "$ARCH" != "x86_64" ]]; then
            echo "WARNING: the LunarG SDK is x86_64 only; building with the system toolchain"
            echo "         (those Vulkan features will be disabled)."
        else
            VK_SDK_VER="$(curl -fsS -m 15 https://vulkan.lunarg.com/sdk/latest/linux.txt 2>/dev/null | tr -d '[:space:]' || true)"
            if [[ "$VK_SDK_VER" =~ ^[0-9.]+$ && ! -x "$VK_SDK_ROOT/$VK_SDK_VER/x86_64/bin/glslc" ]]; then
                echo "Downloading LunarG Vulkan SDK $VK_SDK_VER (~350 MB, one time per version)..."
                VK_TMP="$(mktemp -d)"
                if curl -fL --retry 5 -o "$VK_TMP/sdk.tar.xz" \
                        "https://sdk.lunarg.com/sdk/download/$VK_SDK_VER/linux/vulkan_sdk.tar.xz" &&
                   tar -xJf "$VK_TMP/sdk.tar.xz" -C "$VK_TMP"; then
                    VK_GLSLC_FOUND="$(find "$VK_TMP" -path '*/x86_64/bin/glslc' -type f | head -1)"
                    if [[ -n "$VK_GLSLC_FOUND" ]]; then
                        mkdir -p "$VK_SDK_ROOT"
                        rm -rf "${VK_SDK_ROOT:?}/$VK_SDK_VER"
                        mv "$(dirname "$(dirname "$(dirname "$VK_GLSLC_FOUND")")")" "$VK_SDK_ROOT/$VK_SDK_VER"
                        # keep only the newest SDK
                        for d in "$VK_SDK_ROOT"/*/; do
                            [[ "$(basename "$d")" != "$VK_SDK_VER" ]] && rm -rf "$d"
                        done
                    fi
                else
                    echo "WARNING: Vulkan SDK download failed."
                fi
                rm -rf "$VK_TMP"
            fi
            # Newest installed SDK (also used when lunarg.com can't be reached).
            VK_SDK="$(ls -d "$VK_SDK_ROOT"/*/x86_64 2>/dev/null | sort -V | tail -1 || true)"
            if [[ -n "$VK_SDK" && -x "$VK_SDK/bin/glslc" ]]; then
                echo "Using Vulkan SDK: $VK_SDK"
                CMAKE_ARGS+=(-DVulkan_GLSLC_EXECUTABLE="$VK_SDK/bin/glslc"
                             -DVulkan_INCLUDE_DIR="$VK_SDK/include")
            else
                echo "WARNING: no Vulkan SDK available; those Vulkan features will be disabled."
            fi
        fi
    else
        echo "System glslc and Vulkan headers support all llama.cpp Vulkan features."
    fi
fi

# Toolchain versions: an upgraded compiler / CUDA / glslc / Vulkan headers
# also triggers a rebuild, not just a new llama.cpp commit.
TOOLCHAIN="gcc-$(gcc -dumpfullversion 2>/dev/null)"
[[ "$BACKEND" == "cuda" ]] && TOOLCHAIN+=" nvcc-$("$CUDA_HOME/bin/nvcc" --version | sed -nE 's/.*release ([0-9.]+).*/\1/p')"
if [[ "$BACKEND" == "vulkan" ]]; then
    VK_GLSLC_BIN="${VK_SDK:+$VK_SDK/bin/}glslc"
    VK_HDR="${VK_SDK:+$VK_SDK/include}"; VK_HDR="${VK_HDR:-/usr/include}"
    TOOLCHAIN+=" glslc-$("$VK_GLSLC_BIN" --version 2>/dev/null | head -1 | awk '{ print $NF }')"
    TOOLCHAIN+=" vkhdr-$(sed -nE 's/^#define VK_HEADER_VERSION ([0-9]+).*/\1/p' "$VK_HDR/vulkan/vulkan_core.h" 2>/dev/null)"
fi

STAMP_FILE="$LLAMA_DIR/.install-stamp"
NEW_STAMP="$LLAMA_COMMIT|$TOOLCHAIN|${CMAKE_ARGS[*]}"
OLD_STAMP="$(cat "$STAMP_FILE" 2>/dev/null || true)"
REBUILT=0

if [[ "$FORCE_REBUILD" == "1" || "$NEW_STAMP" != "$OLD_STAMP" || ! -x /usr/local/bin/llama-server ]]; then
    # Different backend/flags/toolchain -> start from a clean build directory.
    if [[ "${OLD_STAMP#*|}" != "${NEW_STAMP#*|}" ]]; then
        rm -rf "$LLAMA_DIR/build"
    fi

    # Build number is passed outside CMAKE_ARGS so it doesn't force clean builds.
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" "${CMAKE_ARGS[@]}" \
        ${LLAMA_BUILD_NUMBER:+-DLLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER"} \
        -DLLAMA_BUILD_COMMIT="$LLAMA_COMMIT" | tee "$LLAMA_DIR/.cmake-configure.log"
    if [[ "$BACKEND" == "vulkan" ]]; then
        echo
        echo "Vulkan shader features compiled in:"
        grep -E 'supported by glslc' "$LLAMA_DIR/.cmake-configure.log" | grep -v -i coopmat2 | sed 's/^-- /  /' || true
    fi
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

# Confirm llama.cpp itself can use the GPU, as the service user (this also
# catches missing video/render group access) with the same device selection.
if [[ "$BACKEND" != "cpu" ]]; then
    echo
    echo "GPU devices seen by llama.cpp (as user $SERVICE_USER):"
    DEV_OUT="$(runuser -u "$SERVICE_USER" -- env HOME="$SERVICE_HOME" \
        ${VK_VISIBLE:+GGML_VK_VISIBLE_DEVICES="$VK_VISIBLE"} \
        /usr/local/bin/llama-server -v --list-devices 2>&1 || true)"
    # -v adds the ggml_vulkan line per device: fp16/bf16/int dot/matrix cores
    grep -E 'ggml_vulkan: [0-9]+ =|^  (Vulkan|CUDA)[0-9]+:|  Device [0-9]+:' <<<"$DEV_OUT" |
        sed -E 's/^.*(ggml_vulkan:)/\1/; s/^/  /' || true
    BACKEND_TAG="Vulkan"; [[ "$BACKEND" == "cuda" ]] && BACKEND_TAG="CUDA"
    if ! grep -qE "^ *${BACKEND_TAG}[0-9]+:" <<<"$DEV_OUT"; then
        echo "WARNING: llama.cpp found no $BACKEND_TAG device - models will run on the CPU."
        echo "         Try a reboot (new kernel/driver/groups), then re-run this script."
        echo "         Full output: sudo -u $SERVICE_USER llama-server --list-devices"
    elif [[ "$BACKEND" == "vulkan" && "$HAS_AMD_GPU" == "1" ]] && grep -q 'matrix cores: none' <<<"$DEV_OUT"; then
        echo "NOTE: 'matrix cores: none' is expected on RDNA1/2 and older; RDNA3/4"
        echo "      (RX 7000/9000, Strix Halo) should show KHR_coopmat with a current Mesa."
    fi
fi

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

# Managed by this script on every run (Vulkan device choice, see step 4).
# To override, set GGML_VK_VISIBLE_DEVICES in global.conf (loaded after this).
{
    echo "# Generated by the llama.cpp installer - overwritten on every run."
    echo "# Override in $CONF_ROOT/global.conf instead."
    if [[ "$BACKEND" == "vulkan" && -n "${VK_VISIBLE:-}" ]]; then
        echo "GGML_VK_VISIBLE_DEVICES=$VK_VISIBLE"
    fi
} >"$CONF_ROOT/backend.conf"

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
EnvironmentFile=-$CONF_ROOT/backend.conf
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

MODEL_KEYS=(qwen3-embed-4b qwen36-35b-a3b nemotron-bolt-e4b qwen38-27b gemma4-26b-a4b inkling2-mini gemma4-e4b)
declare -A M_REPO M_DESC M_PORT M_ARGS M_CANDS M_OVERHEAD M_DEFAULT M_REMOTE M_EXTRA

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

# Mainline llama.cpp needs the Mainline_Support files (inkling2_loop layout,
# llama.cpp b12611+); the repo-root GGUFs only load in Unsloth's fork. They
# are published as .gguf_file and saved locally as .gguf (see M_REMOTE).
M_REPO[inkling2-mini]="unsloth/Inkling-2-Mini-GGUF"
M_DESC[inkling2-mini]="Inkling-2-Mini (26B, 3.4B active) - MoE chat model, text + images"
M_PORT[inkling2-mini]=10024
M_ARGS[inkling2-mini]="-c 32768 --jinja --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 --mmproj $MODEL_DIR/mmproj/inkling2-mini-mmproj-F16.gguf"
M_OVERHEAD[inkling2-mini]=3712       # KV cache etc. + ~610 MiB vision projector
M_DEFAULT[inkling2-mini]=N
M_REMOTE[inkling2-mini]="Mainline_Support/%s_file"
M_EXTRA[inkling2-mini]="mmproj-F16.gguf|$MODEL_DIR/mmproj/inkling2-mini-mmproj-F16.gguf"

M_REPO[gemma4-e4b]="unsloth/gemma-4-E4B-it-qat-GGUF"
M_DESC[gemma4-e4b]="Gemma-4-E4B (QAT) - small, fast chat model from Google, text + images"
M_PORT[gemma4-e4b]=10025
M_ARGS[gemma4-e4b]="-c 32768 --jinja --temp 1.0 --top-p 0.95 --top-k 64 --mmproj $MODEL_DIR/mmproj/gemma4-e4b-mmproj-F16.gguf"
M_OVERHEAD[gemma4-e4b]=3072          # KV cache etc. + ~945 MiB vision projector
M_DEFAULT[gemma4-e4b]=N
M_EXTRA[gemma4-e4b]="mmproj-F16.gguf|$MODEL_DIR/mmproj/gemma4-e4b-mmproj-F16.gguf"

# M_REMOTE: printf format turning the local file name into its path in the
# repo (default "%s"). M_EXTRA: extra files the server needs, as
# "repo-path|local-path ..."; downloaded and updated along with the model.

# Candidate files, best first: "file:bytes". The first one that fits in the
# remaining memory is used. Override with <NAME>_FILE (see header).
M_CANDS[qwen3-embed-4b]="Qwen3-Embedding-4B-Q8_0.gguf:4279660224 Qwen3-Embedding-4B-Q4_K_M.gguf:2496703776"
M_CANDS[qwen36-35b-a3b]="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf:22360456160 Qwen3.6-35B-A3B-UD-IQ4_XS.gguf:17730509792 Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf:16845511648 Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf:13211155424"
M_CANDS[nemotron-bolt-e4b]="Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf:6180446208"
M_CANDS[qwen38-27b]="Qwen3.8-27B-UD-Q4_K_XL.gguf:17559178144 Qwen3.8-27B-UD-Q4_K_S.gguf:15358213024 Qwen3.8-27B-UD-IQ4_XS.gguf:14252845984 Qwen3.8-27B-UD-Q3_K_XL.gguf:13146393504"
M_CANDS[gemma4-26b-a4b]="gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf:14249047104"
M_CANDS[inkling2-mini]="Inkling-2-Mini-UD-IQ4_XS.gguf:15733756512"
M_CANDS[gemma4-e4b]="gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf:4215695776 gemma-4-E4B-it-qat-UD-Q2_K_XL.gguf:3219532192"

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

# download_model <repo> <repo path> <local path>
#   -- returns 0 if the file is new or changed, 1 if unchanged, 2 on failure
download_model() {
    local repo="$1" src="$2" dest="$3"
    local url="https://huggingface.co/$repo/resolve/main/$src"
    local file remote_sha local_sha
    file="$(basename "$dest")"
    mkdir -p "$(dirname "$dest")"

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

# is_candidate <key> <file>  -- the file is one this script manages (can be updated)
is_candidate() {
    local key="$1" file="$2" cand var
    var="$(tr 'a-z-' 'A-Z_' <<<"$key")_FILE"
    [[ "${!var:-}" == "$file" ]] && return 0
    for cand in ${M_CANDS[$key]}; do
        [[ "${cand%%:*}" == "$file" ]] && return 0
    done
    return 1
}

if [[ -n "${MODELS:-}" ]]; then
    echo "MODELS=$MODELS set in the environment - not asking about new models."
fi

# Pass 1: installed models. They are only checked for updates (never newly
# started); the question is whether they should start at boot. Their memory
# is reserved first so the suggestions for new models below are accurate.
# SELECTED entries: "<key>|<file>|<new|update>|<autostart 1/0>"
# find_installed <key>  -> path of the model file already on disk, or nothing.
# 1) the file named in its config, 2) otherwise any known file for it in
# MODEL_DIR (e.g. downloaded by an older script that wrote no config).
find_installed() {
    local key="$1" conf="$CONF_DIR/$1.conf" p var cand
    if [[ -f "$conf" ]]; then
        p="$(sed -n 's/^MODEL=//p' "$conf" | tr -d '"' | head -1)"
        [[ -n "$p" && -f "$p" ]] && { echo "$p"; return; }
    fi
    var="$(tr 'a-z-' 'A-Z_' <<<"$key")_FILE"
    [[ -n "${!var:-}" && -f "$MODEL_DIR/${!var}" ]] && { echo "$MODEL_DIR/${!var}"; return; }
    for cand in ${M_CANDS[$key]}; do
        [[ -f "$MODEL_DIR/${cand%%:*}" ]] && { echo "$MODEL_DIR/${cand%%:*}"; return; }
    done
    return 0
}

# port_owner <port>  -> what is listening on the port, or nothing
port_owner() {
    ss -Hltnp "sport = :$1" 2>/dev/null | sed -nE 's/.*users:\(\("([^"]+)",pid=([0-9]+).*/\1 (pid \2)/p' | head -1
}

# Servers from the older installer (llama-<name>.service) use the same ports.
LEGACY_UNITS="$(systemctl list-unit-files 'llama-*.service' --no-legend 2>/dev/null | awk '{ print $1 }' | tr '\n' ' ' || true)"
if [[ -n "${LEGACY_UNITS// /}" ]]; then
    echo "NOTE: units from an older installer exist: $LEGACY_UNITS"
    echo "      They use the same ports (10020-10022). Disable the ones you replace:"
    echo "      sudo systemctl disable --now <unit>"
    echo
fi

SELECTED=()
declare -A INSTALLED
for key in "${MODEL_KEYS[@]}"; do
    model_path="$(find_installed "$key")"
    [[ -n "$model_path" ]] || continue
    INSTALLED[$key]=1
    file="$(basename "$model_path")"
    mib=$(( $(stat -c %s "$model_path") / 1048576 ))
    REMAINING_MIB=$(( REMAINING_MIB - mib - ${M_OVERHEAD[$key]} ))

    configured=0; [[ -f "$CONF_DIR/$key.conf" ]] && configured=1
    state="stopped"; systemctl is-active --quiet "llama@$key.service" && state="running"
    boot="off";      systemctl is-enabled --quiet "llama@$key.service" 2>/dev/null && boot="on"

    if (( configured )); then
        echo "$key  (port ${M_PORT[$key]})  - installed, $state, autostart $boot"
    else
        echo "$key  (port ${M_PORT[$key]})  - model file found, no service yet (will be set up, not started)"
    fi
    echo "  File: $file  (~$(( (mib + 512) / 1024 )) GiB)"
    owner="$(port_owner "${M_PORT[$key]}")"
    if [[ "$state" != "running" && -n "$owner" ]]; then
        echo "  NOTE: port ${M_PORT[$key]} is in use by $owner - stop that first to run $key."
    fi
    if [[ -n "${MODELS:-}" ]]; then
        auto=0; [[ "$boot" == "on" ]] && auto=1     # non-interactive: keep as is
    else
        auto=0
        ask_yn "  Autostart $key at boot?" "$([[ "$boot" == "on" ]] && echo Y || echo N)" && auto=1
    fi
    echo
    SELECTED+=("$key|$model_path|update|$auto")
done

# Pass 2: models that are not installed yet.
for key in "${MODEL_KEYS[@]}"; do
    [[ -n "${INSTALLED[$key]:-}" ]] && continue
    read -r file mib < <(pick_file "$key")
    default="${M_DEFAULT[$key]}"
    [[ -f "$CONF_DIR/$key.conf" ]] && default=Y   # config left but model file deleted

    echo "$key  (port ${M_PORT[$key]})  - not installed"
    echo "  ${M_DESC[$key]}"
    echo "  https://huggingface.co/${M_REPO[$key]}"
    if (( mib > 0 )); then
        echo "  File: $file  (~$(( (mib + 512) / 1024 )) GiB)"
        if [[ -f "$MODEL_DIR/$file.part" ]]; then
            part_mib=$(( $(stat -c %s "$MODEL_DIR/$file.part") / 1048576 ))
            echo "  Partial download found ($(( part_mib * 100 / (mib > 0 ? mib : 1) ))%) - it will resume."
            default=Y
        fi
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
        SELECTED+=("$key|$MODEL_DIR/$file|new|1")
        REMAINING_MIB=$(( REMAINING_MIB - mib - ${M_OVERHEAD[$key]} ))
    fi
done

# Process in MODEL_KEYS order, so the embedding server starts before chat models.
ORDERED=()
for key in "${MODEL_KEYS[@]}"; do
    for entry in "${SELECTED[@]}"; do
        [[ "${entry%%|*}" == "$key" ]] && ORDERED+=("$entry")
    done
done

STARTED=()
for entry in "${ORDERED[@]}"; do
    IFS='|' read -r key model_path mode auto <<<"$entry"
    file="$(basename "$model_path")"
    echo "== $key"

    changed=0
    rc=0
    if [[ "$mode" == "update" ]] && ! is_candidate "$key" "$file"; then
        echo "  $file is a custom model file - not checked for updates."
        rc=1
    elif [[ "$mode" == "update" && "$(dirname "$model_path")" != "$MODEL_DIR" ]]; then
        echo "  $model_path is outside $MODEL_DIR - not checked for updates."
        rc=1
    else
        [[ "$mode" == "update" ]] && echo "  Checking for updates..."
        # shellcheck disable=SC2059
        src="$(printf "${M_REMOTE[$key]:-%s}" "$file")"
        download_model "${M_REPO[$key]}" "$src" "$model_path" || rc=$?
    fi
    # Extra files (e.g. vision projector) - only for files this script manages.
    if (( rc != 2 )) && [[ -n "${M_EXTRA[$key]:-}" ]] && is_candidate "$key" "$file"; then
        for extra in ${M_EXTRA[$key]}; do
            xrc=0
            download_model "${M_REPO[$key]}" "${extra%%|*}" "${extra##*|}" || xrc=$?
            if (( xrc == 0 )); then
                (( rc == 1 )) && rc=0       # restart a running server for the new file
            elif (( xrc == 2 )) && [[ ! -f "${extra##*|}" ]]; then
                rc=2                        # the server can't start without it
            fi
        done
    fi
    if (( rc == 2 )); then
        if [[ "$mode" == "new" ]]; then
            echo "  Skipping $key (download failed). Re-run the script to retry."
            continue
        fi
        echo "  Update failed - keeping the installed file. Re-run the script to retry."
    fi
    file_updated=0; conf_written=0
    (( rc == 0 )) && changed=1 && file_updated=1
    if [[ ! -f "$model_path" ]]; then
        echo "  Skipping $key ($model_path missing)."
        continue
    fi

    conf="$CONF_DIR/$key.conf"
    # Keep hand edits: only write the config for a new install (or when the
    # config points at a different file than the one just downloaded).
    if [[ ! -f "$conf" ]] || ! grep -qx "MODEL=$model_path" "$conf"; then
        cat >"$conf" <<EOF
# llama.cpp server config, used by llama@$key.service
# Edit, then apply with: llama-start $key
MODEL=$model_path
ALIAS=$key
PORT=${M_PORT[$key]}
ARGS="${M_ARGS[$key]}"
EOF
        changed=1; conf_written=1
        [[ "$mode" == "update" ]] && echo "  Created service config $conf (unit llama@$key)."
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

    if [[ "$auto" == "1" ]]; then
        systemctl enable "llama@$key.service" >/dev/null 2>&1
        echo "  Autostart: on"
    else
        systemctl disable "llama@$key.service" >/dev/null 2>&1 || true
        echo "  Autostart: off"
    fi

    # New installs start now. Installed models are never started here; a
    # running one is restarted only when its file or llama.cpp changed.
    running=0; systemctl is-active --quiet "llama@$key.service" && running=1
    start_it=0
    if [[ "$mode" == "new" ]]; then
        start_it=1
    elif (( running )) && [[ "$changed" == "1" || "$REBUILT" == "1" ]]; then
        start_it=1
    fi
    if (( start_it )); then
        if [[ "$mode" == "new" ]]; then
            echo "  Starting llama@$key ..."
        else
            echo "  Restarting llama@$key to load the update ..."
        fi
        systemctl restart "llama@$key.service"
        if wait_healthy "${M_PORT[$key]}" 600; then
            echo "  Ready on port ${M_PORT[$key]}."
        else
            echo "  WARNING: not healthy yet - check: llama-logs $key"
        fi
    elif (( running )); then
        echo "  Running and up to date."
    else
        (( file_updated )) && echo "  Updated; it will use the new file when started."
        (( file_updated || conf_written )) || echo "  Up to date."
        echo "  Not running (left stopped). Start it with: llama-start $key"
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
echo "Test commands for the running models"
echo "(OpenAI-compatible API; model name = alias)"
echo "----------------------------------------------"
HOST_IP="${HOST_IP:-127.0.0.1}"
# If global.conf requires an API key, add: -H 'Authorization: Bearer <key>'
NEED_KEY=0
grep -qE '^GLOBAL_ARGS=.*--api-key' "$CONF_ROOT/global.conf" 2>/dev/null && NEED_KEY=1
AUTH_LINE=""
(( NEED_KEY )) && AUTH_LINE="    -H 'Authorization: Bearer <your-api-key>' \\"

TESTED=0
for conf in "$CONF_DIR"/*.conf; do
    [[ -e "$conf" ]] || continue
    name=$(basename "$conf" .conf)
    systemctl is-active --quiet "llama@$name.service" || continue
    port=$(sed -n 's/^PORT=//p' "$conf" | tr -d '"')
    alias=$(sed -n 's/^ALIAS=//p' "$conf" | tr -d '"')
    alias="${alias:-$name}"
    args=$(sed -n 's/^ARGS=//p' "$conf")
    TESTED=1

    echo
    if [[ "$args" == *--embedding* ]]; then
        echo "# $alias  (embeddings, port $port)"
        echo "curl -s http://$HOST_IP:$port/v1/embeddings \\"
        echo "    -H 'Content-Type: application/json' \\"
        [[ -n "$AUTH_LINE" ]] && echo "$AUTH_LINE"
        echo "    -d '{\"model\":\"$alias\",\"input\":\"Hello world\"}' | jq '.data[0].embedding | length'"
        echo "# (Qwen3-Embedding: prefix queries, not documents, with"
        echo "#  \"Instruct: <task>\\nQuery: <text>\" for best retrieval.)"
    else
        echo "# $alias  (chat, port $port)"
        echo "curl -s http://$HOST_IP:$port/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        [[ -n "$AUTH_LINE" ]] && echo "$AUTH_LINE"
        echo "    -d '{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":256,\"chat_template_kwargs\":{\"enable_thinking\":false}}' \\"
        echo "    | jq -r '.choices[0].message.content'"
    fi
done

if (( TESTED )); then
    echo
    echo "# Thinking models: drop \"chat_template_kwargs\" to let them reason first."
    echo "# List what a server exposes:  curl -s http://$HOST_IP:<port>/v1/models | jq"
else
    echo
    echo "No model servers are running. Start one with: llama-start <name>|all"
fi
command -v tailscale >/dev/null 2>&1 && echo "Tailscale IP: $(tailscale ip -4 2>/dev/null | head -1)" || true

if [[ -f /var/run/reboot-required ]]; then
    echo
    echo "*** A reboot is required to finish driver/kernel updates: sudo reboot ***"
fi
echo
