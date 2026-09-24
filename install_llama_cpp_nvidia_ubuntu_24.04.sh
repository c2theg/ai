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

# ============================================================
# llama.cpp CUDA installer
# Ubuntu 24.04
# Intel x86_64 CPU + NVIDIA RTX 5080
#
# Installs:
#   - Build dependencies
#   - CUDA Toolkit if nvcc is missing
#   - llama.cpp
#   - CUDA-enabled llama-cli / llama-server / llama-bench
#
# Default install:
#   Source:  /opt/llama.cpp
#   Binaries: /usr/local/bin/
#
# Usage:
#   chmod +x install-llamacpp.sh
#   sudo ./install-llamacpp.sh
# ============================================================

set -Eeuo pipefail

LLAMA_DIR="/opt/llama.cpp"
LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"

echo "============================================================"
echo " llama.cpp CUDA Installer"
echo " Ubuntu 24.04 + Intel CPU + NVIDIA RTX 5080"
echo "============================================================"
echo

# ------------------------------------------------------------
# Must run as root
# ------------------------------------------------------------

if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: Run this script with sudo:"
    echo
    echo "  sudo $0"
    echo
    exit 1
fi

# ------------------------------------------------------------
# Check architecture
# ------------------------------------------------------------

ARCH="$(uname -m)"

if [[ "${ARCH}" != "x86_64" ]]; then
    echo "WARNING: Expected x86_64 but found ${ARCH}"
fi

echo "[1/10] System information"
echo "------------------------------------------------------------"
echo "OS:"
grep PRETTY_NAME /etc/os-release || true

echo
echo "Kernel:"
uname -r

echo
echo "CPU:"
lscpu | grep "Model name" | head -1 || true

echo

# ------------------------------------------------------------
# Check NVIDIA driver
# ------------------------------------------------------------

echo "[2/10] Checking NVIDIA GPU / driver"
echo "------------------------------------------------------------"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "ERROR: nvidia-smi was not found."
    echo
    echo "Install a recent NVIDIA RTX 5080-compatible driver first."
    echo
    echo "Ubuntu can normally detect the recommended driver with:"
    echo
    echo "  ubuntu-drivers devices"
    echo
    echo "Then install it with:"
    echo
    echo "  sudo ubuntu-drivers install"
    echo
    echo "Reboot afterward:"
    echo
    echo "  sudo reboot"
    echo
    exit 1
fi

nvidia-smi

echo

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

echo "Detected GPU: ${GPU_NAME}"

if [[ "${GPU_NAME}" != *"RTX 5080"* ]]; then
    echo
    echo "NOTE: RTX 5080 was expected, but detected:"
    echo "      ${GPU_NAME}"
    echo
    echo "Continuing anyway."
fi

# ------------------------------------------------------------
# Install packages
# ------------------------------------------------------------

echo
echo "[3/10] Installing build dependencies"
echo "------------------------------------------------------------"

export DEBIAN_FRONTEND=noninteractive

apt-get update

apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    cmake \
    ninja-build \
    ccache \
    curl \
    wget \
    pkg-config \
    libssl-dev \
    libcurl4-openssl-dev \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    gnupg

# ------------------------------------------------------------
# CUDA toolkit
# ------------------------------------------------------------

echo
echo "[4/10] Checking CUDA Toolkit"
echo "------------------------------------------------------------"

if command -v nvcc >/dev/null 2>&1; then

    echo "CUDA compiler already installed:"
    nvcc --version

else

    echo "nvcc not found."
    echo
    echo "Installing NVIDIA CUDA repository and CUDA Toolkit..."

    CUDA_KEYRING="/tmp/cuda-keyring.deb"

    wget -q \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        -O "${CUDA_KEYRING}"

    dpkg -i "${CUDA_KEYRING}"

    apt-get update

    # Install toolkit only.
    #
    # Deliberately avoid installing the "cuda" meta-package because
    # it can also alter the NVIDIA driver installation.
    apt-get install -y cuda-toolkit

fi

# ------------------------------------------------------------
# Locate CUDA
# ------------------------------------------------------------

echo
echo "[5/10] Configuring CUDA environment"
echo "------------------------------------------------------------"

if [[ -x /usr/local/cuda/bin/nvcc ]]; then

    CUDA_HOME="/usr/local/cuda"

else

    NVCC_PATH="$(command -v nvcc || true)"

    if [[ -z "${NVCC_PATH}" ]]; then
        echo "ERROR: CUDA installation completed but nvcc is unavailable."
        exit 1
    fi

    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "${NVCC_PATH}")")")"

fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME=${CUDA_HOME}"

nvcc --version

# Persist CUDA PATH

cat >/etc/profile.d/cuda.sh <<EOF
export CUDA_HOME="${CUDA_HOME}"
export PATH="\${CUDA_HOME}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}"
EOF

chmod 644 /etc/profile.d/cuda.sh

# ------------------------------------------------------------
# Clone llama.cpp
# ------------------------------------------------------------

echo
echo "[6/10] Downloading llama.cpp"
echo "------------------------------------------------------------"

if [[ -d "${LLAMA_DIR}/.git" ]]; then

    echo "Existing llama.cpp checkout found."
    echo "Updating..."

    git -C "${LLAMA_DIR}" fetch --all --tags
    git -C "${LLAMA_DIR}" reset --hard origin/master

else

    rm -rf "${LLAMA_DIR}"

    git clone \
        --depth 1 \
        "${LLAMA_REPO}" \
        "${LLAMA_DIR}"

fi

cd "${LLAMA_DIR}"

echo
echo "llama.cpp commit:"
git log -1 --oneline

# ------------------------------------------------------------
# Detect CPU cores
# ------------------------------------------------------------

CPU_THREADS="$(nproc)"

echo
echo "CPU build threads: ${CPU_THREADS}"

# ------------------------------------------------------------
# Configure build
# ------------------------------------------------------------

echo
echo "[7/10] Configuring CUDA build"
echo "------------------------------------------------------------"

rm -rf build

#
# GGML_CUDA=ON
#     Enables NVIDIA CUDA acceleration.
#
# GGML_NATIVE=ON
#     Optimizes CPU portions for this specific Intel CPU.
#
# CMAKE_BUILD_TYPE=Release
#     Enables optimized build.
#

cmake \
    -S . \
    -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=ON

# ------------------------------------------------------------
# Compile
# ------------------------------------------------------------

echo
echo "[8/10] Compiling llama.cpp"
echo "------------------------------------------------------------"

cmake \
    --build build \
    --config Release \
    --parallel "${CPU_THREADS}"

# ------------------------------------------------------------
# Install convenient links
# ------------------------------------------------------------

echo
echo "[9/10] Installing command symlinks"
echo "------------------------------------------------------------"

BIN_DIR="${LLAMA_DIR}/build/bin"

COMMANDS=(
    llama-cli
    llama-server
    llama-bench
    llama-quantize
    llama-perplexity
)

for CMD in "${COMMANDS[@]}"; do

    if [[ -x "${BIN_DIR}/${CMD}" ]]; then

        ln -sf \
            "${BIN_DIR}/${CMD}" \
            "/usr/local/bin/${CMD}"

        echo "Installed: /usr/local/bin/${CMD}"

    fi

done

# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

echo
echo "[10/10] Verifying installation"
echo "------------------------------------------------------------"

echo
echo "llama-cli:"
llama-cli --version || true

echo
echo "CUDA devices visible to llama.cpp:"
echo

llama-cli --list-devices || true

echo
echo "NVIDIA GPU:"
nvidia-smi \
    --query-gpu=name,driver_version,memory.total \
    --format=csv

echo
echo
echo "============================================================"
echo " llama.cpp installation complete"
echo "============================================================"
echo
echo "Source:"
echo "  ${LLAMA_DIR}"
echo
echo "Executables:"
echo "  llama-cli"
echo "  llama-server"
echo "  llama-bench"
echo "  llama-quantize"
echo
echo "Verify GPU:"
echo
echo "  llama-cli --list-devices"
echo
echo "Run a model:"
echo
echo "  llama-cli \\"
echo "      -m /path/to/model.gguf \\"
echo "      -ngl 99 \\"
echo "      -c 8192 \\"
echo "      -p \"Explain how TCP congestion control works.\""
echo
echo "Start OpenAI-compatible server:"
echo
echo "  llama-server \\"
echo "      -m /path/to/model.gguf \\"
echo "      -ngl 99 \\"
echo "      -c 32768 \\"
echo "      --host 0.0.0.0 \\"
echo "      --port 8080"
echo
echo "Start a server with a model name alias (--alias):"
echo "  Clients request the alias as the \"model\" name instead of the file name."
echo
echo "  llama-server \\"
echo "      -m /models/Qwen3-14B-Q4_K_M.gguf \\"
echo "      --alias qwen3-14b \\"
echo "      -ngl 99 \\"
echo "      -c 16384 \\"
echo "      --host 0.0.0.0 \\"
echo "      --port 8001"
echo
echo "Run several models side by side (one port each, share the GPU):"
echo
echo "  llama-server -m /models/chat.gguf  --alias chat-model  -ngl 99 -c 16384 --port 8001 &"
echo "  llama-server -m /models/embed.gguf --alias embed-model -ngl 99 --embeddings --port 8002 &"
echo
echo "Test the server:"
echo
echo "  curl http://localhost:8001/v1/models"
echo
echo "  curl http://localhost:8001/v1/chat/completions \\"
echo "      -H \"Content-Type: application/json\" \\"
echo "      -d '{\"model\": \"qwen3-14b\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo
echo "Run the server in the background and log to a file:"
echo
echo "  nohup llama-server -m /models/model.gguf --alias my-model -ngl 99 --port 8001 \\"
echo "      > /var/log/llama-server-8001.log 2>&1 &"
echo
echo "Benchmark:"
echo
echo "  llama-bench \\"
echo "      -m /path/to/model.gguf \\"
echo "      -ngl 99"
echo
echo "Watch GPU usage:"
echo
echo "  watch -n 1 nvidia-smi"
echo
