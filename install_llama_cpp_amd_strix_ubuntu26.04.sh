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


Version:  0.0.7
Last Updated:  9/24/2026


"
# ============================================================
# llama.cpp installer for AMD Strix Halo / Ubuntu 26.04
#
# Uses Vulkan/RADV for GPU acceleration.
#
# Models:
#   /home/ubuntu/models/*.gguf
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

echo "[1/8] Updating Ubuntu..."
sudo apt-get update
sudo apt-get -y upgrade

# ------------------------------------------------------------
# Install dependencies
# ------------------------------------------------------------

echo
echo "[2/8] Installing build dependencies..."

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
    ca-certificates

# ------------------------------------------------------------
# Verify Vulkan
# ------------------------------------------------------------

echo
echo "[3/8] Checking Vulkan..."

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
echo "[4/8] Creating model directory..."

mkdir -p "$MODEL_DIR"

# ------------------------------------------------------------
# Clone/update llama.cpp
# ------------------------------------------------------------

echo
echo "[5/8] Getting latest llama.cpp..."

if [[ -d "$LLAMA_DIR/.git" ]]; then
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
echo "[6/8] Building llama.cpp with Vulkan..."

cd "$LLAMA_DIR"

rm -rf build

cmake -S . -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON

cmake --build build --parallel

# ------------------------------------------------------------
# Install convenient symlinks
# ------------------------------------------------------------

echo
echo "[7/8] Installing commands..."

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

MODEL_DIR="/home/ubuntu/models"

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

MODEL_DIR="/home/ubuntu/models"

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
echo "  URL:   http://127.0.0.1:8080"
echo

exec llama-server \
    -m "$MODEL" \
    --alias primary \
    -ngl 999 \
    -c 32768 \
    -t "$(nproc)" \
    --host 127.0.0.1 \
    --port 10020 \
    --jinja
EOF

chmod +x "$HOME/llama-server.sh"

# ------------------------------------------------------------
# Finished
# ------------------------------------------------------------

echo
echo "[8/8] Installation complete!"
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
echo "Server:"
echo "  http://127.0.0.1:10020"
echo
echo "----------------------------------------------"
echo "Examples: run a server on a port with a model alias"
echo "----------------------------------------------"
echo
echo "Qwen3.8-27B on port 10020, alias \"qwen-27b\":"
echo "  llama-server -m $MODEL_DIR/Qwen3.8-27B-UD-Q4_K_XL.gguf --alias qwen-27b --port 10020 -ngl 999 -c 32768 --jinja"
echo
echo "Qwen3.5-35B-A3B on port 10021, alias \"qwen-35b-a3b\":"
echo "  llama-server -m $MODEL_DIR/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --alias qwen-35b-a3b --port 10021 -ngl 999 -c 32768 --jinja"
echo
echo "Qwen3.5-9B on port 10022, alias \"qwen-9b\":"
echo "  llama-server -m $MODEL_DIR/Qwen3.5-9B-UD-Q4_K_XL.gguf --alias qwen-9b --port 10022 -ngl 999 -c 32768 --jinja"
echo
echo "Both at once, on all interfaces (LAN/Tailscale), several aliases on one:"
echo "  llama-server -m $MODEL_DIR/Qwen3.8-27B-UD-Q4_K_XL.gguf --alias qwen-27b,primary --host 0.0.0.0 --port 10020 -ngl 999 -c 32768 --jinja &"
echo "  llama-server -m $MODEL_DIR/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --alias qwen-35b-a3b --host 0.0.0.0 --port 10021 -ngl 999 -c 32768 --jinja &"
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
