#!/usr/bin/env bash
set -euo pipefail
# By: Christopher Gray
# Version: 0.0.29
# Updated: 5/24/2026
#  curl -fsSL https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/strix-halo-ai-host-tune.sh -o strix-halo-ai-host-tune.sh && bash strix-halo-ai-host-tune.sh
#
#
# Tune Ubuntu 26.04+ on AMD Strix Halo / Ryzen AI Max systems for AI + Docker workloads.
#   Then install AI runtimes (Ollama, vLLM, llama.cpp) and web UIs (OpenWebUI).
#
# Usage:
#   sudo bash strix-halo-ai-host-tune.sh 96
#   sudo bash strix-halo-ai-host-tune.sh 104
#   sudo bash strix-halo-ai-host-tune.sh 112
#   sudo bash strix-halo-ai-host-tune.sh 120
# 
# Recommended starting value for a 128 GB box running Docker + databases:
#   96
#
# IMPORTANT:
#   This script cannot change BIOS UMA/iGPU/VRAM allocation.
#   In BIOS, set UMA/iGPU/VRAM/Frame Buffer to 512M, 1G, Auto, or Low.
#   Do NOT leave it at 96G if you want Linux to see most of the 128 GB.
#
# BIOS STEPS:
    # BIOS VRAM / UMA setting for Beelink Strix Halo / AMI Aptio:

    # 1. Reboot and enter BIOS/UEFI.
    # 2. Go to: Advanced → AMD CBS → NBIO Common Options → GFX Configuration.
    # 3. Select: Dedicated Graphics Memory.
    # 4. Change from 96G to 4G.
    # - For Linux AI + Docker server use, 4G dedicated VRAM is a good starting point.
    # - This leaves about 124 GB as system RAM.
    # 5. Press F4 to Save & Exit.
    # 6. Boot Ubuntu and verify:

    # free -h
    # sudo dmesg | grep -Ei "VRAM|GTT|amdgpu.*memory|HMM registered"

    # Expected result after BIOS change:

    # System RAM: about 120–124 GiB
    # AMDGPU VRAM: about 4096M


# This was designed to work on a server that wants to run the following:
#        Docker
#        Ollama / vLLM / llama.cpp - OpenWebUI
#        nginx
#        MongoDB
#        ClickHouse
#        Qdrant
#        Redis
#        other containers
#        Linux page cache
#        model download/cache space



GTT_GIB="${1:-96}"

if [[ "$EUID" -ne 0 ]]; then
  echo "Run as root: sudo bash $0 ${GTT_GIB}"
  exit 1
fi

# ── AI runtime install switches ───────────────────────────────────────────────
INSTALL_OLLAMA=true    # Ollama native systemd service (recommended default)
INSTALL_VLLM=false     # vLLM via Docker ROCm image (OpenAI-compatible API on :8000)
INSTALL_LLAMACPP=false # llama.cpp built from source with HIP/ROCm
# OpenWebUI is always installed — works with Ollama or any OpenAI-compatible backend.
# ─────────────────────────────────────────────────────────────────────────────

echo "=== Memory / GPU diagnostics ==="
free -h
dmesg | grep -Ei "VRAM|GTT|amdgpu.*memory|HMM registered"
cat /sys/module/ttm/parameters/pages_limit
cat /sys/module/ttm/parameters/page_pool_size
echo "================================"

case "$GTT_GIB" in
  64|80|96|104|108|112|120) ;;
  *)
    echo "Refusing unusual GTT size: ${GTT_GIB} GiB"
    echo "Use one of: 64, 80, 96, 104, 108, 112, 120"
    exit 1
    ;;
esac

PAGES=$(( GTT_GIB * 1024 * 1024 / 4 ))

echo "Configuring AI host tuning for AMD Strix Halo / Ryzen AI Max"
echo "Requested GTT/TTM pool: ${GTT_GIB} GiB"
echo "TTM pages: ${PAGES}"

echo
echo "Current memory/GPU state:"
free -h || true
dmesg | grep -Ei "VRAM|GTT|amdgpu.*memory" | tail -n 20 || true
echo

echo "Checking kernel..."
uname -a
KERNEL_MAJOR="$(uname -r | cut -d. -f1)"
KERNEL_MINOR="$(uname -r | cut -d. -f2)"
if (( KERNEL_MAJOR < 6 )) || { (( KERNEL_MAJOR == 6 )) && (( KERNEL_MINOR < 14 )); }; then
  echo "WARNING: Kernel appears older than 6.14. Ryzen AI Max may need a newer kernel."
fi

echo
echo "Installing useful host packages..."
apt-get update
apt-get install -y \
  curl \
  wget \
  git \
  jq \
  htop \
  nvtop \
  numactl \
  pciutils \
  usbutils \
  lsb-release \
  ca-certificates \
  gnupg \
  docker.io \
  docker-compose-v2

echo
echo "Enabling Docker..."
systemctl enable --now docker

echo
echo "Adding primary sudo user to docker/video/render groups where possible..."
# Prefer the user who invoked sudo; fallback to ubuntu if it exists.
TARGET_USER="${SUDO_USER:-}"
if [[ -z "$TARGET_USER" || "$TARGET_USER" == "root" ]]; then
  if id ubuntu >/dev/null 2>&1; then
    TARGET_USER="ubuntu"
  else
    TARGET_USER=""
  fi
fi

if [[ -n "$TARGET_USER" ]]; then
  groupadd -f render || true
  groupadd -f video || true
  usermod -aG docker,video,render "$TARGET_USER"
  echo "Added ${TARGET_USER} to docker, video, render groups."
  echo "You must log out/in or reboot for group membership to fully apply."
else
  echo "No non-root target user found; skipping usermod."
fi

echo
echo "Configuring udev rule for /dev/kfd (ROCm GPU compute access)..."
cat > /etc/udev/rules.d/70-kfd.rules <<'EOF'
SUBSYSTEM=="kfd", GROUP="render", MODE="0660"
EOF
udevadm control --reload-rules && udevadm trigger || true

echo
echo "Configuring Docker daemon defaults..."
mkdir -p /etc/docker

if [[ -f /etc/docker/daemon.json ]]; then
  cp -a /etc/docker/daemon.json "/etc/docker/daemon.json.bak.$(date +%Y%m%d-%H%M%S)"
fi

cat > /etc/docker/daemon.json <<'JSON'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "5"
  },
  "default-shm-size": "16G",
  "storage-driver": "overlay2"
}
JSON

systemctl restart docker

echo
echo "Installing ROCm 6.4 (first release with official Strix Halo / gfx1151 support)..."
ROCM_VER="6.4.1"
AMDGPU_PKG_VER="6.4.60401-1"
UBUNTU_CS="$(lsb_release -cs)"
DEB_BASE="https://repo.radeon.com/amdgpu-install/${ROCM_VER}/ubuntu"
DEB_NAME="amdgpu-install_${AMDGPU_PKG_VER}_all.deb"
if wget -q --spider "${DEB_BASE}/${UBUNTU_CS}/${DEB_NAME}" 2>/dev/null; then
  DEB_URL="${DEB_BASE}/${UBUNTU_CS}/${DEB_NAME}"
else
  echo "  No ROCm installer for ${UBUNTU_CS}; falling back to noble build."
  DEB_URL="${DEB_BASE}/noble/${DEB_NAME}"
fi
wget -q -O /tmp/amdgpu-install.deb "$DEB_URL"
dpkg -i /tmp/amdgpu-install.deb
apt-get update -qq
# --no-dkms: Ubuntu 26.04 ships amdgpu in-kernel; skip DKMS module build.
amdgpu-install -y --usecase=rocm --no-dkms
rm -f /tmp/amdgpu-install.deb

echo
echo "Setting HSA_OVERRIDE_GFX_VERSION for Strix Halo (gfx1151)..."
# ROCm 6.4+ officially supports gfx1151; this override is a fallback for
# container images or tools that ship older ROCm versions.
sed -i '/^HSA_OVERRIDE_GFX_VERSION=/d' /etc/environment
echo 'HSA_OVERRIDE_GFX_VERSION=11.0.0' >> /etc/environment
echo "  HSA_OVERRIDE_GFX_VERSION=11.0.0 written to /etc/environment"

if [[ "$INSTALL_OLLAMA" == "true" ]]; then
  echo
  echo "Installing Ollama (ROCm is already installed; Ollama installer auto-detects it)..."
  curl -fsSL https://ollama.com/install.sh | sh

  echo
  echo "Configuring Ollama systemd service for Strix Halo..."
  mkdir -p /etc/systemd/system/ollama.service.d
  cat > /etc/systemd/system/ollama.service.d/override.conf <<'EOF'
[Service]
# Listen on all interfaces so Docker containers and LAN clients can reach it.
Environment="OLLAMA_HOST=0.0.0.0"
# gfx1151 fallback — harmless with ROCm 6.4+, required for older tooling.
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
EOF
  systemctl daemon-reload
  systemctl enable --now ollama
  echo "  Ollama enabled and started on 0.0.0.0:11434"
fi

if [[ "$INSTALL_VLLM" == "true" ]]; then
  echo
  echo "Pulling vLLM ROCm Docker image..."
  docker pull vllm/vllm-rocm:latest
  echo "  vLLM image ready. See /opt/ai/compose.vllm.yaml for usage."
fi

if [[ "$INSTALL_LLAMACPP" == "true" ]]; then
  echo
  echo "Building llama.cpp with HIP/ROCm support..."
  apt-get install -y cmake build-essential libcurl4-openssl-dev
  git clone --depth=1 https://github.com/ggerganov/llama.cpp /opt/ai/llama.cpp-src
  cmake -S /opt/ai/llama.cpp-src -B /opt/ai/llama.cpp-src/build \
    -DGGML_HIPBLAS=ON \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build /opt/ai/llama.cpp-src/build --parallel "$(nproc)"
  cmake --install /opt/ai/llama.cpp-src/build --prefix /opt/ai/llama.cpp
  echo "  llama.cpp installed to /opt/ai/llama.cpp/bin/"
  echo "  Start server: /opt/ai/llama.cpp/bin/llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 8080 -ngl 99"
fi

echo
echo "Configuring sysctl values for AI + database containers..."
cat > /etc/sysctl.d/99-ai-host.conf <<'EOF'
# AI/database host tuning

# ClickHouse / MongoDB / Qdrant / Redis can all use many mappings/files.
vm.max_map_count = 1048576
fs.file-max = 2097152

# Keep swap as emergency headroom but avoid using it aggressively.
vm.swappiness = 10

# Better for large RAM machines; avoid premature cache pressure.
vm.vfs_cache_pressure = 50

# Network backlog for nginx/proxy/service containers.
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 250000

# Hugepages reduce TLB pressure for large model inference (vLLM, llama.cpp).
vm.nr_hugepages = 1024
EOF

sysctl --system >/dev/null

echo
echo "Configuring user limits..."
cat > /etc/security/limits.d/99-ai-host.conf <<'EOF'
* soft nofile 1048576
* hard nofile 1048576
* soft memlock unlimited
* hard memlock unlimited
root soft nofile 1048576
root hard nofile 1048576
root soft memlock unlimited
root hard memlock unlimited
EOF

echo
echo "Configuring systemd default limits..."
mkdir -p /etc/systemd/system.conf.d /etc/systemd/user.conf.d

cat > /etc/systemd/system.conf.d/99-ai-host-limits.conf <<'EOF'
[Manager]
DefaultLimitNOFILE=1048576
DefaultLimitMEMLOCK=infinity
EOF

cat > /etc/systemd/user.conf.d/99-ai-host-limits.conf <<'EOF'
[Manager]
DefaultLimitNOFILE=1048576
DefaultLimitMEMLOCK=infinity
EOF

systemctl daemon-reexec

echo
echo "Configuring GRUB kernel parameters for TTM/GTT..."
if [[ ! -f /etc/default/grub ]]; then
  echo "ERROR: /etc/default/grub not found."
  exit 1
fi

cp -a /etc/default/grub "/etc/default/grub.bak.$(date +%Y%m%d-%H%M%S)"

# Remove older/conflicting TTM/amdgpu GTT params from GRUB_CMDLINE_LINUX_DEFAULT, then add desired ones.
python3 - "$PAGES" "$GTT_GIB" <<'PY'
import re
import shlex
import sys
from pathlib import Path

pages = sys.argv[1]
gtt_mib = int(sys.argv[2]) * 1024
path = Path("/etc/default/grub")
text = path.read_text()

line_re = re.compile(r'^(GRUB_CMDLINE_LINUX_DEFAULT=)(["\'])(.*?)(\2)\s*$', re.M)

def clean_args(s: str):
    try:
        args = shlex.split(s)
    except Exception:
        args = s.split()

    remove_prefixes = (
        "ttm.pages_limit=",
        "ttm.page_pool_size=",
        "amdgpu.gttsize=",
        "amdttm.pages_limit=",
        "amdttm.page_pool_size=",
    )
    kept = []
    for a in args:
        if any(a.startswith(p) for p in remove_prefixes):
            continue
        kept.append(a)

    kept.append(f"ttm.pages_limit={pages}")
    kept.append(f"ttm.page_pool_size={pages}")
    kept.append(f"amdgpu.gttsize={gtt_mib}")
    return " ".join(kept)

m = line_re.search(text)
if not m:
    text += f'\nGRUB_CMDLINE_LINUX_DEFAULT="quiet splash ttm.pages_limit={pages} ttm.page_pool_size={pages} amdgpu.gttsize={gtt_mib}"\n'
else:
    new_args = clean_args(m.group(3))
    text = line_re.sub(lambda mm: f'{mm.group(1)}"{new_args}"', text, count=1)

path.write_text(text)
PY

update-grub

echo
echo "Setting TTM pages limit at runtime..."
TTM_PAGES=25165824
for param in pages_limit page_pool_size; do
  sysfs="/sys/module/ttm/parameters/${param}"
  if [[ -w "$sysfs" ]]; then
    echo "$TTM_PAGES" > "$sysfs"
    echo "  ${param} = ${TTM_PAGES}"
  else
    echo "  ${sysfs} not writable (ttm module may not be loaded yet — value takes effect at next boot via GRUB)"
  fi
done

echo
echo "Creating a ROCm Docker Compose snippet at /opt/ai/compose.rocm-example.yaml ..."
mkdir -p /opt/ai

cat > /opt/ai/compose.rocm-example.yaml <<'YAML'
# Example ROCm-capable service.
# Copy this pattern into your vLLM / llama.cpp / custom containers.
#
# Notes:
# - /dev/kfd is the ROCm compute interface.
# - /dev/dri exposes AMD render nodes.
# - seccomp=unconfined is commonly needed for ROCm containers.
# - ipc: host and large shm_size help inference workloads.

services:
  rocm-test:
    image: rocm/dev-ubuntu-24.04:latest
    container_name: rocm-test
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
    security_opt:
      - seccomp=unconfined
    ipc: host
    shm_size: 16g
    network_mode: host
    command: /bin/bash
    tty: true
    stdin_open: true
    environment:
      # May not be needed with ROCm 6.4+ (native gfx1151 support); keeps older images working.
      - HSA_OVERRIDE_GFX_VERSION=11.0.0
YAML

echo
echo "Creating OpenWebUI + Ollama compose at /opt/ai/compose.ollama-webui.yaml ..."
cat > /opt/ai/compose.ollama-webui.yaml <<'YAML'
# OpenWebUI connecting to native Ollama running on the host (systemd service).
# Ollama must be running: systemctl status ollama
#
# Start: docker compose -f /opt/ai/compose.ollama-webui.yaml up -d
# UI at: http://<host-ip>:3000

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    restart: unless-stopped
    ports:
      - "3000:8080"
    volumes:
      - open-webui-data:/app/backend/data
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      # Points to native Ollama on the host via the bridge gateway address.
      - OLLAMA_BASE_URL=http://host.docker.internal:11434

volumes:
  open-webui-data:
YAML

echo
echo "Creating vLLM ROCm compose at /opt/ai/compose.vllm.yaml ..."
cat > /opt/ai/compose.vllm.yaml <<'YAML'
# vLLM with ROCm — OpenAI-compatible inference API.
# Set INSTALL_VLLM=true in strix-halo-ai-host-tune.sh to pre-pull the image, or let
# Docker pull it on first start.
#
# Edit the --model arg before running. HuggingFace token required for gated models.
# Start: docker compose -f /opt/ai/compose.vllm.yaml up -d
# API at: http://<host-ip>:8000/v1

services:
  vllm:
    image: vllm/vllm-rocm:latest
    container_name: vllm
    restart: unless-stopped
    ports:
      - "8000:8000"
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
    security_opt:
      - seccomp=unconfined
    ipc: host
    shm_size: 32g
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HSA_OVERRIDE_GFX_VERSION=11.0.0
      # - HUGGING_FACE_HUB_TOKEN=your_token_here
    command: >
      --model meta-llama/Llama-3.1-8B-Instruct
      --host 0.0.0.0
      --port 8000
      --dtype float16
      --max-model-len 8192
YAML

echo
echo "Creating verification script at /usr/local/sbin/ai-host-check ..."
cat > /usr/local/sbin/ai-host-check <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "=== Kernel ==="
uname -a
echo

echo "=== Kernel cmdline ==="
cat /proc/cmdline
echo

echo "=== RAM ==="
free -h
echo

echo "=== TTM params ==="
if [[ -r /sys/module/ttm/parameters/pages_limit ]]; then
  echo -n "pages_limit: "
  cat /sys/module/ttm/parameters/pages_limit
fi
if [[ -r /sys/module/ttm/parameters/page_pool_size ]]; then
  echo -n "page_pool_size: "
  cat /sys/module/ttm/parameters/page_pool_size
fi
echo

echo "=== AMDGPU memory ==="
dmesg | grep -Ei "VRAM|GTT|amdgpu.*memory|HMM registered" | tail -n 50 || true
echo

echo "=== GPU devices ==="
ls -l /dev/kfd /dev/dri 2>/dev/null || true
echo

echo "=== Groups ==="
id "${SUDO_USER:-$USER}" || true
echo

echo "=== Docker ==="
docker version --format '{{.Server.Version}}' 2>/dev/null || echo "Docker not responding"
docker compose version 2>/dev/null || true
echo

echo "=== Sysctl ==="
sysctl vm.max_map_count fs.file-max vm.swappiness vm.vfs_cache_pressure net.core.somaxconn net.ipv4.tcp_max_syn_backlog vm.nr_hugepages 2>/dev/null || true
echo

echo "=== ROCm ==="
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo | grep -E "Name:|Marketing Name:|gfx|HSA" | head -n 20 || true
else
  echo "rocminfo not found (ROCm may not be installed or PATH not updated yet)"
fi
echo

echo "=== HSA override ==="
grep HSA_OVERRIDE /etc/environment 2>/dev/null || echo "(not set in /etc/environment)"
echo

echo "=== Ollama ==="
if systemctl is-active --quiet ollama 2>/dev/null; then
  echo "ollama service: running"
  curl -s http://localhost:11434/api/tags 2>/dev/null | \
    python3 -c "import json,sys; d=json.load(sys.stdin); models=[m['name'] for m in d.get('models',[])]; print('models:', models if models else '(none pulled yet)')" \
    2>/dev/null || echo "  (could not query Ollama API)"
else
  echo "ollama service: not running"
fi
echo

echo "=== vLLM ==="
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^vllm$'; then
  echo "vllm container: running"
  curl -s http://localhost:8000/health 2>/dev/null && echo "  API: healthy" || echo "  API: not responding"
else
  echo "vllm container: not running (see /opt/ai/compose.vllm.yaml)"
fi
echo

echo "=== llama.cpp ==="
if [[ -x /opt/ai/llama.cpp/bin/llama-server ]]; then
  echo -n "llama-server: "
  /opt/ai/llama.cpp/bin/llama-server --version 2>/dev/null || echo "installed"
else
  echo "llama-server: not installed"
fi
echo
EOF

chmod +x /usr/local/sbin/ai-host-check

echo
echo "Done."
echo
echo "NEXT REQUIRED STEP:"
echo "1. Reboot into BIOS."
echo "2. Set UMA/iGPU/VRAM/Frame Buffer to 512M, 1G, Auto, or Low."
echo "3. Boot Ubuntu."
echo "4. Run: ai-host-check"
echo
echo "After reboot you want something like:"
echo "  Mem: close to 120 GiB+ total"
echo "  VRAM: 512M or 1024M"
echo "  GTT: around ${GTT_GIB} GiB"
echo
echo "Reboot required."
