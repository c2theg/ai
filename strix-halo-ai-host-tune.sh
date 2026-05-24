#!/usr/bin/env bash
set -euo pipefail

# ai-host-tune.sh
# Tune Ubuntu 26.04+ on AMD Strix Halo / Ryzen AI Max systems for AI + Docker workloads.
#
# Usage:
#   sudo bash ai-host-tune.sh 96
#   sudo bash ai-host-tune.sh 104
#   sudo bash ai-host-tune.sh 112
#   sudo bash ai-host-tune.sh 120
# 
# Recommended starting value for a 128 GB box running Docker + databases:
#   96
#
# IMPORTANT:
#   This script cannot change BIOS UMA/iGPU/VRAM allocation.
#   In BIOS, set UMA/iGPU/VRAM/Frame Buffer to 512M, 1G, Auto, or Low.
#   Do NOT leave it at 96G if you want Linux to see most of the 128 GB.

GTT_GIB="${1:-96}"

if [[ "$EUID" -ne 0 ]]; then
  echo "Run as root: sudo bash $0 ${GTT_GIB}"
  exit 1
fi

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
python3 - "$PAGES" <<'PY'
import re
import shlex
import sys
from pathlib import Path

pages = sys.argv[1]
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
    return " ".join(kept)

m = line_re.search(text)
if not m:
    text += f'\nGRUB_CMDLINE_LINUX_DEFAULT="quiet splash ttm.pages_limit={pages} ttm.page_pool_size={pages}"\n'
else:
    new_args = clean_args(m.group(3))
    text = line_re.sub(lambda mm: f'{mm.group(1)}"{new_args}"', text, count=1)

path.write_text(text)
PY

update-grub

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
sysctl vm.max_map_count fs.file-max vm.swappiness vm.vfs_cache_pressure net.core.somaxconn net.ipv4.tcp_max_syn_backlog 2>/dev/null || true
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
