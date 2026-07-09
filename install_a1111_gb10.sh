#!/usr/bin/env bash
#    By: Christopher Gray
#    Version: 0.0.7
#    Updated: 7/9/2026
#
#   Installer:
#     curl -sSL https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_a1111_gb10.sh | bash
#
#   0.0.7: works when run detached (curl | bash) — guards unset BASH_SOURCE,
#          fetches the Dockerfile itself, and fixes the apt env-prefix bug.
#
# install_a1111_gb10.sh — one-shot installer for Automatic1111 (Stable Diffusion
# WebUI) on an NVIDIA DGX Spark / GB10 (arm64 + Blackwell), reachable over the LAN.
#
# What it does, idempotently (safe to re-run — a re-run UPDATES to the latest):
#   1. Preflight: confirm arm64 + an NVIDIA driver (nvidia-smi).
#   2. Docker: install if missing.
#   3. NVIDIA Container Toolkit: install + wire into Docker if the GPU isn't yet
#      visible to containers.
#   4. Prove a container can see the GPU.
#   5. Open the LAN firewall port (ufw) if a firewall is active.
#   6. Build the A1111 image with a Blackwell-capable torch (the crux — A1111's
#      pinned torch is x86 CUDA-12.1 and has no arm64/Blackwell build).
#   7. (Re)start the container bound to 0.0.0.0 so any LAN host can reach it.
#   8. Verify the API is up and the GPU is actually being used.
#
# Re-running always rebuilds with --pull, so it picks up a newer base image /
# newer synced Dockerfile automatically. The A1111 app version itself is pinned
# in Dockerfile.automatic1111 (git checkout vX.Y.Z); bump that to move it.
#
# Everything below is overridable via environment variables (see the block).
set -euo pipefail

# ─────────────────────────────── configuration ───────────────────────────────
IMAGE_NAME="${A1111_IMAGE_NAME:-a1111-gb10}"
IMAGE_TAG="${A1111_IMAGE_TAG:-latest}"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
CONTAINER_NAME="${A1111_CONTAINER_NAME:-a1111-gb10}"
PORT="${A1111_PORT:-7860}"

# arm64 CUDA base (Docker auto-selects the arm64/sbsa variant on the GB10).
BASE_IMAGE="${A1111_BASE_IMAGE:-nvidia/cuda:12.8.0-runtime-ubuntu22.04}"

# Blackwell-capable torch. cu128 ships arm64 wheels with Blackwell (sm_120)
# kernels. If generation later fails with "no kernel image available for
# execution on the device", GB10 needs newer kernels — re-run with:
#   A1111_TORCH_COMMAND="pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128" ./install_a1111_gb10.sh
TORCH_COMMAND="${A1111_TORCH_COMMAND:-pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128}"

# Optional login. Empty = OPEN to everyone on the LAN. Set A1111_GRADIO_AUTH=user:pass to require a login.
GRADIO_AUTH="${A1111_GRADIO_AUTH:-}"

# Where to fetch build files from when run detached (curl | bash) rather than
# from a checked-out copy of the repo. The Dockerfile has no COPY/ADD, so the
# single file IS the whole build context.
REPO_RAW="${A1111_REPO_RAW:-https://raw.githubusercontent.com/c2theg/ai/refs/heads/main}"
DOCKERFILE_URL="${A1111_DOCKERFILE_URL:-${REPO_RAW}/Dockerfile.automatic1111}"

# Resolve our own location. Under `curl | bash` there is no script file, so
# BASH_SOURCE is unset — guard it (set -u) and fall back to detached mode.
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  SCRIPT_DIR=""
fi

# Build context: prefer a synced checkout (Dockerfile next to this script);
# otherwise a self-managed work dir we fetch the Dockerfile into.
if [ -n "${SCRIPT_DIR}" ] && [ -f "${SCRIPT_DIR}/Dockerfile.automatic1111" ]; then
  BUILD_DIR="${SCRIPT_DIR}"
  FETCH_DOCKERFILE=0
else
  BUILD_DIR="${A1111_WORK_DIR:-${HOME:-/root}/.a1111-gb10}"
  FETCH_DOCKERFILE=1
fi
DATA_DIR="${A1111_DATA_DIR:-${BUILD_DIR}/runtime/automatic1111}"

# ──────────────────────────────── helpers ────────────────────────────────────
c_g=$'\033[32m'; c_y=$'\033[33m'; c_r=$'\033[31m'; c_b=$'\033[1m'; c_0=$'\033[0m'
log()  { printf '%s\n' "${c_g}▶${c_0} $*"; }
warn() { printf '%s\n' "${c_y}⚠${c_0} $*" >&2; }
die()  { printf '%s\n' "${c_r}✖ $*${c_0}" >&2; exit 1; }
step() { printf '\n%s\n' "${c_b}== $* ==${c_0}"; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  command -v sudo >/dev/null 2>&1 || die "Not root and 'sudo' not found. Re-run as root."
  SUDO="sudo"
fi

lan_ip() {
  python3 - <<'PY' 2>/dev/null || true
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80)); print(s.getsockname()[0])
except Exception:
    pass
finally:
    s.close()
PY
}

# ─────────────────────────────── 1. preflight ────────────────────────────────
step "1/8  Preflight"
ARCH="$(uname -m)"
if [ "${ARCH}" != "aarch64" ] && [ "${ARCH}" != "arm64" ]; then
  warn "This machine is '${ARCH}', not arm64. This script targets the DGX Spark / GB10."
  warn "The x86 path is './start_image.sh --mode nvidia'. Continuing anyway in 5s (Ctrl-C to abort)…"
  sleep 5
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  log "NVIDIA driver present:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/    /'
  CUDA_VER="$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}' || true)"
  [ -n "${CUDA_VER:-}" ] && log "Host CUDA (driver) version: ${CUDA_VER}"
else
  die "nvidia-smi not working — the GPU driver isn't loaded.
    On a DGX Spark the driver ships with DGX OS; do NOT install a generic driver
    over it (that can break the box). Verify DGX OS is up to date and the driver
    stack is healthy, then re-run. See NVIDIA's DGX Spark setup docs."
fi

# ─────────────────────────────── 2. docker ───────────────────────────────────
step "2/8  Docker engine"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  log "Docker present: $(docker --version)"
else
  log "Installing Docker via get.docker.com …"
  curl -fsSL https://get.docker.com | ${SUDO} sh || die "Docker install failed."
  ${SUDO} systemctl enable --now docker || true
  docker info >/dev/null 2>&1 || die "Docker installed but the daemon isn't responding."
fi

# ──────────────────────── 3. NVIDIA Container Toolkit ─────────────────────────
step "3/8  NVIDIA Container Toolkit"
if docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia' || \
   docker info 2>/dev/null | grep -qi '"nvidia"'; then
  log "NVIDIA container runtime already wired into Docker."
else
  log "Installing nvidia-container-toolkit …"
  KEYRING=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | ${SUDO} gpg --dearmor -o "${KEYRING}"
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed "s#deb https://#deb [signed-by=${KEYRING}] https://#g" \
    | ${SUDO} tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  ${SUDO} apt-get update
  # `env` sets the var reliably whether ${SUDO} is empty (root) or 'sudo'; a bare
  # `${SUDO} VAR=val cmd` mis-parses VAR=val as the command when ${SUDO} is empty.
  ${SUDO} env DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
  ${SUDO} nvidia-ctk runtime configure --runtime=docker
  ${SUDO} systemctl restart docker
  log "Toolkit installed and Docker restarted."
fi

# ────────────────────────── 4. prove GPU-in-Docker ───────────────────────────
step "4/8  Verify a container can see the GPU"
if ${SUDO} docker run --rm --gpus all "${BASE_IMAGE}" nvidia-smi -L >/tmp/a1111_gpu_check 2>&1; then
  log "GPU visible inside containers:"; sed 's/^/    /' /tmp/a1111_gpu_check
else
  cat /tmp/a1111_gpu_check >&2 || true
  die "A container could not access the GPU with '--gpus all'. Toolkit/runtime issue — see output above."
fi

# ─────────────────────────────── 5. firewall ─────────────────────────────────
step "5/8  LAN firewall"
if command -v ufw >/dev/null 2>&1 && ${SUDO} ufw status 2>/dev/null | grep -qi '^Status: active'; then
  ${SUDO} ufw allow "${PORT}/tcp" >/dev/null 2>&1 || true
  log "Opened ${PORT}/tcp in ufw."
else
  log "No active ufw firewall — nothing to open (port ${PORT} governed by your network)."
fi

# ──────────────────────────────── 6. build ───────────────────────────────────
step "6/8  Build the A1111 image (Blackwell torch) — first build takes a while"
mkdir -p "${BUILD_DIR}"
if [ "${FETCH_DOCKERFILE}" -eq 1 ]; then
  log "Detached run — fetching Dockerfile → ${BUILD_DIR}/Dockerfile.automatic1111"
  curl -fsSL "${DOCKERFILE_URL}" -o "${BUILD_DIR}/Dockerfile.automatic1111" \
    || die "Could not download the Dockerfile from ${DOCKERFILE_URL}
    (set A1111_DOCKERFILE_URL / A1111_REPO_RAW if it lives elsewhere)."
else
  log "Building from synced repo at ${BUILD_DIR}"
fi
cd "${BUILD_DIR}"
[ -f Dockerfile.automatic1111 ] || die "Dockerfile.automatic1111 not found in ${BUILD_DIR}."
log "base image:  ${BASE_IMAGE}"
log "torch:       ${TORCH_COMMAND}"
${SUDO} docker build --pull \
  -f Dockerfile.automatic1111 \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "TORCH_COMMAND=${TORCH_COMMAND}" \
  -t "${IMAGE}" . \
  || die "Image build failed — see the error above (this is where a bad torch wheel or a network issue surfaces)."
log "Built ${IMAGE}."

# ──────────────────────────────── 7. run ─────────────────────────────────────
step "7/8  (Re)start the container"
mkdir -p "${DATA_DIR}/models" "${DATA_DIR}/outputs" "${DATA_DIR}/extensions"
if ${SUDO} docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  log "Removing previous container '${CONTAINER_NAME}' …"
  ${SUDO} docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

extra_args=()
if [ -n "${GRADIO_AUTH}" ]; then
  extra_args+=(--gradio-auth "${GRADIO_AUTH}")
  log "Login required (--gradio-auth set)."
else
  warn "No login configured — anyone on your LAN can use this A1111 (and its script/extension features)."
  warn "Set A1111_GRADIO_AUTH=user:pass and re-run to require a login."
fi

# ENTRYPOINT is [webui.sh -f --listen --api --port 7860]; args here append to it.
# No --use-cpu / --skip-torch-cuda-test: we WANT it to use the Blackwell GPU and
# fail loudly if it can't.
${SUDO} docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --gpus all \
  -p "0.0.0.0:${PORT}:7860" \
  -v "${DATA_DIR}/models:/app/models" \
  -v "${DATA_DIR}/outputs:/app/outputs" \
  -v "${DATA_DIR}/extensions:/app/extensions" \
  "${IMAGE}" ${extra_args[@]+"${extra_args[@]}"} >/dev/null
log "Container started."

# ─────────────────────────────── 8. verify ───────────────────────────────────
step "8/8  Verify (first start downloads the ~4GB default model — allow a few minutes)"
API="http://127.0.0.1:${PORT}/sdapi/v1/sd-models"
ok=0
for i in $(seq 1 60); do
  if curl -fsS "${API}" >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
if [ "${ok}" -eq 1 ]; then
  log "A1111 API is responding."
  if ${SUDO} docker exec "${CONTAINER_NAME}" python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    gpu="$(${SUDO} docker exec "${CONTAINER_NAME}" python3 -c "import torch;print(torch.cuda.get_device_name(0))" 2>/dev/null || true)"
    log "torch sees the GPU ✅  (${gpu:-unknown})"
  else
    warn "API is up but torch.cuda.is_available() is FALSE — it'll run on CPU."
    warn "Check:  ${SUDO:+sudo }docker logs ${CONTAINER_NAME}   (look for a torch/CUDA arch error)"
  fi
else
  warn "API didn't respond within ~5 min. Watch startup with:"
  warn "  ${SUDO:+sudo }docker logs -f ${CONTAINER_NAME}"
fi

# ─────────────────────────────── summary ─────────────────────────────────────
IP="$(lan_ip)"
cat <<EOF

${c_b}Done.${c_0}
  Local:      http://127.0.0.1:${PORT}
$( [ -n "${IP}" ] && printf '  LAN:        http://%s:%s   (use this URL from other hosts)\n' "${IP}" "${PORT}" )
  Logs:       ${SUDO:+sudo }docker logs -f ${CONTAINER_NAME}
  Stop:       ${SUDO:+sudo }docker rm -f ${CONTAINER_NAME}
  Update:     re-run this script (rebuilds with --pull, recreates the container)

${c_b}Point the transcription app at this box${c_0} (on the app host, e.g. ai2-strixhelo):
  In the app's AI Providers page, set the Automatic1111 / image_generation
  connection's base_url to:  http://${IP:-<dgx-ip>}:${PORT}
  (or set AUTOMATIC1111_URL=http://${IP:-<dgx-ip>}:${PORT} for that service).

If generation errors with "no kernel image available for execution on the device",
the GB10 needs newer kernels than cu128 stable — re-run with the nightly wheel:
  A1111_TORCH_COMMAND="pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128" ${SUDO:+sudo }./install_a1111_gb10.sh
EOF
