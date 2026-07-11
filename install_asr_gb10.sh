#!/usr/bin/env bash
#    By: Christopher Gray
#    Version: 0.1.1
#    Updated: 7/11/2026
#
#    This script installs the ASR sidecar on an NVIDIA DGX Spark / GB10 (arm64 + Blackwell).
#
#    curl -sSL https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_asr_gb10.sh | bash
#
#   Installer:
#     ./install_asr_gb10.sh        (from a synced checkout — preferred)
#
#   0.1.0: initial release — builds and runs the ASR sidecar (Qwen3-ASR-1.7B
#          batch + Nemotron 3.5 streaming 0.6B) on a DGX Spark / GB10.
#
# install_asr_gb10.sh — one-shot installer for the ASR sidecar on an NVIDIA
# DGX Spark / GB10 (arm64 + Blackwell), reachable over the LAN. Mirrors
# install_a1111_gb10.sh step-for-step.
#
# What it does, idempotently (safe to re-run — a re-run UPDATES to the latest):
#   1. Preflight: confirm arm64 + an NVIDIA driver (nvidia-smi).
#   2. Docker: install if missing.
#   3. NVIDIA Container Toolkit: install + wire into Docker if needed.
#   4. Prove a container can see the GPU.
#   5. Open the LAN firewall port (ufw) if a firewall is active.
#   6. Build the sidecar image (Blackwell torch cu128 + NeMo + qwen-asr).
#   7. Pre-fetch the three HF models into the host-mounted cache (idempotent).
#   8. (Re)start the container bound to 0.0.0.0 so the app hosts can reach it.
#   9. Verify: /healthz reports all models loaded, a test WAV transcribes,
#      and torch sees the GPU.
#
# Unlike the A1111 installer there is no embedded Dockerfile: the sidecar
# needs its app/ source tree, so run this from a synced checkout of the repo
# (Dockerfile + app code next to this script under asr_sidecar/), or set
# ASR_REPO_GIT_URL to a git repo containing asr_sidecar/ for detached runs.
#
# Everything below is overridable via environment variables (see the block).
set -euo pipefail

# ─────────────────────────────── configuration ───────────────────────────────
IMAGE_NAME="${ASR_IMAGE_NAME:-asr-gb10}"
IMAGE_TAG="${ASR_IMAGE_TAG:-latest}"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
CONTAINER_NAME="${ASR_CONTAINER_NAME:-asr-gb10}"
PORT="${ASR_PORT:-8790}"

# arm64 CUDA base (Docker auto-selects the arm64/sbsa variant on the GB10).
BASE_IMAGE="${ASR_BASE_IMAGE:-nvidia/cuda:12.8.0-runtime-ubuntu22.04}"

# Blackwell-capable torch. cu128 is proven on this box by the A1111 install.
TORCH_COMMAND="${ASR_TORCH_COMMAND:-pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128}"

# all | batch | stream — build/run only one engine per container if the NeMo
# and qwen-asr dependency stacks ever conflict (see asr_sidecar/Dockerfile).
VARIANT="${ASR_VARIANT:-all}"

# Detached runs only: a git repo that contains asr_sidecar/ (cloned when this
# script isn't sitting next to a synced checkout).
REPO_GIT_URL="${ASR_REPO_GIT_URL:-}"

# Model pre-fetch: 0 to skip; HF_TOKEN forwarded for gated/rate-limited repos.
DOWNLOAD_MODELS="${ASR_DOWNLOAD_MODELS:-1}"
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

# Resolve our own location. Under `curl | bash` there is no script file, so
# BASH_SOURCE is unset — guard it (set -u) and fall back to detached mode.
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  SCRIPT_DIR=""
fi

# Build context: prefer a synced checkout (asr_sidecar/ next to this script);
# otherwise a self-managed work dir we clone the repo into.
if [ -n "${SCRIPT_DIR}" ] && [ -f "${SCRIPT_DIR}/asr_sidecar/Dockerfile" ]; then
  BUILD_DIR="${SCRIPT_DIR}/asr_sidecar"
  FETCH_SOURCE=0
else
  WORK_DIR="${ASR_WORK_DIR:-${HOME:-/root}/.asr-gb10}"
  BUILD_DIR="${WORK_DIR}/repo/asr_sidecar"
  FETCH_SOURCE=1
fi
DATA_DIR="${ASR_DATA_DIR:-${BUILD_DIR}/runtime}"

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
step "1/9  Preflight"
ARCH="$(uname -m)"
if [ "${ARCH}" != "aarch64" ] && [ "${ARCH}" != "arm64" ]; then
  warn "This machine is '${ARCH}', not arm64. This script targets the DGX Spark / GB10."
  warn "Continuing anyway in 5s (Ctrl-C to abort)…"
  sleep 5
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  log "NVIDIA driver present:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/    /'
else
  die "nvidia-smi not working — the GPU driver isn't loaded.
    On a DGX Spark the driver ships with DGX OS; do NOT install a generic driver
    over it. Verify DGX OS is up to date, then re-run."
fi

# ─────────────────────────────── 2. docker ───────────────────────────────────
step "2/9  Docker engine"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  log "Docker present: $(docker --version)"
else
  log "Installing Docker via get.docker.com …"
  curl -fsSL https://get.docker.com | ${SUDO} sh || die "Docker install failed."
  ${SUDO} systemctl enable --now docker || true
  docker info >/dev/null 2>&1 || die "Docker installed but the daemon isn't responding."
fi

# ──────────────────────── 3. NVIDIA Container Toolkit ─────────────────────────
step "3/9  NVIDIA Container Toolkit"
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
step "4/9  Verify a container can see the GPU"
if ${SUDO} docker run --rm --gpus all "${BASE_IMAGE}" nvidia-smi -L >/tmp/asr_gpu_check 2>&1; then
  log "GPU visible inside containers:"; sed 's/^/    /' /tmp/asr_gpu_check
else
  cat /tmp/asr_gpu_check >&2 || true
  die "A container could not access the GPU with '--gpus all'. Toolkit/runtime issue — see output above."
fi

# ─────────────────────────────── 5. firewall ─────────────────────────────────
step "5/9  LAN firewall"
if command -v ufw >/dev/null 2>&1 && ${SUDO} ufw status 2>/dev/null | grep -qi '^Status: active'; then
  ${SUDO} ufw allow "${PORT}/tcp" >/dev/null 2>&1 || true
  log "Opened ${PORT}/tcp in ufw."
else
  log "No active ufw firewall — nothing to open (port ${PORT} governed by your network)."
fi

# ──────────────────────────────── 6. build ───────────────────────────────────
step "6/9  Build the ASR sidecar image — first build takes a while"
if [ "${FETCH_SOURCE}" -eq 1 ]; then
  [ -n "${REPO_GIT_URL}" ] || die "Detached run: asr_sidecar/ not found next to this script.
    Either run from a synced checkout of the repo, or set ASR_REPO_GIT_URL to a
    git URL containing asr_sidecar/ and re-run."
  mkdir -p "${WORK_DIR}"
  if [ -d "${WORK_DIR}/repo/.git" ]; then
    log "Updating source clone in ${WORK_DIR}/repo …"
    git -C "${WORK_DIR}/repo" pull --ff-only
  else
    log "Cloning ${REPO_GIT_URL} → ${WORK_DIR}/repo …"
    git clone --depth=1 "${REPO_GIT_URL}" "${WORK_DIR}/repo"
  fi
  [ -f "${BUILD_DIR}/Dockerfile" ] || die "Clone has no asr_sidecar/Dockerfile."
else
  log "Building from synced repo at ${BUILD_DIR}"
fi
cd "${BUILD_DIR}"
log "base image:  ${BASE_IMAGE}"
log "torch:       ${TORCH_COMMAND}"
log "variant:     ${VARIANT}"
${SUDO} docker build --pull \
  -f Dockerfile \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "TORCH_COMMAND=${TORCH_COMMAND}" \
  -t "${IMAGE}" . \
  || die "Image build failed. If 'pip check' or the import smoke test failed, the
    NeMo × qwen-asr stacks conflict on this base — build two single-engine
    containers instead:
      ASR_VARIANT=batch  ASR_CONTAINER_NAME=asr-gb10-batch  ./install_asr_gb10.sh
      ASR_VARIANT=stream ASR_CONTAINER_NAME=asr-gb10-stream ASR_PORT=8791 ./install_asr_gb10.sh
    NeMo wheel pain on aarch64 (numba/llvmlite)? Try the NeMo base image:
      ASR_BASE_IMAGE=nvcr.io/nvidia/nemo:25.04 ASR_TORCH_COMMAND=true ./install_asr_gb10.sh"
log "Built ${IMAGE}."

# ─────────────────────────── 7. pre-fetch models ─────────────────────────────
step "7/9  Pre-fetch HF models into ${DATA_DIR} (idempotent)"
mkdir -p "${DATA_DIR}"
if [ "${DOWNLOAD_MODELS}" != "0" ]; then
  [ -z "${HF_TOKEN}" ] && warn "HF_TOKEN not set — public repos still work; set it if a download 401s/429s."
  ${SUDO} docker run --rm \
    -v "${DATA_DIR}:/models" \
    ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
    "${IMAGE}" python3 -m app.prefetch \
    || warn "Model pre-fetch reported failures — the container will retry lazily at startup."
else
  log "ASR_DOWNLOAD_MODELS=0 — skipping model pre-fetch."
fi

# ──────────────────────────────── 8. run ─────────────────────────────────────
step "8/9  (Re)start the container"
if ${SUDO} docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  log "Removing previous container '${CONTAINER_NAME}' …"
  ${SUDO} docker rm -f "${CONTAINER_NAME}" >/dev/null
fi
${SUDO} docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --gpus all \
  -p "0.0.0.0:${PORT}:8790" \
  -v "${DATA_DIR}:/models" \
  -e "ASR_VARIANT=${VARIANT}" \
  ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
  "${IMAGE}" >/dev/null
log "Container started."

# ─────────────────────────────── 9. verify ───────────────────────────────────
step "9/9  Verify (model loading can take a few minutes)"
HEALTH="http://127.0.0.1:${PORT}/healthz"
ok=0
for i in $(seq 1 120); do
  body="$(curl -fsS "${HEALTH}" 2>/dev/null || true)"
  if [ -n "${body}" ] && ! printf '%s' "${body}" | grep -q '"loading"'; then
    ok=1; break
  fi
  sleep 5
done
if [ "${ok}" -eq 1 ]; then
  log "healthz: ${body}"
  if printf '%s' "${body}" | grep -q '"error'; then
    warn "One or more engines failed to load — check: ${SUDO:+sudo }docker logs ${CONTAINER_NAME}"
  fi
  if [ "${VARIANT}" != "stream" ]; then
    log "Posting a 2s test WAV to /v1/audio/transcriptions …"
    TESTWAV=/tmp/asr_test_$$.wav
    if command -v ffmpeg >/dev/null 2>&1; then
      ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 -ac 1 "${TESTWAV}" >/dev/null 2>&1
    else
      ${SUDO} docker exec "${CONTAINER_NAME}" ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" \
        -ar 16000 -ac 1 /tmp/asr_test.wav >/dev/null 2>&1
      ${SUDO} docker cp "${CONTAINER_NAME}:/tmp/asr_test.wav" "${TESTWAV}" >/dev/null
    fi
    if curl -fsS -F "file=@${TESTWAV}" -F "response_format=verbose_json" \
         "http://127.0.0.1:${PORT}/v1/audio/transcriptions" | grep -q '"segments"'; then
      log "Batch transcription endpoint responds ✅"
    else
      warn "Batch endpoint didn't return segments — check: ${SUDO:+sudo }docker logs ${CONTAINER_NAME}"
    fi
    rm -f "${TESTWAV}"
  fi
  if ${SUDO} docker exec "${CONTAINER_NAME}" python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    gpu="$(${SUDO} docker exec "${CONTAINER_NAME}" python3 -c "import torch;print(torch.cuda.get_device_name(0))" 2>/dev/null || true)"
    log "torch sees the GPU ✅  (${gpu:-unknown})"
  else
    warn "torch.cuda.is_available() is FALSE inside the container — it'll run on CPU (slow)."
  fi
else
  warn "healthz still reports loading after ~10 min. Watch startup with:"
  warn "  ${SUDO:+sudo }docker logs -f ${CONTAINER_NAME}"
fi

# ─────────────────────────────── summary ─────────────────────────────────────
IP="$(lan_ip)"
cat <<EOF

${c_b}Done.${c_0}
  Local:      http://127.0.0.1:${PORT}/healthz
$( [ -n "${IP}" ] && printf '  LAN:        http://%s:%s   (use this URL from the app hosts)\n' "${IP}" "${PORT}" )
  Logs:       ${SUDO:+sudo }docker logs -f ${CONTAINER_NAME}
  Stop:       ${SUDO:+sudo }docker rm -f ${CONTAINER_NAME}
  Update:     re-run this script (rebuilds with --pull, recreates the container)
  Models:     ${DATA_DIR}  (HF cache; safe to keep across rebuilds)

${c_b}Point the transcription app at this box${c_0} (on the app host, e.g. ai2-strixhelo):
  Option A — AI Providers page: add an 'openai_compatible' connection with
    base_url http://${IP:-<dgx-ip>}:${PORT} named "ASR Sidecar (DGX)", then route
    task 'transcription' → qwen3-asr-1.7b and task 'live_transcription' →
    nemotron-3.5-asr-streaming-0.6b.
  Option B — env vars on the app host:
    ASR_SIDECAR_URL=http://${IP:-<dgx-ip>}:${PORT}
    LIVE_ASR_WS_URL=ws://${IP:-<dgx-ip>}:${PORT}/v1/audio/stream
  Quality gate tuning (app host): ASR_QUALITY_THRESHOLD=0.90 (default),
    ASR_DUAL_TRANSCRIPTION=1 to archive whisper output alongside every run.

If generation errors with "no kernel image available for execution on the device",
the GB10 needs newer kernels than cu128 stable — re-run with the nightly wheel:
  ASR_TORCH_COMMAND="pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128" ${SUDO:+sudo }./install_asr_gb10.sh
EOF
