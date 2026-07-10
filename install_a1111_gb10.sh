#!/usr/bin/env bash
#    By: Christopher Gray
#    Version: 0.1.0
#    Updated: 7/9/2026
#
#   Installer:
#     curl -sSL https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_a1111_gb10.sh | bash
#
#   0.1.0: seed SDXL checkpoints host-side on install (new step 7) — realistic,
#          photorealistic, concept-art, product-render, cyberpunk. Idempotent
#          (skips files already downloaded). Civitai's download endpoint now
#          401s without an API key, so the three Civitai models each have a
#          tokenless HuggingFace fallback (used when Civitai is unavailable or
#          CIVITAI_API_TOKEN is unset). Toggle A1111_DOWNLOAD_MODELS=0, subset
#          A1111_MODELS="realistic,cyberpunk", auth CIVITAI_API_TOKEN.
#   0.0.9: add build-essential + python3-dev so arm64 source-only wheels
#          (psutil, …) compile during the build.
#   0.0.8: fully self-contained — embeds the Dockerfile, so a detached run needs
#          nothing else hosted (set A1111_DOCKERFILE_URL to override).
#   0.0.7: works when run detached (curl | bash) — guards unset BASH_SOURCE and
#          fixes the apt env-prefix bug.
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

# When run detached (curl | bash) the Dockerfile isn't on disk. By default we
# write the copy EMBEDDED at the bottom of this script (fully self-contained —
# nothing else to host). Set A1111_DOCKERFILE_URL to fetch it from a URL instead.
DOCKERFILE_URL="${A1111_DOCKERFILE_URL:-}"

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

# ─────────────────────────── model checkpoints ───────────────────────────────
# On install we seed the A1111 checkpoint dir on the HOST. It has to be host-side:
# /app/models is a bind mount, so anything baked into the image at that path is
# shadowed at runtime. Files land in ${DATA_DIR}/models/Stable-diffusion.
#   A1111_DOWNLOAD_MODELS=0            skip all checkpoint downloads
#   A1111_MODELS="realistic,cyberpunk" download only these keys (default: all)
#   CIVITAI_API_TOKEN=xxxx            attached to the Civitai-hosted checkpoints
DOWNLOAD_MODELS="${A1111_DOWNLOAD_MODELS:-1}"
MODELS_FILTER="${A1111_MODELS:-all}"
CIVITAI_TOKEN="${CIVITAI_API_TOKEN:-${A1111_CIVITAI_TOKEN:-}}"

# One SDXL checkpoint per requested category, as:
#     key | primary_filename | primary_url | fallback_filename | fallback_url
# The downloader tries the primary, then the (tokenless) fallback if the primary
# fails. HuggingFace /resolve/ URLs are tokenless direct downloads; Civitai URLs
# are public but its download endpoint now returns HTTP 401 without an API key —
# so each Civitai model has a tokenless HF fallback for when Civitai is
# unavailable or CIVITAI_API_TOKEN is unset. Set CIVITAI_API_TOKEN to get the
# exact Civitai models instead of the fallback.
HF_REALVIS="https://huggingface.co/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors"
HF_SDXL_BASE="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
CHECKPOINTS=(
  "realistic|RealVisXL_V5.0_fp16.safetensors|${HF_REALVIS}||"
  "photorealistic|Juggernaut-XI-byRunDiffusion.safetensors|https://huggingface.co/RunDiffusion/Juggernaut-XI-v11/resolve/main/Juggernaut-XI-byRunDiffusion.safetensors||"
  "concept-art|artUniverse_v80SDXL.safetensors|https://civitai.com/api/download/models/2653358|sd_xl_base_1.0.safetensors|${HF_SDXL_BASE}"
  "product-render|easyProductPhotoXL_v20.safetensors|https://civitai.com/api/download/models/434028|RealVisXL_V5.0_fp16.safetensors|${HF_REALVIS}"
  "cyberpunk|sdxlHK_v12.safetensors|https://civitai.com/api/download/models/1654136|sd_xl_base_1.0.safetensors|${HF_SDXL_BASE}"
)

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

# Byte size of a file, portable across Linux (DGX host) and BSD/macOS.
file_size() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0; }

model_selected() {
  # $1 = key; true when MODELS_FILTER is 'all' or its comma list contains the key.
  [ "${MODELS_FILTER}" = "all" ] && return 0
  case ",${MODELS_FILTER}," in *",$1,"*) return 0;; *) return 1;; esac
}

fetch_checkpoint() {
  # $1=url  $2=dest.part  — resumable download via wget (preferred) or curl.
  local url="$1" tmp="$2"
  case "$url" in
    *civitai.com*)
      if [ -n "${CIVITAI_TOKEN}" ]; then
        case "$url" in
          *\?*) url="${url}&token=${CIVITAI_TOKEN}";;
             *) url="${url}?token=${CIVITAI_TOKEN}";;
        esac
      fi ;;
  esac
  if command -v wget >/dev/null 2>&1; then
    wget --show-progress -q -c -O "$tmp" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL -C - -o "$tmp" "$url"
  else
    warn "Neither wget nor curl is installed — cannot download checkpoints."; return 1
  fi
}

present() { [ -f "$1" ] && [ "$(file_size "$1")" -gt 1073741824 ]; }

# Download one (filename,url) candidate into $dir. Echoes "ok"/"small"/"fail".
try_candidate() {
  local dir="$1" fname="$2" url="$3" dest="${1}/${2}" sz
  [ -n "$fname" ] && [ -n "$url" ] || { echo "skip"; return; }
  if present "$dest"; then echo "ok"; return; fi   # a prior category already got it
  if fetch_checkpoint "$url" "${dest}.part"; then
    sz="$(file_size "${dest}.part")"
    # A redirect-to-login (e.g. Civitai 401) returns a tiny page; reject <100MB.
    if [ "$sz" -lt 104857600 ]; then
      warn "    got only ${sz} bytes (likely an auth redirect) — discarding ${fname}."
      rm -f "${dest}.part"; echo "small"
    else
      mv -f "${dest}.part" "$dest"; log "    ${fname}: done ($((sz/1024/1024)) MB)." >&2; echo "ok"
    fi
  else
    rm -f "${dest}.part"; echo "fail"
  fi
}

download_checkpoints() {
  local dir="${DATA_DIR}/models/Stable-diffusion"
  mkdir -p "$dir"
  local ok=0 skip=0 fail=0 entry key f1 u1 f2 u2 r
  for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r key f1 u1 f2 u2 <<<"$entry"
    model_selected "$key" || continue
    # Already have the primary or its fallback? Nothing to do.
    if present "${dir}/${f1}" || { [ -n "$f2" ] && present "${dir}/${f2}"; }; then
      log "✓ ${key}: already present — skipping."; skip=$((skip+1)); continue
    fi
    log "↓ ${key}: fetching ${f1} …"
    r="$(try_candidate "$dir" "$f1" "$u1")"
    if [ "$r" != "ok" ] && [ -n "$f2" ]; then
      warn "  ${key}: primary unavailable — falling back to ${f2}"
      r="$(try_candidate "$dir" "$f2" "$u2")"
    fi
    case "$r" in
      ok) ok=$((ok+1));;
      *)  warn "  ${key}: could not obtain a checkpoint — continuing (it just won't show in the dropdown)."; fail=$((fail+1));;
    esac
  done
  log "Checkpoints: ${ok} obtained, ${skip} already present, ${fail} failed → ${dir}"
  [ "$fail" -gt 0 ] && warn "Some checkpoints failed; A1111 still runs with whatever landed."
  return 0
}

# Embedded copy of Dockerfile.automatic1111 so a detached `curl | bash` run has
# no second file to host. KEEP IN SYNC with Dockerfile.automatic1111 in the repo
# (the quoted heredoc means everything below is written verbatim — no expansion).
write_dockerfile() {
  cat > "$1" <<'__A1111_DOCKERFILE__'
# syntax=docker/dockerfile:1.7
#
# Automatic1111 (Stable Diffusion WebUI), containerized. (Embedded copy — see
# install_a1111_gb10.sh; mirror of Dockerfile.automatic1111.)
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    venv_dir=-

# build-essential + python3-dev: on arm64 several requirements (psutil, …) have
# no prebuilt wheel and compile from source (needs a C toolchain + Py headers).
RUN apt-get update && apt-get install -y --no-install-recommends \
      git python3 python3-pip python3-dev python-is-python3 build-essential \
      libgl1 libglib2.0-0 libgoogle-perftools4 bc wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pin to a released tag so the sub-repo commit hashes are reproducible.
WORKDIR /app
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git . \
    && git checkout v1.10.1

# Anonymous clone; GIT_TERMINAL_PROMPT=0 fails fast instead of hanging. Do NOT
# set GIT_ASKPASS=/bin/true (empty creds → GitHub 404s public repos).
ENV GIT_TERMINAL_PROMPT=0

# Upstream DELETED github.com/Stability-AI/stablediffusion; w-e-w (an A1111 core
# maintainer) mirrors it with the exact pinned commit. A1111 reads each sub-repo
# URL from an env var, so this redirects both build-time and runtime clones.
ENV STABLE_DIFFUSION_REPO=https://github.com/w-e-w/stablediffusion.git

# Clone the pinned sub-repos explicitly (webui.sh --exit returns before the
# clone stage). URL env-name + commit are read from launch_utils.py and honor
# the env override above; each is fetched by exact SHA into a detached HEAD so
# A1111's own git_clone() sees the expected HEAD and never hits the network.
RUN python3 <<'PY'
import os, re, subprocess
src = open("modules/launch_utils.py").read()
def resolve(var):
    m = re.search(rf"{var}\s*=\s*os\.environ\.get\(\s*[\"']([^\"']+)[\"']\s*,\s*[\"']([^\"']+)[\"']", src)
    if not m:
        raise SystemExit(f"could not parse {var} in launch_utils.py")
    env_name, default = m.group(1), m.group(2)
    return os.environ.get(env_name, default)
repos = [
    ("stable_diffusion_repo",    "stable_diffusion_commit_hash",    "stable-diffusion-stability-ai"),
    ("stable_diffusion_xl_repo", "stable_diffusion_xl_commit_hash", "generative-models"),
    ("k_diffusion_repo",         "k_diffusion_commit_hash",         "k-diffusion"),
    ("blip_repo",                "blip_commit_hash",                "BLIP"),
    ("assets_repo",              "assets_commit_hash",              "stable-diffusion-webui-assets"),
]
os.makedirs("repositories", exist_ok=True)
for url_var, hash_var, dirname in repos:
    url, commit = resolve(url_var), resolve(hash_var)
    dest = os.path.join("repositories", dirname)
    print(f"==> {url} @ {commit} -> {dest}", flush=True)
    subprocess.run(["git", "init", "-q", dest], check=True)
    subprocess.run(["git", "-C", dest, "remote", "add", "origin", url], check=True)
    subprocess.run(["git", "-C", dest, "fetch", "-q", "--depth=1", "origin", commit], check=True)
    subprocess.run(["git", "-C", dest, "checkout", "-q", "FETCH_HEAD"], check=True)
PY

# TORCH_COMMAND: A1111's default targets an x86 CUDA-12.1 wheel that has no
# arm64/Blackwell build. Pass a cu128 (or newer) wheel for the DGX Spark/GB10.
ARG TORCH_COMMAND=""
RUN if [ -n "$TORCH_COMMAND" ]; then export TORCH_COMMAND="$TORCH_COMMAND"; fi; \
    ./webui.sh -f --skip-torch-cuda-test --skip-python-version-check --exit
RUN python3 -m pip install -r requirements_versions.txt

# Fail the build if the sub-repo clones didn't land.
RUN set -eux; for repo in \
      stable-diffusion-stability-ai generative-models k-diffusion BLIP \
      stable-diffusion-webui-assets; do \
      test -d "repositories/$repo/.git" || { echo "MISSING repositories/$repo — bootstrap did not complete"; exit 1; }; \
    done

VOLUME ["/app/models", "/app/outputs", "/app/extensions"]
EXPOSE 7860
ENTRYPOINT ["./webui.sh", "-f", "--listen", "--api", "--port", "7860"]
__A1111_DOCKERFILE__
}

# ─────────────────────────────── 1. preflight ────────────────────────────────
step "1/9  Preflight"
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
if ${SUDO} docker run --rm --gpus all "${BASE_IMAGE}" nvidia-smi -L >/tmp/a1111_gpu_check 2>&1; then
  log "GPU visible inside containers:"; sed 's/^/    /' /tmp/a1111_gpu_check
else
  cat /tmp/a1111_gpu_check >&2 || true
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
step "6/9  Build the A1111 image (Blackwell torch) — first build takes a while"
mkdir -p "${BUILD_DIR}"
if [ "${FETCH_DOCKERFILE}" -eq 1 ]; then
  if [ -n "${DOCKERFILE_URL}" ]; then
    log "Detached run — fetching Dockerfile from ${DOCKERFILE_URL}"
    curl -fsSL "${DOCKERFILE_URL}" -o "${BUILD_DIR}/Dockerfile.automatic1111" \
      || die "Could not download the Dockerfile from ${DOCKERFILE_URL}"
  else
    log "Detached run — writing embedded Dockerfile → ${BUILD_DIR}/Dockerfile.automatic1111"
    write_dockerfile "${BUILD_DIR}/Dockerfile.automatic1111"
  fi
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

# ─────────────────────────── 7. download checkpoints ─────────────────────────
step "7/9  Seed SDXL checkpoints (host-side; skips any already downloaded)"
if [ "${DOWNLOAD_MODELS}" != "0" ]; then
  log "Target: ${DATA_DIR}/models/Stable-diffusion  (filter: ${MODELS_FILTER})"
  [ -z "${CIVITAI_TOKEN}" ] && warn "CIVITAI_API_TOKEN not set — public Civitai models still work, but set it if any download comes back as an auth redirect."
  download_checkpoints
else
  log "A1111_DOWNLOAD_MODELS=0 — skipping checkpoint downloads."
fi

# ──────────────────────────────── 8. run ─────────────────────────────────────
step "8/9  (Re)start the container"
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

# ─────────────────────────────── 9. verify ───────────────────────────────────
step "9/9  Verify (allow a few minutes for the API to come up)"
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
  Models:     ${DATA_DIR}/models/Stable-diffusion  (drop more .safetensors here)
              re-run with A1111_MODELS="realistic,cyberpunk" for a subset, or
              A1111_DOWNLOAD_MODELS=0 to skip; CIVITAI_API_TOKEN for Civitai auth.

${c_b}Point the transcription app at this box${c_0} (on the app host, e.g. ai2-strixhelo):
  In the app's AI Providers page, set the Automatic1111 / image_generation
  connection's base_url to:  http://${IP:-<dgx-ip>}:${PORT}
  (or set AUTOMATIC1111_URL=http://${IP:-<dgx-ip>}:${PORT} for that service).

If generation errors with "no kernel image available for execution on the device",
the GB10 needs newer kernels than cu128 stable — re-run with the nightly wheel:
  A1111_TORCH_COMMAND="pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128" ${SUDO:+sudo }./install_a1111_gb10.sh
EOF
