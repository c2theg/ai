#!/usr/bin/env bash
# Christopher Gray  |  Version: 0.1.0  |  Update: 7/11/2026
# llama.cpp + NeMo ASR install / download / serve script for DGX Spark (GB10, aarch64)
#
# What this stands up (TWO engines — llama.cpp cannot run ASR, so ASR runs in NeMo):
#   1. llama.cpp  (llama-server, built from source w/ CUDA) serving the LLM:
#        unsloth/Qwen3.6-35B-A3B-GGUF   (Q6_K, vision MoE)         -> :8080
#      NOTE: you asked for nvidia/Qwen3.6-35B-A3B-NVFP4 — that is TensorRT/vLLM
#      FP4 and llama.cpp only loads GGUF, so we pull the identical model as a
#      GGUF quant instead. Same weights, llama.cpp-compatible.
#   2. NVIDIA NeMo (Docker) serving the streaming speech-recognition model:
#        nvidia/nemotron-3.5-asr-streaming-0.6b  (.nemo FastConformer-RNNT) -> :8090
#      This model is NOT GGUF and cannot run under llama.cpp at all.
#
# Update yourself:
#   curl -fsSL -o 'llamacpp_install_dgx.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/llamacpp_install_dgx.sh' && chmod u+x llamacpp_install_dgx.sh
#
# Move to DGX Spark / GB10:
#   scp llamacpp_install_dgx.sh user@10.11.1.10:/home/user/llamacpp_install_dgx.sh
#
# Usage:
#   ./llamacpp_install_dgx.sh                 — full install: deps, build llama.cpp,
#                                               download GGUF, pull NeMo image, download .nemo
#   ./llamacpp_install_dgx.sh --start         — start BOTH services (llama-server + ASR)
#   ./llamacpp_install_dgx.sh --start llm     — start only the llama.cpp LLM
#   ./llamacpp_install_dgx.sh --start asr     — start only the NeMo ASR container
#   ./llamacpp_install_dgx.sh --stop [llm|asr]— stop both / one service
#   ./llamacpp_install_dgx.sh --health        — probe both endpoints, report up/down
#   ./llamacpp_install_dgx.sh --status        — show pids/containers/ports/logs
#   ./llamacpp_install_dgx.sh --install-service — write + enable systemd units (boot start)
#   ./llamacpp_install_dgx.sh --build-only     — (re)build llama.cpp only
#   ./llamacpp_install_dgx.sh --download-only  — fetch models only
#
# ── Changelog ─────────────────────────────────────────────────────────────────
# v0.1.0  7/11/2026  Initial. llama.cpp CUDA build for GB10 (sm_121), Qwen3.6 Q6_K
#                    GGUF + mmproj, NeMo ASR Docker service, CUDA 13.2 gibberish guard.
set -euo pipefail

# ── Config (env-overridable) ──────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-$HOME/ai_llamacpp}"
LLAMA_DIR="${LLAMA_DIR:-$BASE_DIR/llama.cpp}"
MODELS_DIR="${MODELS_DIR:-$BASE_DIR/models}"
ASR_DIR="${ASR_DIR:-$BASE_DIR/asr}"
LOG_DIR="${LOG_DIR:-$BASE_DIR/logs}"
RUN_DIR="${RUN_DIR:-$BASE_DIR/run}"

# GB10 / Blackwell compute capability. "native" also works on recent nvcc.
CUDA_ARCH="${CUDA_ARCH:-121}"

# LLM (GGUF — llama.cpp compatible replacement for the NVFP4 repo)
LLM_REPO="${LLM_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
LLM_FILE="${LLM_FILE:-Qwen3.6-35B-A3B-UD-Q6_K.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-F16.gguf}"
LLM_PORT="${LLM_PORT:-8080}"
LLM_CTX="${LLM_CTX:-32768}"
LLM_NGL="${LLM_NGL:-999}"          # offload all layers (unified memory)

# ASR (NeMo — Docker)
ASR_REPO="${ASR_REPO:-nvidia/nemotron-3.5-asr-streaming-0.6b}"
ASR_FILE="${ASR_FILE:-nemotron-3.5-asr-streaming-0.6b.nemo}"
ASR_PORT="${ASR_PORT:-8090}"
ASR_IMAGE="${ASR_IMAGE:-nvcr.io/nvidia/nemo:25.09}"   # override to an arm64-tagged NeMo image
ASR_CONTAINER="${ASR_CONTAINER:-nemo-asr}"

HOST_BIND="${HOST_BIND:-0.0.0.0}"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"

# ── Logging ───────────────────────────────────────────────────────────────────
c_g=$'\033[32m'; c_y=$'\033[33m'; c_r=$'\033[31m'; c_b=$'\033[36m'; c_0=$'\033[0m'
log()  { printf '%s[*]%s %s\n' "$c_b" "$c_0" "$*"; }
ok()   { printf '%s[+]%s %s\n' "$c_g" "$c_0" "$*"; }
warn() { printf '%s[!]%s %s\n' "$c_y" "$c_0" "$*" >&2; }
err()  { printf '%s[x]%s %s\n' "$c_r" "$c_0" "$*" >&2; }
die()  { err "$*"; exit 1; }

mkdirs() { mkdir -p "$BASE_DIR" "$MODELS_DIR" "$ASR_DIR" "$LOG_DIR" "$RUN_DIR"; }

# ── Preflight ─────────────────────────────────────────────────────────────────
check_arch() {
  local a; a="$(uname -m)"
  [[ "$a" == "aarch64" || "$a" == "arm64" ]] || warn "Arch is $a, not aarch64 — expected on a DGX Spark GB10."
}

check_cuda() {
  if ! command -v nvcc >/dev/null 2>&1; then
    for p in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
      [[ -x "$p/nvcc" ]] && { export PATH="$p:$PATH"; break; }
    done
  fi
  command -v nvcc >/dev/null 2>&1 || die "nvcc not found. Install the CUDA toolkit (or add /usr/local/cuda/bin to PATH)."
  export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"

  local ver; ver="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
  ok "CUDA toolkit $ver at $CUDA_HOME"
  # KNOWN ISSUE (Jul 2026): Qwen3.6 produces gibberish on CUDA 13.2. NVIDIA fixing.
  if [[ "$ver" == "13.2" ]]; then
    warn "CUDA 13.2 detected — Qwen3.6 is known to emit GIBBERISH on 13.2."
    warn "Use CUDA 13.1 or 13.3+ for the LLM. Set ALLOW_CUDA_132=1 to override."
    [[ "${ALLOW_CUDA_132:-0}" == "1" ]] || die "Refusing to build the LLM on CUDA 13.2 (override with ALLOW_CUDA_132=1)."
  fi
}

# ── Dependencies ──────────────────────────────────────────────────────────────
install_deps() {
  log "Installing build + runtime dependencies (apt)…"
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
    build-essential cmake git ccache pkg-config \
    libcurl4-openssl-dev python3 python3-pip python3-venv \
    ca-certificates curl ffmpeg
  # huggingface CLI for downloads
  if ! command -v hf >/dev/null 2>&1; then
    pip3 install --user --upgrade "huggingface_hub[cli]" >/dev/null
    export PATH="$HOME/.local/bin:$PATH"
  fi
  command -v hf >/dev/null 2>&1 || die "hf CLI not on PATH after install (add ~/.local/bin to PATH)."
  ok "Dependencies installed."
}

# ── Build llama.cpp (CUDA) ────────────────────────────────────────────────────
build_llama() {
  check_arch; check_cuda
  if [[ -d "$LLAMA_DIR/.git" ]]; then
    log "Updating existing llama.cpp checkout…"
    git -C "$LLAMA_DIR" pull --ff-only || warn "git pull failed; building current checkout."
  else
    log "Cloning llama.cpp…"
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  fi
  log "Configuring (CUDA, sm_$CUDA_ARCH)…"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DLLAMA_CURL=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_BUILD_TYPE=Release
  log "Building llama-server (this takes a while)…"
  cmake --build "$LLAMA_DIR/build" --config Release -j "$(nproc)" --target llama-server llama-cli
  [[ -x "$LLAMA_SERVER" ]] || die "Build finished but $LLAMA_SERVER is missing."
  ok "llama.cpp built: $LLAMA_SERVER"
}

# ── Downloads ─────────────────────────────────────────────────────────────────
download_llm() {
  mkdirs
  log "Downloading LLM GGUF: $LLM_REPO :: $LLM_FILE (+ $MMPROJ_FILE)…"
  hf download "$LLM_REPO" "$LLM_FILE"    --local-dir "$MODELS_DIR"
  hf download "$LLM_REPO" "$MMPROJ_FILE" --local-dir "$MODELS_DIR"
  [[ -f "$MODELS_DIR/$LLM_FILE" ]] || die "LLM file missing after download: $MODELS_DIR/$LLM_FILE"
  ok "LLM ready: $MODELS_DIR/$LLM_FILE"
}

download_asr() {
  mkdirs
  [[ -n "${HF_TOKEN:-}" ]] || warn "HF_TOKEN not set — the NeMo repo may require accepting terms / auth."
  log "Downloading ASR model: $ASR_REPO :: $ASR_FILE…"
  hf download "$ASR_REPO" "$ASR_FILE" --local-dir "$ASR_DIR"
  [[ -f "$ASR_DIR/$ASR_FILE" ]] || die "ASR model missing after download: $ASR_DIR/$ASR_FILE"
  ok "ASR model ready: $ASR_DIR/$ASR_FILE"
  write_asr_server
  pull_asr_image
}

pull_asr_image() {
  command -v docker >/dev/null 2>&1 || die "docker not installed — needed for the NeMo ASR service."
  log "Pulling NeMo image: $ASR_IMAGE (must be an arm64 tag for GB10)…"
  docker pull "$ASR_IMAGE" || warn "docker pull failed — set ASR_IMAGE to an arm64-compatible NeMo tag."
}

# Minimal FastAPI ASR server, mounted into the NeMo container.
write_asr_server() {
  cat > "$ASR_DIR/asr_server.py" <<'PY'
#!/usr/bin/env python3
"""Minimal NeMo ASR HTTP server. POST an audio file to /transcribe.
Streaming (cache-aware) needs NeMo's streaming API — this exposes batch transcribe,
which is the reliable starting point. Health at /health."""
import os, tempfile, subprocess
from fastapi import FastAPI, UploadFile, File
import uvicorn
import nemo.collections.asr as nemo_asr

MODEL_PATH = os.environ.get("ASR_MODEL", "/models/nemotron-3.5-asr-streaming-0.6b.nemo")
app = FastAPI()
_model = None

def model():
    global _model
    if _model is None:
        _model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location="cuda")
        _model.eval()
    return _model

@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH), "loaded": _model is not None}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".in", delete=False) as fin:
        fin.write(raw); src = fin.name
    wav = src + ".wav"
    # normalize to 16 kHz mono PCM for the model
    subprocess.run(["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", wav],
                   check=True, capture_output=True)
    out = model().transcribe([wav])
    text = out[0].text if hasattr(out[0], "text") else str(out[0])
    for p in (src, wav):
        try: os.unlink(p)
        except OSError: pass
    return {"text": text}

if __name__ == "__main__":
    model()  # warm load
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8090")))
PY
  ok "Wrote ASR server: $ASR_DIR/asr_server.py"
}

# ── Serve: LLM ────────────────────────────────────────────────────────────────
start_llm() {
  [[ -x "$LLAMA_SERVER" ]] || die "llama-server not built. Run --build-only first."
  [[ -f "$MODELS_DIR/$LLM_FILE" ]] || die "LLM not downloaded. Run --download-only first."
  if [[ -f "$RUN_DIR/llm.pid" ]] && kill -0 "$(cat "$RUN_DIR/llm.pid")" 2>/dev/null; then
    warn "LLM already running (pid $(cat "$RUN_DIR/llm.pid")) on :$LLM_PORT."; return 0
  fi
  local mmproj_arg=()
  [[ -f "$MODELS_DIR/$MMPROJ_FILE" ]] && mmproj_arg=(--mmproj "$MODELS_DIR/$MMPROJ_FILE")
  log "Starting llama-server on :$LLM_PORT (ctx $LLM_CTX, ngl $LLM_NGL)…"
  nohup "$LLAMA_SERVER" \
    -m "$MODELS_DIR/$LLM_FILE" "${mmproj_arg[@]}" \
    --host "$HOST_BIND" --port "$LLM_PORT" \
    -c "$LLM_CTX" -ngl "$LLM_NGL" -fa on --jinja \
    --alias "Qwen3.6-35B-A3B" \
    > "$LOG_DIR/llm.log" 2>&1 &
  echo $! > "$RUN_DIR/llm.pid"
  ok "llama-server pid $(cat "$RUN_DIR/llm.pid") — log: $LOG_DIR/llm.log"
}

stop_llm() {
  if [[ -f "$RUN_DIR/llm.pid" ]]; then
    kill "$(cat "$RUN_DIR/llm.pid")" 2>/dev/null && ok "Stopped LLM." || warn "LLM not running."
    rm -f "$RUN_DIR/llm.pid"
  else warn "No LLM pidfile."; fi
}

# ── Serve: ASR (Docker) ───────────────────────────────────────────────────────
start_asr() {
  command -v docker >/dev/null 2>&1 || die "docker not installed."
  [[ -f "$ASR_DIR/$ASR_FILE" ]] || die "ASR model not downloaded. Run --download-only first."
  [[ -f "$ASR_DIR/asr_server.py" ]] || write_asr_server
  if docker ps --format '{{.Names}}' | grep -qx "$ASR_CONTAINER"; then
    warn "ASR container '$ASR_CONTAINER' already running on :$ASR_PORT."; return 0
  fi
  docker rm -f "$ASR_CONTAINER" >/dev/null 2>&1 || true
  log "Starting NeMo ASR container on :$ASR_PORT…"
  docker run -d --name "$ASR_CONTAINER" --restart unless-stopped \
    --gpus all --ipc=host \
    -p "$ASR_PORT":8090 \
    -v "$ASR_DIR":/models \
    -e ASR_MODEL="/models/$ASR_FILE" -e PORT=8090 \
    "$ASR_IMAGE" \
    bash -lc "pip install -q fastapi uvicorn python-multipart && python /models/asr_server.py" \
    > "$LOG_DIR/asr.cid" 2>&1
  ok "ASR container started — logs: docker logs -f $ASR_CONTAINER"
}

stop_asr() {
  docker rm -f "$ASR_CONTAINER" >/dev/null 2>&1 && ok "Stopped ASR container." || warn "ASR container not running."
}

# ── Health / status ───────────────────────────────────────────────────────────
health() {
  local llm asr
  if curl -fsS "http://127.0.0.1:$LLM_PORT/health" >/dev/null 2>&1; then llm="${c_g}UP${c_0}"; else llm="${c_r}DOWN${c_0}"; fi
  if curl -fsS "http://127.0.0.1:$ASR_PORT/health" >/dev/null 2>&1; then asr="${c_g}UP${c_0}"; else asr="${c_r}DOWN${c_0}"; fi
  printf 'LLM (llama.cpp)  :%s  %b\n' "$LLM_PORT" "$llm"
  printf 'ASR (NeMo)       :%s  %b\n' "$ASR_PORT" "$asr"
}

status() {
  echo "== llama.cpp =="
  [[ -f "$RUN_DIR/llm.pid" ]] && ps -p "$(cat "$RUN_DIR/llm.pid")" -o pid,etime,rss,cmd 2>/dev/null || echo "  not running"
  echo "  log: $LOG_DIR/llm.log"
  echo "== NeMo ASR (docker) =="
  docker ps --filter "name=$ASR_CONTAINER" --format '  {{.Names}}  {{.Status}}  {{.Ports}}' 2>/dev/null || echo "  docker unavailable"
  echo
  health
}

# ── systemd (optional boot persistence) ───────────────────────────────────────
install_service() {
  local self; self="$(readlink -f "$0")"
  log "Writing systemd units…"
  sudo tee /etc/systemd/system/llamacpp-llm.service >/dev/null <<EOF
[Unit]
Description=llama.cpp LLM server (Qwen3.6-35B-A3B)
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
User=$USER
Environment=BASE_DIR=$BASE_DIR
ExecStart=$LLAMA_SERVER -m $MODELS_DIR/$LLM_FILE --mmproj $MODELS_DIR/$MMPROJ_FILE --host $HOST_BIND --port $LLM_PORT -c $LLM_CTX -ngl $LLM_NGL -fa on --jinja --alias Qwen3.6-35B-A3B
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
  sudo tee /etc/systemd/system/llamacpp-asr.service >/dev/null <<EOF
[Unit]
Description=NeMo ASR server (nemotron-3.5-asr-streaming-0.6b)
Requires=docker.service
After=docker.service
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=$self --start asr
ExecStop=$self --stop asr
[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now llamacpp-llm.service llamacpp-asr.service
  ok "Enabled llamacpp-llm.service + llamacpp-asr.service (start on boot)."
}

# ── Full install ──────────────────────────────────────────────────────────────
full_install() {
  mkdirs
  install_deps
  build_llama
  download_llm
  download_asr
  ok "Install complete. Start everything with:  $0 --start"
  echo
  health || true
}

usage() { sed -n '1,60p' "$0" | grep -E '^#' | sed 's/^# \{0,1\}//'; }

# ── Dispatch ──────────────────────────────────────────────────────────────────
main() {
  local cmd="${1:-install}"
  case "$cmd" in
    ""|install)        full_install ;;
    --build-only)      mkdirs; install_deps; build_llama ;;
    --download-only)   install_deps; download_llm; download_asr ;;
    --start)
      case "${2:-both}" in
        llm) start_llm ;; asr) start_asr ;; both|"") start_llm; start_asr ;;
        *) die "Unknown --start target: $2 (llm|asr|both)";;
      esac ;;
    --stop)
      case "${2:-both}" in
        llm) stop_llm ;; asr) stop_asr ;; both|"") stop_llm; stop_asr ;;
        *) die "Unknown --stop target: $2 (llm|asr|both)";;
      esac ;;
    --health)          health ;;
    --status)          status ;;
    --install-service) install_service ;;
    -h|--help)         usage ;;
    *)                 err "Unknown command: $cmd"; usage; exit 1 ;;
  esac
}
main "$@"
