#!/usr/bin/env bash
#    By: Christopher Gray
#    Version: 0.2.0
#    Updated: 7/18/2026
#
#    This script installs the ASR sidecar on an NVIDIA DGX Spark / GB10 (arm64 + Blackwell).
#
#    curl -fsSL -o 'install_asr_gb10.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_asr_gb10.sh' && chmod u+x install_asr_gb10.sh
#
#   Installer:
#     ./install_asr_gb10.sh        (from a synced checkout — preferred)
#
#   0.2.0: scheduler/observability overhaul (job queue + duration-aware GPU
#          batching, X-ASR-* timing headers, /ready + /metrics + /v1/audio/jobs,
#          chunk retry + resume, benchmark tool — see asr_sidecar/README.md).
#          Also: a persistent tuning env file (/etc/qwen-asr/qwen-asr.env,
#          override via ASR_ENV_FILE) is now written on first install and
#          only back-filled (never overwritten) on re-runs, wired in via
#          `docker run --env-file` plus --shm-size=2g and log rotation;
#          ASR_VARIANT is locked to 'batch' (Qwen3-ASR only) — the Nemotron
#          streaming engine's NeMo/transformers dependency conflict made
#          'all'/'stream' fail at runtime, and live mic already falls back to
#          local faster-whisper, so it's no longer loaded or configurable;
#          step 8 now also stops/removes any OTHER container already
#          publishing the target port (not just one matching CONTAINER_NAME),
#          so a stale leftover can't keep silently answering requests after
#          the script reports success; the model pre-fetch step now passes
#          --entrypoint python3 (without it, "python3 -m app.prefetch" was
#          appended as extra args to the image's fixed uvicorn ENTRYPOINT
#          instead of replacing it, which failed as "No such option '-m'").
#   0.1.8: `pip check` gate no longer fails the build on aarch64 platform
#          notes (torch cu128 pulls nvidia cu12 sub-wheels like cusparselt
#          whose tags don't match arm64) — it now fails only on real
#          "X requires Y" dependency conflicts.
#   0.1.7: print the installer version at start (verifies you're running
#          the latest copy, not a stale raw.githubusercontent cache).
#   0.1.6: the Dockerfile now runs NEMO_COMMAND through `eval` — plain
#          ${VAR} expansion never re-parses the quotes inside the value, so
#          0.1.5's quoted requirement reached pip with a literal ' attached.
#   0.1.5: NeMo-from-git installed WITHOUT its [asr] extras (no hydra) —
#          Ubuntu 22.04's apt pip (22.0) silently drops extras on direct-URL
#          requirements. The Dockerfile now upgrades pip first and uses the
#          canonical 'nemo_toolkit[asr] @ git+...' form.
#   0.1.4: nemotron-3.5-asr-streaming needs EncDecRNNTBPEModelWithPrompt,
#          which no NeMo pypi release ships yet — install NeMo from git main
#          (the model card's documented install; override with
#          ASR_NEMO_COMMAND). Streaming engine also self-heals to rolling
#          re-decode if the cache-aware step API mismatches at runtime.
#   0.1.3: fix arm64 build failure — NeMo's `sox` dep imports numpy in its
#          legacy setup.py, which always fails in pip's isolated build env;
#          the Dockerfile now preinstalls numpy/Cython/pybind11 and installs
#          sox with --no-build-isolation before NeMo.
#   0.1.2: fully self-contained — embeds the Dockerfile AND the sidecar app
#          sources (generated block; regenerate with
#          scripts/embed_asr_sidecar.py), so a detached `curl | bash` run
#          needs nothing else hosted. ASR_REPO_GIT_URL still wins when set.
#   0.1.1: header/usage tweaks.
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
# Source resolution for the build, in order:
#   1. synced checkout — asr_sidecar/ next to this script (preferred: always
#      builds exactly what's in the repo);
#   2. ASR_REPO_GIT_URL — clone/pull a git repo containing asr_sidecar/;
#   3. detached (curl | bash) — write the EMBEDDED copy of asr_sidecar/ carried
#      at the bottom of this script (kept in sync by scripts/embed_asr_sidecar.py).
#
# Everything below is overridable via environment variables (see the block).
set -euo pipefail

# KEEP IN SYNC with the Version/Updated header lines above (printed at start
# so a curl|bash run can tell a fresh script from a stale CDN-cached copy).
INSTALLER_VERSION="0.2.0"
INSTALLER_UPDATED="7/18/2026"

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

# NeMo install. Default: git main — the nemotron-3.5 streaming model needs
# EncDecRNNTBPEModelWithPrompt, which no pypi release includes yet. Re-pin
# here (e.g. "pip install nemo_toolkit[asr]==2.x") once a release has it.
NEMO_COMMAND="${ASR_NEMO_COMMAND:-pip install 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main'}"

# Locked to 'batch' (Qwen3-ASR + ForcedAligner only) — this deployment does
# not load the Nemotron streaming engine. Reasons: (1) NeMo's dependency
# chain (torchmetrics -> transformers.AutoModel) conflicts with the
# transformers version qwen-asr needs, so 'stream'/'all' currently fail to
# load Nemotron at runtime on this box (see the Dockerfile's NEMO_COMMAND
# comment); (2) live mic transcription already falls back to the local
# faster-whisper singleton when no streaming sidecar is configured, so
# nothing regresses. ASR_VARIANT is intentionally NOT read here — a stray
# ASR_VARIANT=all in your shell env silently re-triggering the broken
# Nemotron load was the actual cause of the "AutoModel" error. To revisit
# streaming later, fix the dependency conflict (a second single-engine image
# per the Dockerfile's own comment is the documented escape hatch), then
# reintroduce the override deliberately.
VARIANT="batch"

# Detached runs only: a git repo that contains asr_sidecar/ (cloned when this
# script isn't sitting next to a synced checkout).
REPO_GIT_URL="${ASR_REPO_GIT_URL:-}"

# Model pre-fetch: 0 to skip; HF_TOKEN forwarded for gated/rate-limited repos.
DOWNLOAD_MODELS="${ASR_DOWNLOAD_MODELS:-1}"

# Persistent tuning config, read by the container via `docker run --env-file`.
# First install: written with every documented default. Re-run: existing
# lines are left EXACTLY as-is (your customizations survive an upgrade) —
# only keys that are still missing get appended. Back up before touching.
ENV_FILE="${ASR_ENV_FILE:-/etc/qwen-asr/qwen-asr.env}"
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
  BUILD_DIR="${SCRIPT_DIR}/asr_sidecar"        # script sits next to asr_sidecar/ (repo root)
  FETCH_SOURCE=0
elif [ -n "${SCRIPT_DIR}" ] && [ -f "${SCRIPT_DIR}/Dockerfile" ]; then
  BUILD_DIR="${SCRIPT_DIR}"                     # script sits INSIDE asr_sidecar/ (copied there)
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

# Every tuning var this release knows about, "KEY=default" one per line —
# the single source of truth for what ensure_env_file() writes on a fresh
# install and what it back-fills on an upgrade. Keep in sync with
# asr_sidecar/app/config.py's defaults.
_ASR_ENV_DEFAULTS='ASR_BATCH_SIZE=4
ASR_BATCH_SIZE_MAX=8
ASR_CHUNK_TARGET_SECONDS=120
ASR_CHUNK_MIN_SECONDS=30
ASR_CHUNK_MAX_SECONDS=180
ASR_CHUNK_OVERLAP_SECONDS=1.0
ASR_MIN_SILENCE_MS=500
ASR_SPEECH_PADDING_MS=250
ASR_DURATION_BUCKETING=true
ASR_CHUNK_RETRY_COUNT=2
ASR_CHUNK_RETRY_BACKOFF_SECONDS=2
ASR_GPU_METRICS_ENABLED=true
ASR_GPU_METRICS_INTERVAL_SECONDS=1
ASR_CONTENTION_POLICY=observe
ASR_PROGRESS_ENABLED=true
ASR_JOB_STATE_DIR=/models/jobs
ASR_TEMP_DIR=/tmp/qwen-asr
ASR_BENCHMARK_DIR=/models/benchmarks
ASR_MAX_UPLOAD_BYTES=2147483648
ASR_MAX_AUDIO_DURATION_SECONDS=21600
ASR_MAX_QUEUE_JOBS=8
ASR_MAX_CHUNKS_PER_JOB=2000'

# Create $ENV_FILE with every default on first install; on a re-run, only
# APPEND keys that are still missing (a customized value already present is
# never touched) and back up the file first. Also flags the deprecated
# ASR_MAX_CHUNK_S name (still read by config.py — see its migration note —
# but never auto-rewritten here, since that would be silently mutating a
# user's config).
ensure_env_file() {
  local dir; dir="$(dirname "${ENV_FILE}")"
  if ! ${SUDO} mkdir -p "${dir}" 2>/dev/null || ! ${SUDO} test -w "${dir}" 2>/dev/null; then
    warn "Cannot write ${dir} — falling back to ${DATA_DIR}/qwen-asr.env for the tuning config."
    ENV_FILE="${DATA_DIR}/qwen-asr.env"
    mkdir -p "$(dirname "${ENV_FILE}")"
  fi

  if [ ! -f "${ENV_FILE}" ]; then
    log "Writing new tuning config → ${ENV_FILE}"
    {
      echo "# qwen-asr sidecar tuning config — safe to hand-edit."
      echo "# Re-running install_asr_gb10.sh preserves every value you change here;"
      echo "# it only appends keys introduced by a newer installer version."
      echo "${_ASR_ENV_DEFAULTS}"
    } | ${SUDO} tee "${ENV_FILE}" >/dev/null
  else
    ${SUDO} cp "${ENV_FILE}" "${ENV_FILE}.bak.$(date +%s)"
    log "Existing tuning config found — backed up, appending any new keys only."
    local added=0
    while IFS= read -r line; do
      local key="${line%%=*}"
      grep -q "^${key}=" "${ENV_FILE}" 2>/dev/null || {
        echo "${line}" | ${SUDO} tee -a "${ENV_FILE}" >/dev/null
        added=$((added + 1))
      }
    done <<< "${_ASR_ENV_DEFAULTS}"
    [ "${added}" -gt 0 ] && log "Appended ${added} new tuning key(s) with their defaults." \
      || log "No new tuning keys to add — config already current."
    if grep -q '^ASR_MAX_CHUNK_S=' "${ENV_FILE}" 2>/dev/null; then
      warn "${ENV_FILE} still sets the deprecated ASR_MAX_CHUNK_S — it still works (config.py"
      warn "migrates it), but rename it to ASR_CHUNK_MAX_SECONDS when convenient."
    fi
  fi
}

# ────────────────────── embedded sidecar source (detached) ────────────────────
# __EMBEDDED_ASR_SIDECAR_BEGIN__  (generated — do not edit by hand)
# Embedded copy of asr_sidecar/ (Dockerfile + app/) so a detached
# `curl | bash` run needs nothing else hosted. KEEP IN SYNC: after
# editing asr_sidecar/, re-run:  python3 scripts/embed_asr_sidecar.py
write_embedded_source() {
  local dest="$1"
  mkdir -p "${dest}/app"
  cat > "${dest}/Dockerfile" <<'__ASR_EOF__'
# syntax=docker/dockerfile:1.7
#
# ASR sidecar for the DGX Spark / GB10 (arm64 + Blackwell): Qwen3-ASR-1.7B
# batch transcription + Nemotron 3.5 streaming ASR behind one FastAPI app.
# Built by install_asr_gb10.sh; see asr_sidecar/README.md for the API.
ARG BASE_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf

# build-essential + python3-dev: several arm64 deps (numba/llvmlite via NeMo,
# psutil, ...) have no prebuilt wheel and compile from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev python-is-python3 build-essential git \
      ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Blackwell-capable torch. A1111 on this same box proved the cu128 wheels; if
# inference fails with "no kernel image available", rebuild with the nightly
# index via --build-arg TORCH_COMMAND (see install_asr_gb10.sh trailer).
ARG TORCH_COMMAND="pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128"
RUN ${TORCH_COMMAND}

# Ubuntu 22.04's apt pip is 22.0 — old enough to silently DROP the [asr]
# extras on a direct-URL requirement like 'nemo_toolkit[asr] @ git+...'
# (observed: NeMo installed without hydra/lightning). Upgrade pip first.
RUN pip install -U pip setuptools wheel

# arm64 source-build shim: NeMo's `sox` dep (pysox, used by NeMo releases;
# dropped on main) has a legacy setup.py that imports numpy at
# metadata-generation time, which always fails inside pip's isolated build
# env. Preinstall the build helpers and install sox WITHOUT isolation (so
# its setup.py sees numpy) before NeMo resolves it.
RUN pip install numpy Cython pybind11 packaging && \
    pip install --no-build-isolation sox

# NeMo from git main: nemotron-3.5-asr-streaming needs
# EncDecRNNTBPEModelWithPrompt (rnnt_bpe_models_prompt), which no pypi
# release ships yet — the model card's documented install is @main.
# Re-pin to a release via --build-arg NEMO_COMMAND once one includes it.
ARG NEMO_COMMAND="pip install 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main'"
# Order matters: NeMo pins its stack first, qwen-asr layers on top. `pip
# check` + the import smoke test below turn any transformers/lightning
# conflict into a BUILD failure instead of a runtime one. If they ever
# conflict for real, build two images with ASR_VARIANT=batch / stream
# (the app only needs the live WS URL pointed at the second container).
# eval, not plain expansion: the shell never re-parses quotes coming out of
# a variable, and the requirement's quoted '[asr] @ url' form needs them.
RUN eval "${NEMO_COMMAND}" && \
    pip install qwen-asr && \
    pip install "fastapi>=0.115" "uvicorn[standard]" websockets python-multipart \
                silero-vad soundfile numpy huggingface_hub

# Dependency gate: fail only on real conflicts ("X requires Y ..."). pip 25's
# `pip check` also emits platform-support notes on aarch64 (torch cu128 pulls
# nvidia cu12 sub-wheels like cusparselt whose tags don't match arm64) —
# print those but don't fail; torch runs fine on this box regardless.
RUN pip check > /tmp/pipcheck 2>&1 || true; \
    cat /tmp/pipcheck; \
    ! grep -Eq ' requires | has requirement ' /tmp/pipcheck

RUN python3 -c "import nemo.collections.asr, qwen_asr, silero_vad, fastapi, multipart"

COPY app /srv/asr/app
WORKDIR /srv/asr

# Job resume manifests, benchmark output, and scratch upload/decode files —
# all under /models so they persist on the same host-mounted volume as the
# HF cache (install_asr_gb10.sh mounts DATA_DIR there). Defaults in
# app/config.py match these paths; override via ASR_JOB_STATE_DIR /
# ASR_TEMP_DIR / ASR_BENCHMARK_DIR if you want them elsewhere.
RUN mkdir -p /models/jobs /models/benchmarks /tmp/qwen-asr

VOLUME ["/models"]
EXPOSE 8790

# /ready checks model+GPU-worker+queue+disk, not just process liveness —
# start-period is generous because Qwen3-ASR + the ForcedAligner can take a
# couple of minutes to load from a cold HF cache.
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=6 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8790/ready',timeout=4).status==200 else 1)" || exit 1

ENTRYPOINT ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8790"]
__ASR_EOF__
  cat > "${dest}/app/__init__.py" <<'__ASR_EOF__'
"""ASR sidecar — GPU speech-to-text service for the transcription app.

Runs on an NVIDIA box (DGX Spark GB10) and serves:
  * Qwen3-ASR-1.7B (+ ForcedAligner) — batch transcription, OpenAI-compatible.
  * Nemotron 3.5 ASR streaming 0.6B — live captions over WebSocket.
"""
__ASR_EOF__
  cat > "${dest}/app/batch_qwen.py" <<'__ASR_EOF__'
"""Qwen3-ASR batch engine: WAV in → verbose_json-style transcript out.

Heavy imports (torch, qwen_asr) are deferred to load() so this module —
including the pure cue-grouping logic — imports cleanly on hosts without
the GPU stack.

Inference now runs through ASRInferenceScheduler (scheduler.py) instead of
a bare `with self._lock:` — see that module's docstring for the fairness
model. Progress, retry, and OOM handling live there; this module owns audio
I/O, VAD chunk planning, cue construction, resume, and boundary merging.
"""
from __future__ import annotations

import logging
import math
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

from . import boundary_merge, config, resume, vad_chunker
from .scheduler import ASRInferenceScheduler, ChunkStatus

log = logging.getLogger("asr.batch")

_CUE_MAX_GAP_S = 0.6
_CUE_MAX_LEN_S = 7.0
_CUE_MAX_CHARS = 120
_CUE_BREAK_CHARS = ".?!。？！"


def group_words_into_cues(
    words: list[dict],
    max_gap: float = _CUE_MAX_GAP_S,
    max_len_s: float = _CUE_MAX_LEN_S,
    max_chars: int = _CUE_MAX_CHARS,
    break_on: str = _CUE_BREAK_CHARS,
) -> list[dict]:
    """Group word-level timestamps into subtitle-style cues {start,end,text}.

    A cue closes on sentence-ending punctuation, a pause > max_gap, or when
    it exceeds max_len_s / max_chars. Words are {"word","start","end"} dicts.
    """
    cues: list[dict] = []
    cur: list[dict] = []

    def close() -> None:
        if not cur:
            return
        text = " ".join(w["word"].strip() for w in cur if w["word"].strip())
        if text:
            cues.append({"start": cur[0]["start"], "end": cur[-1]["end"], "text": text})
        cur.clear()

    for w in words:
        word = str(w.get("word", "")).strip()
        if not word:
            continue
        if cur:
            gap = float(w["start"]) - float(cur[-1]["end"])
            length = float(w["end"]) - float(cur[0]["start"])
            chars = sum(len(x["word"]) + 1 for x in cur)
            if gap > max_gap or length > max_len_s or chars >= max_chars:
                close()
        cur.append({"word": word, "start": float(w["start"]), "end": float(w["end"])})
        if word[-1] in break_on:
            close()
    close()
    return cues


def _to_16k_mono(src: Path) -> Path:
    """Return a 16 kHz mono WAV path for *src* (converting via ffmpeg if needed)."""
    try:
        with wave.open(str(src), "rb") as wf:
            if wf.getframerate() == config.SAMPLE_RATE and wf.getnchannels() == 1:
                return src
    except Exception:
        pass  # not a plain 16k mono WAV — convert
    dst = Path(tempfile.mkstemp(suffix=".wav", prefix="asr16k_")[1])
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", str(config.SAMPLE_RATE),
         "-ac", "1", "-f", "wav", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate() or config.SAMPLE_RATE)


def _normalize_words(time_stamps, offset: float) -> list[dict]:
    """Normalize qwen-asr forced-aligner timestamps into {word,start,end} dicts."""
    words = []
    for item in time_stamps or []:
        if isinstance(item, dict):
            d = item
        else:
            d = {k: getattr(item, k, None)
                 for k in ("word", "text", "start", "end", "start_time", "end_time")}
        word = str(d.get("word") or d.get("text") or "").strip()
        start = d.get("start", d.get("start_time"))
        end = d.get("end", d.get("end_time"))
        if not word or start is None or end is None:
            continue
        words.append({"word": word,
                      "start": float(start) + offset,
                      "end": float(end) + offset})
    return words


def _extract_confidence(res) -> float | None:
    """Best-effort per-chunk confidence from a qwen-asr result object.

    Tries an explicit confidence attribute first, then exp(mean logprob)
    from generation scores. Returns None when neither is exposed — the
    app's quality gate then falls back to heuristics-only scoring.
    """
    val = getattr(res, "confidence", None)
    if isinstance(val, (int, float)):
        return round(float(val), 4)
    logprobs = getattr(res, "logprobs", None) or getattr(res, "token_logprobs", None)
    try:
        if logprobs:
            vals = [float(x) for x in logprobs]
            return round(math.exp(sum(vals) / len(vals)), 4)
    except Exception:
        pass
    return None


def _result_to_cues(res, c_start: float, c_end: float, want_words: bool) -> tuple[list[dict], str | None]:
    text = str(getattr(res, "text", "") or "").strip()
    language = getattr(res, "language", None)
    confidence = _extract_confidence(res)
    words = _normalize_words(getattr(res, "time_stamps", None), c_start)
    if words:
        cues = group_words_into_cues(words)
    elif text:
        cues = [{"start": c_start, "end": c_end, "text": text}]
    else:
        cues = []
    for cue in cues:
        cue["confidence"] = confidence
        if want_words:
            cue["words"] = [w for w in words if cue["start"] <= w["start"] < cue["end"]]
    return cues, language


class QwenBatchEngine:
    """Wrapper around Qwen3-ASR (+ ForcedAligner) fronted by a bounded
    single-GPU-worker scheduler (see scheduler.py)."""

    def __init__(self) -> None:
        self._model = None
        self._model_lock = threading.Lock()  # protects model reference swap only, not inference
        self.scheduler = ASRInferenceScheduler(max_queue_jobs=config.ASR_MAX_QUEUE_JOBS)
        self.scheduler.start()

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        log.info("Loading %s (+aligner %s) ...", config.ASR_BATCH_MODEL, config.ASR_ALIGNER_MODEL)
        with self._model_lock:
            self._model = Qwen3ASRModel.from_pretrained(
                config.ASR_BATCH_MODEL,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                max_new_tokens=1024,
                forced_aligner=config.ASR_ALIGNER_MODEL,
                forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
            )
        log.info("Qwen3-ASR loaded.")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _infer_batch(self, pieces_with_sr: list, language: str | None):
        return self._model.transcribe(audio=pieces_with_sr, language=language, return_time_stamps=True)

    def _infer_one(self, piece, sr: int, language: str | None):
        out = self._model.transcribe(audio=(piece, sr), language=language, return_time_stamps=True)
        return out[0] if isinstance(out, (list, tuple)) else out

    def transcribe(self, audio_path: Path, language: str | None = None,
                   want_words: bool = False, timing=None,
                   batch_size: int | None = None, chunk_target_s: float | None = None,
                   bypass_resume: bool = False) -> dict:
        """Full-file transcription → OpenAI verbose_json-style dict.

        Long audio is VAD-chunked (configurable target/min/max/overlap —
        see config.py), duration-bucketed, and run through the shared
        scheduler so only one GPU forward pass runs at a time across all
        requests. *timing*, if given, is a timing.RequestTiming the caller
        marks phases into (see main.py) — this method marks the phases it
        owns (normalize/vad/chunking/queue/inference/merge).
        *batch_size* / *chunk_target_s* optionally override the configured
        defaults for this one request (used by the benchmark tool to test
        configs without a container restart — chunk_target_s is still
        clamped to [ASR_CHUNK_MIN_SECONDS, ASR_CHUNK_MAX_SECONDS], the
        container-wide ceiling main.py already validated against).
        *bypass_resume*, also for the benchmark tool: skip the resume-manifest
        lookup/write entirely, so repeated runs of the SAME file+config for
        timing purposes always genuinely re-run inference instead of the
        second+ run silently resuming (which would report near-zero
        inference time and corrupt the measurement).
        """
        if self._model is None:
            raise RuntimeError("model not loaded")

        def mark(phase, at=None):
            if timing is not None:
                timing.mark(phase, at)

        eff_chunk_target_s = chunk_target_s if chunk_target_s is not None else config.ASR_CHUNK_TARGET_SECONDS

        mark("audio_normalization_started")
        wav = _to_16k_mono(audio_path)
        duration = _wav_duration(wav)
        mark("audio_normalization_completed")

        mark("vad_started")
        chunk_boundaries = vad_chunker.plan_chunks(
            str(wav), duration, max_chunk_s=config.ASR_CHUNK_MAX_SECONDS,
            target_chunk_s=eff_chunk_target_s, min_chunk_s=config.ASR_CHUNK_MIN_SECONDS,
            overlap_s=config.ASR_CHUNK_OVERLAP_SECONDS, min_silence_ms=config.ASR_MIN_SILENCE_MS,
            speech_padding_ms=config.ASR_SPEECH_PADDING_MS,
        )
        mark("vad_completed")
        if len(chunk_boundaries) > config.ASR_MAX_CHUNKS_PER_JOB:
            raise RuntimeError(
                f"Audio would produce {len(chunk_boundaries)} chunks, over the configured "
                f"limit of {config.ASR_MAX_CHUNKS_PER_JOB} (ASR_MAX_CHUNKS_PER_JOB)")
        log.info("asr_job_started duration_s=%.1f chunks=%d chunk_target_s=%.0f",
                 duration, len(chunk_boundaries), config.ASR_CHUNK_TARGET_SECONDS)

        mark("chunk_creation_started")
        import numpy as np  # noqa: F401  (soundfile returns np arrays; kept for clarity)
        import soundfile as sf

        audio, sr = sf.read(str(wav), dtype="float32")
        pieces: list[tuple[int, float, float, object]] = []  # (orig_index, start, end, samples)
        for i, (c_start, c_end) in enumerate(chunk_boundaries):
            piece = audio[int(c_start * sr):int(c_end * sr)]
            if len(piece) < sr // 10:
                continue
            pieces.append((i, c_start, c_end, piece))
        mark("chunk_creation_completed")

        # ── resume: skip chunks already completed by a prior attempt at the
        # exact same (file, model, VAD settings, boundaries) ──
        fp = None
        manifest = None
        if not bypass_resume:
            try:
                src_hash = resume.file_hash(audio_path)
                fp = resume.compute_fingerprint(
                    source_hash=src_hash, model=config.ASR_BATCH_MODEL, sample_rate=sr,
                    chunk_boundaries=[(p[1], p[2]) for p in pieces],
                    min_silence_ms=config.ASR_MIN_SILENCE_MS, speech_padding_ms=config.ASR_SPEECH_PADDING_MS,
                    chunk_overlap_s=config.ASR_CHUNK_OVERLAP_SECONDS, language=language,
                    sidecar_version=config.VERSION,
                )
                manifest = resume.load_manifest(config.ASR_JOB_STATE_DIR, fp) or resume.new_manifest(fp, len(pieces))
            except Exception as exc:
                log.warning("asr_resume_unavailable error=%s (continuing without resume)", exc)

        done_cues: dict[int, list[dict]] = {}
        needs_infer = []
        for idx, c_start, c_end, piece in pieces:
            cid = f"chunk-{idx:06d}"
            cached = resume.completed_cues(manifest, cid) if manifest else None
            if cached is not None:
                done_cues[idx] = cached
            else:
                needs_infer.append((idx, c_start, c_end, piece))
        if manifest and len(done_cues) > 0:
            log.info("asr_resume_hit resumed_chunks=%d of=%d fingerprint=%s",
                     len(done_cues), len(pieces), fp)

        detected_language = {"v": None}
        # ChunkRecord.sequence is assigned inside submit_chunks() BEFORE the
        # job is queued, so this mapping is race-free — do not be tempted to
        # patch job.chunks[i] after submit_job() returns: the worker thread
        # can (and, under real load, does) start processing and firing
        # on_chunk_done before the submitting thread gets scheduled again,
        # dropping every chunk that finished in that window.
        seq_to_orig_idx = {local_seq: orig_idx for local_seq, (orig_idx, *_rest) in enumerate(needs_infer)}

        def on_chunk_done(chunk_rec) -> None:
            idx = seq_to_orig_idx[chunk_rec.sequence]
            if chunk_rec.status != ChunkStatus.COMPLETED:
                return
            cues, lang = _result_to_cues(chunk_rec.result, chunk_rec.absolute_start_seconds,
                                         chunk_rec.absolute_end_seconds, want_words)
            done_cues[idx] = cues
            detected_language["v"] = detected_language["v"] or lang
            if manifest is not None and fp:
                try:
                    resume.mark_chunk_complete(manifest, f"chunk-{idx:06d}", cues)
                    resume.save_manifest(config.ASR_JOB_STATE_DIR, fp, manifest)
                except Exception:
                    log.exception("asr_resume_persist_failed fingerprint=%s", fp)

        mark("queue_entered")
        if needs_infer:
            specs = [(c_start, c_end, (piece, sr)) for _, c_start, c_end, piece in needs_infer]
            eff_batch_size = batch_size or config.ASR_BATCH_SIZE
            # Reuse the HTTP request_id as the scheduler job_id (when the
            # caller gave us a timing object) so GET /v1/audio/jobs/{id}
            # works with the SAME id the X-ASR-Request-Id response header
            # returns — otherwise a client has no way to poll progress for
            # the request it's actually waiting on.
            job = self.scheduler.submit_job(
                timing.request_id if timing is not None else None, specs,
                infer_batch=lambda pieces_with_sr: self._infer_batch(pieces_with_sr, language),
                infer_one=lambda piece, srate: self._infer_one(piece, srate, language),
                batch_size=eff_batch_size, duration_bucketing=config.ASR_DURATION_BUCKETING,
                bucket_edges_s=config.ASR_DURATION_BUCKET_EDGES, sample_rate=sr,
                on_chunk_done=on_chunk_done,
                retry_count=config.ASR_CHUNK_RETRY_COUNT, retry_backoff_s=config.ASR_CHUNK_RETRY_BACKOFF_SECONDS,
            )
            mark("gpu_worker_acquired")
            mark("model_inference_started")
            result_chunks = job.future.result(timeout=config.ASR_REQUEST_TIMEOUT_SECONDS)
            mark("model_inference_completed")

            failed = [c for c in result_chunks if c.status == ChunkStatus.FAILED]
            if failed:
                log.warning("asr_job_partial_failure failed_chunks=%d of=%d",
                           len(failed), len(result_chunks))
        else:
            mark("gpu_worker_acquired")
            mark("model_inference_started")
            mark("model_inference_completed")

        mark("chunk_merge_started")
        ordered = [(pieces[j][1], pieces[j][2], done_cues.get(pieces[j][0], []))
                  for j in range(len(pieces))]
        labels = [f"chunk-{pieces[j][0]:06d}" for j in range(len(pieces))]
        segments = boundary_merge.merge_all_boundaries(ordered, labels)
        mark("chunk_merge_completed")

        if wav != audio_path:
            wav.unlink(missing_ok=True)

        for i, seg in enumerate(segments):
            seg["id"] = i
        confidences = [s["confidence"] for s in segments if s.get("confidence") is not None]

        if timing is not None:
            timing.audio_duration_s = round(duration, 2)
            timing.chunk_count = len(pieces)
            timing.batch_size = batch_size or config.ASR_BATCH_SIZE
            timing.chunk_target_s = eff_chunk_target_s
            timing.model = config.ASR_BATCH_MODEL
            if needs_infer:
                timing.batch_count = job.batches_planned

        log.info("asr_job_completed duration_s=%.1f chunks=%d segments=%d resumed_chunks=%d",
                 duration, len(pieces), len(segments), len(pieces) - len(needs_infer))

        return {
            "task": "transcribe",
            "language": str(detected_language["v"] or language or ""),
            "duration": round(duration, 3),
            "text": " ".join(s["text"] for s in segments).strip(),
            "segments": segments,
            "x_engine": config.ASR_BATCH_MODEL,
            "x_confidence_overall": (
                round(sum(confidences) / len(confidences), 4) if confidences else None
            ),
        }
__ASR_EOF__
  cat > "${dest}/app/benchmark.py" <<'__ASR_EOF__'
"""Repeatable ASR benchmark utility.

Drives a RUNNING sidecar over HTTP (stdlib urllib only — no new runtime
dependency) with a fixed audio file across a matrix of (chunk_target,
batch_size) configs, using the per-request overrides main.py accepts
(chunk_target_seconds / batch_size form fields) so no container restart is
needed between configs. Runs ONE configuration at a time (never concurrent),
with optional warm-up iterations excluded from the reported numbers.

Usage:
    python -m app.benchmark \\
      --audio /data/test-meeting.wav \\
      --url http://127.0.0.1:8790 \\
      --chunk-targets 90,120,180,280 \\
      --batch-sizes 4,6,8 \\
      --warmup-runs 1 --measured-runs 3 \\
      --output /data/results/asr-benchmark

Writes <output>.json, <output>.csv, and <output>.md (rankings + a
network-vs-compute section per the "would 10GbE help" question).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_MATRIX = {
    90: (4, 6, 8),
    120: (4, 6, 8),
    180: (4, 6),
    280: (4, 6),
}


@dataclass
class RunResult:
    chunk_target_s: int
    batch_size: int
    run_index: int  # -1 = warmup
    ok: bool
    error: str | None = None
    audio_duration_s: float | None = None
    speech_duration_s: float | None = None
    chunk_count: int | None = None
    batch_count: int | None = None
    total_ms: float | None = None
    inference_ms: float | None = None
    queue_ms: float | None = None
    upload_ms: float | None = None
    rtf: float | None = None
    speed_realtime: float | None = None
    upload_mbps: float | None = None
    transcript_length: int | None = None
    transcript_hash: str | None = None
    empty_chunk_ratio: float | None = None


@dataclass
class ConfigSummary:
    chunk_target_s: int
    batch_size: int
    runs: list[RunResult] = field(default_factory=list)

    @property
    def measured(self) -> list[RunResult]:
        return [r for r in self.runs if r.run_index >= 0 and r.ok]

    @property
    def all_ok(self) -> bool:
        return bool(self.measured) and all(r.ok for r in self.runs if r.run_index >= 0)

    def to_summary_dict(self) -> dict:
        m = self.measured
        totals = [r.total_ms for r in m if r.total_ms is not None]
        infs = [r.inference_ms for r in m if r.inference_ms is not None]
        rtfs = [r.rtf for r in m if r.rtf is not None]
        hashes = {r.transcript_hash for r in m if r.transcript_hash}
        return {
            "chunk_target_s": self.chunk_target_s,
            "batch_size": self.batch_size,
            "measured_runs": len(m),
            "failures": sum(1 for r in self.runs if r.run_index >= 0 and not r.ok),
            "mean_total_ms": round(statistics.mean(totals), 1) if totals else None,
            "mean_inference_ms": round(statistics.mean(infs), 1) if infs else None,
            "mean_rtf": round(statistics.mean(rtfs), 4) if rtfs else None,
            "transcript_consistent": len(hashes) <= 1,
            "transcript_hash_variants": len(hashes),
        }


def _post_transcription(url: str, audio_path: Path, chunk_target_s: int, batch_size: int,
                        timeout_s: float, api_key: str | None) -> tuple[dict, dict]:
    """Returns (headers_dict, body_dict). Raises on any HTTP/network error."""
    boundary = "----asrbench" + hashlib.sha1(str(time.time()).encode()).hexdigest()[:16]
    audio_bytes = audio_path.read_bytes()
    body = bytearray()

    def field(name: str, value: str) -> None:
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())

    field("response_format", "verbose_json")
    field("chunk_target_seconds", str(chunk_target_s))
    field("batch_size", str(batch_size))
    # Every benchmark call re-runs inference for real — resuming from a
    # prior identical run would report near-zero inference time and
    # invalidate the measurement.
    field("bypass_resume", "true")
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
               f"filename=\"{audio_path.name}\"\r\nContent-Type: audio/wav\r\n\r\n".encode())
    body.extend(audio_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode())

    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if api_key:
        headers["X-Api-Key"] = api_key
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/audio/transcriptions",
                                 data=bytes(body), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        resp_headers = dict(resp.headers.items())
        resp_body = json.loads(resp.read().decode("utf-8"))
    return resp_headers, resp_body


def _run_once(url: str, audio_path: Path, chunk_target_s: int, batch_size: int,
             run_index: int, timeout_s: float, api_key: str | None) -> RunResult:
    try:
        headers, body = _post_transcription(url, audio_path, chunk_target_s, batch_size, timeout_s, api_key)
    except Exception as exc:
        return RunResult(chunk_target_s=chunk_target_s, batch_size=batch_size,
                         run_index=run_index, ok=False, error=f"{type(exc).__name__}: {exc}")

    # HTTP header names are case-insensitive on the wire (uvicorn/starlette
    # send them lowercase) — normalize both sides so lookups actually hit.
    headers_lower = {k.lower(): v for k, v in headers.items()}

    def h(name: str, cast=float):
        v = headers_lower.get(name.lower())
        return cast(v) if v not in (None, "") else None

    text = body.get("text", "") or ""
    segments = body.get("segments") or []
    empty = sum(1 for s in segments if not (s.get("text") or "").strip())
    empty_ratio = round(empty / len(segments), 3) if segments else 0.0

    return RunResult(
        chunk_target_s=chunk_target_s, batch_size=batch_size, run_index=run_index, ok=True,
        audio_duration_s=h("X-ASR-Audio-Duration-S"), speech_duration_s=h("X-ASR-Speech-Duration-S"),
        chunk_count=h("X-ASR-Chunk-Count", int), batch_count=h("X-ASR-Batch-Count", int),
        total_ms=h("X-ASR-Total-Ms"), inference_ms=h("X-ASR-Inference-Ms"),
        queue_ms=h("X-ASR-Queue-Ms"), upload_ms=h("X-ASR-Upload-Ms"),
        rtf=h("X-ASR-RTF"), speed_realtime=h("X-ASR-Speed-Realtime"),
        upload_mbps=h("X-ASR-Upload-Mbps"),
        transcript_length=len(text), transcript_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
        empty_chunk_ratio=empty_ratio,
    )


def run_matrix(url: str, audio_path: Path, matrix: dict[int, tuple[int, ...]],
              warmup_runs: int, measured_runs: int, timeout_s: float,
              api_key: str | None, log=print) -> list[ConfigSummary]:
    summaries = []
    for chunk_target_s, batch_sizes in matrix.items():
        for batch_size in batch_sizes:
            log(f"== chunk_target={chunk_target_s}s batch_size={batch_size} ==")
            summary = ConfigSummary(chunk_target_s=chunk_target_s, batch_size=batch_size)
            for i in range(warmup_runs):
                log(f"  warmup {i + 1}/{warmup_runs} ...")
                r = _run_once(url, audio_path, chunk_target_s, batch_size, -1, timeout_s, api_key)
                r.run_index = -1
                summary.runs.append(r)
                if not r.ok:
                    log(f"    FAILED: {r.error}")
            for i in range(measured_runs):
                log(f"  measured {i + 1}/{measured_runs} ...")
                r = _run_once(url, audio_path, chunk_target_s, batch_size, i, timeout_s, api_key)
                summary.runs.append(r)
                if r.ok:
                    log(f"    total_ms={r.total_ms} inference_ms={r.inference_ms} rtf={r.rtf} "
                       f"speed={r.speed_realtime}x")
                else:
                    log(f"    FAILED: {r.error}")
            summaries.append(summary)
    return summaries


def rank(summaries: list[ConfigSummary]) -> list[ConfigSummary]:
    """Rank by: successful completion, transcript consistency, total wall
    time, inference RTF — per the spec, never rank on speed alone if
    transcripts diverge or chunks failed."""
    def key(s: ConfigSummary):
        d = s.to_summary_dict()
        return (
            0 if s.all_ok else 1,
            0 if d["transcript_consistent"] else 1,
            d["mean_total_ms"] if d["mean_total_ms"] is not None else float("inf"),
            d["mean_rtf"] if d["mean_rtf"] is not None else float("inf"),
        )
    return sorted(summaries, key=key)


def write_outputs(summaries: list[ConfigSummary], output_prefix: Path,
                  network_report: dict | None = None) -> None:
    ranked = rank(summaries)
    json_path = output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps({
        "configs": [{"summary": s.to_summary_dict(), "runs": [asdict(r) for r in s.runs]} for s in summaries],
        "ranking": [s.to_summary_dict() for s in ranked],
        "network": network_report,
    }, indent=2))

    csv_path = output_prefix.with_suffix(".csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["chunk_target_s", "batch_size", "run_index", "ok", "total_ms", "inference_ms",
                   "queue_ms", "rtf", "speed_realtime", "transcript_length", "transcript_hash", "error"])
        for s in summaries:
            for r in s.runs:
                w.writerow([r.chunk_target_s, r.batch_size, r.run_index, r.ok, r.total_ms, r.inference_ms,
                           r.queue_ms, r.rtf, r.speed_realtime, r.transcript_length, r.transcript_hash, r.error])

    md_path = output_prefix.with_suffix(".md")
    lines = ["# ASR Benchmark Results", "", "| Rank | Chunk Target | Batch Size | Mean Total ms | "
            "Mean Inference ms | Mean RTF | Consistent | Failures |",
            "|---|---|---|---|---|---|---|---|"]
    for i, s in enumerate(ranked, 1):
        d = s.to_summary_dict()
        lines.append(f"| {i} | {d['chunk_target_s']}s | {d['batch_size']} | {d['mean_total_ms']} | "
                    f"{d['mean_inference_ms']} | {d['mean_rtf']} | "
                    f"{'yes' if d['transcript_consistent'] else 'NO'} | {d['failures']} |")
    if ranked:
        best = ranked[0].to_summary_dict()
        lines += ["", f"**Recommended: chunk_target={best['chunk_target_s']}s, "
                     f"batch_size={best['batch_size']}** (fastest config with a consistent "
                     "transcript and zero failures among those tested)."]
    if network_report:
        lines += ["", "## Network vs. compute", "",
                  f"- Network transfer: {network_report['network_pct_of_total']}% of total job time",
                  f"- Server compute: {network_report['compute_pct_of_total']}% of total job time",
                  f"- Estimated time over 10 Gbps: {network_report['estimated_10gbe_total_s']}s "
                  f"(vs {network_report['observed_total_s']}s observed)",
                  f"- Estimated max savings from a 10 Gbps upgrade: "
                  f"{network_report['estimated_max_savings_s']}s "
                  f"({network_report['estimated_max_savings_pct']}%)",
                  "",
                  network_report["recommendation"]]
    md_path.write_text("\n".join(lines) + "\n")


def network_assessment(summaries: list[ConfigSummary], current_mbps: float = 2500.0,
                       upgrade_mbps: float = 10000.0) -> dict | None:
    """Per the "do not recommend 10GbE when upload is a negligible fraction
    of total time" requirement — computed from whichever config's measured
    runs have the most complete upload/total timing."""
    best = None
    for s in summaries:
        for r in s.measured:
            if r.upload_ms is not None and r.total_ms:
                best = r
    if best is None:
        return None
    upload_s = best.upload_ms / 1000
    total_s = best.total_ms / 1000
    compute_s = max(0.0, total_s - upload_s)
    network_pct = round(100 * upload_s / total_s, 2) if total_s else 0.0
    est_upload_at_upgrade_s = upload_s * (current_mbps / upgrade_mbps)
    est_total_at_upgrade_s = compute_s + est_upload_at_upgrade_s
    max_savings_s = round(total_s - est_total_at_upgrade_s, 2)
    max_savings_pct = round(100 * max_savings_s / total_s, 2) if total_s else 0.0
    if network_pct < 5:
        rec = (f"Upload is {network_pct}% of total job time — a 10 Gbps upgrade would NOT "
              f"materially reduce single-job completion time (estimated savings: "
              f"{max_savings_s}s, {max_savings_pct}%). The bottleneck is server-side ASR compute.")
    else:
        rec = (f"Upload is {network_pct}% of total job time — a 10 Gbps upgrade could save "
              f"roughly {max_savings_s}s ({max_savings_pct}%) per job of this size.")
    return {
        "observed_total_s": round(total_s, 2), "observed_upload_s": round(upload_s, 2),
        "network_pct_of_total": network_pct, "compute_pct_of_total": round(100 - network_pct, 2),
        "estimated_10gbe_total_s": round(est_total_at_upgrade_s, 2),
        "estimated_max_savings_s": max_savings_s, "estimated_max_savings_pct": max_savings_pct,
        "recommendation": rec,
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", required=True, type=Path)
    p.add_argument("--url", default="http://127.0.0.1:8790")
    p.add_argument("--chunk-targets", default=None,
                   help="comma-separated seconds, e.g. 90,120,180,280 "
                        "(default: the full spec matrix — see DEFAULT_MATRIX)")
    p.add_argument("--batch-sizes", default=None,
                   help="comma-separated sizes, e.g. 4,6,8 — applied uniformly to every "
                        "--chunk-targets value (default: the full spec matrix)")
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--measured-runs", type=int, default=3)
    p.add_argument("--timeout-seconds", type=float, default=3600.0)
    p.add_argument("--api-key", default=None)
    p.add_argument("--output", default="asr-benchmark", type=Path,
                   help="output path prefix — writes <prefix>.json/.csv/.md")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.audio.exists():
        print(f"error: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    if args.chunk_targets is None and args.batch_sizes is None:
        matrix = DEFAULT_MATRIX
    else:
        targets = [int(x) for x in (args.chunk_targets or "90,120,180,280").split(",") if x.strip()]
        sizes = tuple(int(x) for x in (args.batch_sizes or "4,6,8").split(",") if x.strip())
        matrix = {t: sizes for t in targets}

    print(f"Benchmarking {args.audio} against {args.url}")
    print(f"Matrix: {matrix}")
    summaries = run_matrix(args.url, args.audio, matrix, args.warmup_runs, args.measured_runs,
                           args.timeout_seconds, args.api_key)
    net = network_assessment(summaries)
    write_outputs(summaries, args.output, net)
    print(f"\nWrote {args.output}.json / .csv / .md")
    ranked = rank(summaries)
    if ranked:
        best = ranked[0].to_summary_dict()
        print(f"Recommended: chunk_target={best['chunk_target_s']}s batch_size={best['batch_size']} "
             f"(mean_total_ms={best['mean_total_ms']}, mean_rtf={best['mean_rtf']})")
    if net:
        print(net["recommendation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
__ASR_EOF__
  cat > "${dest}/app/boundary_merge.py" <<'__ASR_EOF__'
"""Duplicate-text reconciliation at overlapping chunk boundaries.

Chunks can overlap by ~ASR_CHUNK_OVERLAP_SECONDS (see vad_chunker's fixed-
window fallback), so the same few seconds of speech may be transcribed by
both the end of chunk N and the start of chunk N+1. Naively concatenating
cue text would then repeat those words. This does NOT blindly delete
repeated speech — it only trims when chunk N+1's LEADING words are a strong
token match for chunk N's TRAILING words *and* the chunks' time windows
actually overlap; anything short of that is left alone (a false negative —
a rare leftover duplicate word — is far cheaper than eating real repeated
speech, e.g. someone saying "no, no, wait").
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("asr.boundary_merge")

_WORD_RE = re.compile(r"[\w']+")
_MAX_COMPARE_TOKENS = 12
_MIN_MATCH_TOKENS = 2
_MIN_MATCH_RATIO = 0.6


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _longest_suffix_prefix_match(tail_tokens: list[str], head_tokens: list[str]) -> int:
    """Longest N such that tail_tokens[-N:] == head_tokens[:N]. 0 if none."""
    limit = min(len(tail_tokens), len(head_tokens), _MAX_COMPARE_TOKENS)
    for n in range(limit, 0, -1):
        if tail_tokens[-n:] == head_tokens[:n]:
            return n
    return 0


def _trim_leading_words(text: str, n_tokens: int) -> str:
    """Drop the first n_tokens words from *text*, preserving the rest verbatim
    (including original casing/punctuation) via a token-boundary re-split."""
    parts = text.strip().split()
    return " ".join(parts[n_tokens:]).strip()


def reconcile_boundary(prev_cues: list[dict], next_cues: list[dict],
                       prev_chunk_end: float, next_chunk_start: float,
                       boundary_label: str = "") -> list[dict]:
    """Trim a detected duplicate prefix off *next_cues[0]* when it overlaps
    the tail of *prev_cues[-1]*. Returns a (possibly modified) copy of
    next_cues; prev_cues is never modified. No-ops if the chunks don't
    actually overlap in time, or if there's no cue on either side.
    """
    if not prev_cues or not next_cues:
        return next_cues
    if next_chunk_start >= prev_chunk_end:
        return next_cues  # chunks don't overlap — nothing to reconcile

    tail_text = prev_cues[-1].get("text", "")
    head_cue = next_cues[0]
    head_text = head_cue.get("text", "")
    tail_tokens = _tokens(tail_text)
    head_tokens = _tokens(head_text)
    if not tail_tokens or not head_tokens:
        return next_cues

    n = _longest_suffix_prefix_match(tail_tokens, head_tokens)
    if n < _MIN_MATCH_TOKENS:
        return next_cues
    ratio = n / min(len(tail_tokens), len(head_tokens))
    if ratio < _MIN_MATCH_RATIO:
        return next_cues

    trimmed_text = _trim_leading_words(head_text, n)
    out = [dict(c) for c in next_cues]
    if trimmed_text:
        out[0] = {**head_cue, "text": trimmed_text}
        if head_cue.get("words"):
            out[0]["words"] = head_cue["words"][n:]
    else:
        # The whole head cue was a duplicate — drop it rather than leave an
        # empty-text cue in the transcript.
        out = out[1:]

    log.info("asr_boundary_merge boundary=%s matched_tokens=%d confidence=%.2f "
             "removed_prefix=%r", boundary_label, n, ratio,
             " ".join(head_text.split()[:n]))
    return out


def merge_all_boundaries(chunk_cue_lists: list[tuple[float, float, list[dict]]],
                         chunk_labels: list[str] | None = None) -> list[dict]:
    """chunk_cue_lists: [(chunk_start, chunk_end, cues), ...] in sequence
    order. Returns the flattened, boundary-reconciled cue list."""
    if not chunk_cue_lists:
        return []
    merged: list[dict] = list(chunk_cue_lists[0][2])
    prev_end = chunk_cue_lists[0][1]
    for i in range(1, len(chunk_cue_lists)):
        start, end, cues = chunk_cue_lists[i]
        label = (f"{chunk_labels[i - 1]} -> {chunk_labels[i]}"
                if chunk_labels and len(chunk_labels) > i else f"chunk-{i - 1} -> chunk-{i}")
        cues = reconcile_boundary(merged[-1:] if merged else [], cues, prev_end, start, label)
        merged.extend(cues)
        prev_end = end
    return merged
__ASR_EOF__
  cat > "${dest}/app/config.py" <<'__ASR_EOF__'
"""Environment-driven configuration for the ASR sidecar.

`load_config()` is a pure function (env dict in, ASRConfig out) so parsing
and validation are unit-testable without touching real env vars or process
state — see asr_sidecar/tests/test_config.py. The module-level `config.FOO`
attributes below are the actual runtime surface every other module imports
(`from . import config; config.ASR_BATCH_SIZE`) — kept for backward
compatibility with the pre-upgrade single-flat-namespace style.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields

log = logging.getLogger("asr.config")

# The ForcedAligner's documented per-call ceiling — chunks longer than this
# will fail alignment, so ASR_CHUNK_MAX_SECONDS is warned (not blocked) past it.
ALIGNER_HARD_LIMIT_S = 300.0

_VALID_CONTENTION_POLICIES = ("observe", "warn", "throttle", "exclusive")
_IMPLEMENTED_CONTENTION_POLICIES = ("observe", "warn")


class ConfigError(ValueError):
    """A present-but-invalid env var — the caller should fail startup on this."""


def _raw(env: dict, key: str) -> str | None:
    v = env.get(key)
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def _int(env: dict, key: str, default: int) -> int:
    v = _raw(env, key)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        raise ConfigError(f"{key}={v!r} is not a valid integer")


def _float(env: dict, key: str, default: float) -> float:
    v = _raw(env, key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        raise ConfigError(f"{key}={v!r} is not a valid number")


def _bool(env: dict, key: str, default: bool) -> bool:
    v = _raw(env, key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _csv_floats(env: dict, key: str, default: tuple[float, ...]) -> tuple[float, ...]:
    v = _raw(env, key)
    if v is None:
        return default
    try:
        return tuple(float(x) for x in v.split(",") if x.strip())
    except ValueError:
        raise ConfigError(f"{key}={v!r} is not a comma-separated list of numbers")


def _csv_str(env: dict, key: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    v = _raw(env, key)
    if v is None:
        return default
    return tuple(x.strip() for x in v.split(",") if x.strip())


@dataclass(frozen=True)
class ASRConfig:
    port: int
    batch_model: str
    aligner_model: str
    stream_model: str
    stream_chunk_ms: int
    variant: str

    # Chunking (replaces the old single ASR_MAX_CHUNK_S)
    chunk_target_s: float
    chunk_min_s: float
    chunk_max_s: float
    chunk_overlap_s: float
    min_silence_ms: int
    speech_padding_ms: int
    duration_bucketing: bool
    bucket_edges_s: tuple[float, ...]

    # Batch size, with a hard safety ceiling
    batch_size: int
    batch_size_max: int

    # Retry / OOM handling
    chunk_retry_count: int
    chunk_retry_backoff_s: float

    # GPU telemetry + vLLM contention observation
    gpu_metrics_enabled: bool
    gpu_metrics_interval_s: float
    contention_policy: str
    vllm_metrics_urls: tuple[str, ...]

    # Progress, state, and working directories
    progress_enabled: bool
    job_state_dir: str
    temp_dir: str
    log_dir: str
    benchmark_dir: str

    # Request/queue limits (abuse & runaway-resource guards)
    max_upload_bytes: int
    max_audio_duration_s: float
    max_queue_jobs: int
    max_chunks_per_job: int
    request_timeout_s: float

    # Optional shared-network-only auth
    api_key: str

    sample_rate: int = 16000
    version: str = "0.3.0"


def load_config(env: dict | None = None) -> ASRConfig:
    """Parse + validate env into an ASRConfig. Raises ConfigError on anything
    that should fail startup outright; clamps + warns on anything recoverable."""
    env = os.environ if env is None else env

    # ── legacy alias migration: ASR_MAX_CHUNK_S -> ASR_CHUNK_MAX_SECONDS ──
    legacy_max_chunk = _raw(env, "ASR_MAX_CHUNK_S")
    if legacy_max_chunk and not _raw(env, "ASR_CHUNK_MAX_SECONDS"):
        log.warning("ASR_MAX_CHUNK_S is deprecated (value=%s) — treating it as "
                    "ASR_CHUNK_MAX_SECONDS. Rename it in your env file; both are "
                    "read for now so nothing breaks.", legacy_max_chunk)

    chunk_min = _float(env, "ASR_CHUNK_MIN_SECONDS", 30.0)
    if chunk_min <= 0:
        raise ConfigError(f"ASR_CHUNK_MIN_SECONDS must be > 0, got {chunk_min}")

    chunk_max_default = float(legacy_max_chunk) if legacy_max_chunk else 180.0
    chunk_max = _float(env, "ASR_CHUNK_MAX_SECONDS", chunk_max_default)
    if chunk_max < chunk_min:
        raise ConfigError(
            f"ASR_CHUNK_MAX_SECONDS ({chunk_max}) must be >= ASR_CHUNK_MIN_SECONDS ({chunk_min})")
    if chunk_max > ALIGNER_HARD_LIMIT_S:
        log.warning("ASR_CHUNK_MAX_SECONDS=%s exceeds the ForcedAligner's ~%ss/call limit — "
                    "chunks that long will fail alignment and fall back to text-only cues.",
                    chunk_max, ALIGNER_HARD_LIMIT_S)

    chunk_target = _float(env, "ASR_CHUNK_TARGET_SECONDS", 120.0)
    if not (chunk_min <= chunk_target <= chunk_max):
        clamped = min(max(chunk_target, chunk_min), chunk_max)
        log.warning("ASR_CHUNK_TARGET_SECONDS=%s outside [%s, %s] — clamped to %s.",
                    chunk_target, chunk_min, chunk_max, clamped)
        chunk_target = clamped

    chunk_overlap = _float(env, "ASR_CHUNK_OVERLAP_SECONDS", 1.0)
    if chunk_overlap < 0:
        raise ConfigError(f"ASR_CHUNK_OVERLAP_SECONDS must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_min:
        log.warning("ASR_CHUNK_OVERLAP_SECONDS=%s >= ASR_CHUNK_MIN_SECONDS=%s — "
                    "chunks could degenerate to pure overlap; clamping to %s.",
                    chunk_overlap, chunk_min, chunk_min / 2)
        chunk_overlap = chunk_min / 2

    batch_size_max = _int(env, "ASR_BATCH_SIZE_MAX", 8)
    if batch_size_max < 1:
        raise ConfigError(f"ASR_BATCH_SIZE_MAX must be >= 1, got {batch_size_max}")

    batch_size = _int(env, "ASR_BATCH_SIZE", 4)
    if batch_size < 1:
        raise ConfigError(f"ASR_BATCH_SIZE must be >= 1, got {batch_size}")
    if batch_size > batch_size_max:
        log.warning("!!! ASR_BATCH_SIZE=%s exceeds ASR_BATCH_SIZE_MAX=%s — CLAMPING to %s. "
                    "Raise ASR_BATCH_SIZE_MAX only after a benchmark proves it's safe. !!!",
                    batch_size, batch_size_max, batch_size_max)
        batch_size = batch_size_max

    contention_policy = (_raw(env, "ASR_CONTENTION_POLICY") or "observe").lower()
    if contention_policy not in _VALID_CONTENTION_POLICIES:
        raise ConfigError(
            f"ASR_CONTENTION_POLICY={contention_policy!r} must be one of {_VALID_CONTENTION_POLICIES}")
    if contention_policy not in _IMPLEMENTED_CONTENTION_POLICIES:
        log.warning("ASR_CONTENTION_POLICY=%s is not implemented yet (only observe/warn are) — "
                    "behaving as 'warn'.", contention_policy)

    chunk_retry_count = _int(env, "ASR_CHUNK_RETRY_COUNT", 2)
    if chunk_retry_count < 0:
        raise ConfigError(f"ASR_CHUNK_RETRY_COUNT must be >= 0, got {chunk_retry_count}")

    max_queue_jobs = _int(env, "ASR_MAX_QUEUE_JOBS", 8)
    if max_queue_jobs < 1:
        raise ConfigError(f"ASR_MAX_QUEUE_JOBS must be >= 1, got {max_queue_jobs}")

    max_chunks_per_job = _int(env, "ASR_MAX_CHUNKS_PER_JOB", 2000)
    if max_chunks_per_job < 1:
        raise ConfigError(f"ASR_MAX_CHUNKS_PER_JOB must be >= 1, got {max_chunks_per_job}")

    return ASRConfig(
        port=_int(env, "ASR_PORT", 8790),
        batch_model=env.get("ASR_BATCH_MODEL", "Qwen/Qwen3-ASR-1.7B"),
        aligner_model=env.get("ASR_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B"),
        stream_model=env.get("ASR_STREAM_MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b"),
        stream_chunk_ms=_int(env, "ASR_STREAM_CHUNK_MS", 160),
        variant=(_raw(env, "ASR_VARIANT") or "all").lower(),

        chunk_target_s=chunk_target,
        chunk_min_s=chunk_min,
        chunk_max_s=chunk_max,
        chunk_overlap_s=chunk_overlap,
        min_silence_ms=_int(env, "ASR_MIN_SILENCE_MS", 500),
        speech_padding_ms=_int(env, "ASR_SPEECH_PADDING_MS", 250),
        duration_bucketing=_bool(env, "ASR_DURATION_BUCKETING", True),
        bucket_edges_s=_csv_floats(env, "ASR_DURATION_BUCKET_EDGES", (60.0, 90.0, 120.0, 150.0, 180.0)),

        batch_size=batch_size,
        batch_size_max=batch_size_max,

        chunk_retry_count=chunk_retry_count,
        chunk_retry_backoff_s=_float(env, "ASR_CHUNK_RETRY_BACKOFF_SECONDS", 2.0),

        gpu_metrics_enabled=_bool(env, "ASR_GPU_METRICS_ENABLED", True),
        gpu_metrics_interval_s=_float(env, "ASR_GPU_METRICS_INTERVAL_SECONDS", 1.0),
        contention_policy=contention_policy,
        vllm_metrics_urls=_csv_str(env, "ASR_VLLM_METRICS_URLS", (
            "http://127.0.0.1:8006/metrics", "http://127.0.0.1:8007/metrics")),

        progress_enabled=_bool(env, "ASR_PROGRESS_ENABLED", True),
        job_state_dir=env.get("ASR_JOB_STATE_DIR", "/models/jobs"),
        temp_dir=env.get("ASR_TEMP_DIR", "/tmp/qwen-asr"),
        log_dir=env.get("ASR_LOG_DIR", ""),
        benchmark_dir=env.get("ASR_BENCHMARK_DIR", "/models/benchmarks"),

        max_upload_bytes=_int(env, "ASR_MAX_UPLOAD_BYTES", 2 * 1024 * 1024 * 1024),
        max_audio_duration_s=_float(env, "ASR_MAX_AUDIO_DURATION_SECONDS", 6 * 3600.0),
        max_queue_jobs=max_queue_jobs,
        max_chunks_per_job=max_chunks_per_job,
        request_timeout_s=_float(env, "ASR_REQUEST_TIMEOUT_SECONDS", 3600.0 * 2),

        api_key=env.get("ASR_API_KEY", ""),
    )


# ── module-level singleton — every existing caller does `config.ASR_BATCH_SIZE` ──
_cfg = load_config()


def _apply(cfg: ASRConfig) -> None:
    g = globals()
    g["_cfg"] = cfg
    # Names existing callers already use.
    g["ASR_PORT"] = cfg.port
    g["ASR_BATCH_MODEL"] = cfg.batch_model
    g["ASR_ALIGNER_MODEL"] = cfg.aligner_model
    g["ASR_MAX_CHUNK_S"] = cfg.chunk_max_s          # legacy name, now == chunk_max_s
    g["ASR_BATCH_SIZE"] = cfg.batch_size
    g["ASR_STREAM_MODEL"] = cfg.stream_model
    g["ASR_STREAM_CHUNK_MS"] = cfg.stream_chunk_ms
    g["ASR_VARIANT"] = cfg.variant
    g["SAMPLE_RATE"] = cfg.sample_rate
    g["VERSION"] = cfg.version
    # New surface.
    g["ASR_CHUNK_TARGET_SECONDS"] = cfg.chunk_target_s
    g["ASR_CHUNK_MIN_SECONDS"] = cfg.chunk_min_s
    g["ASR_CHUNK_MAX_SECONDS"] = cfg.chunk_max_s
    g["ASR_CHUNK_OVERLAP_SECONDS"] = cfg.chunk_overlap_s
    g["ASR_MIN_SILENCE_MS"] = cfg.min_silence_ms
    g["ASR_SPEECH_PADDING_MS"] = cfg.speech_padding_ms
    g["ASR_DURATION_BUCKETING"] = cfg.duration_bucketing
    g["ASR_DURATION_BUCKET_EDGES"] = cfg.bucket_edges_s
    g["ASR_BATCH_SIZE_MAX"] = cfg.batch_size_max
    g["ASR_CHUNK_RETRY_COUNT"] = cfg.chunk_retry_count
    g["ASR_CHUNK_RETRY_BACKOFF_SECONDS"] = cfg.chunk_retry_backoff_s
    g["ASR_GPU_METRICS_ENABLED"] = cfg.gpu_metrics_enabled
    g["ASR_GPU_METRICS_INTERVAL_SECONDS"] = cfg.gpu_metrics_interval_s
    g["ASR_CONTENTION_POLICY"] = cfg.contention_policy
    g["ASR_VLLM_METRICS_URLS"] = cfg.vllm_metrics_urls
    g["ASR_PROGRESS_ENABLED"] = cfg.progress_enabled
    g["ASR_JOB_STATE_DIR"] = cfg.job_state_dir
    g["ASR_TEMP_DIR"] = cfg.temp_dir
    g["ASR_LOG_DIR"] = cfg.log_dir
    g["ASR_BENCHMARK_DIR"] = cfg.benchmark_dir
    g["ASR_MAX_UPLOAD_BYTES"] = cfg.max_upload_bytes
    g["ASR_MAX_AUDIO_DURATION_SECONDS"] = cfg.max_audio_duration_s
    g["ASR_MAX_QUEUE_JOBS"] = cfg.max_queue_jobs
    g["ASR_MAX_CHUNKS_PER_JOB"] = cfg.max_chunks_per_job
    g["ASR_REQUEST_TIMEOUT_SECONDS"] = cfg.request_timeout_s
    g["ASR_API_KEY"] = cfg.api_key


def reload() -> ASRConfig:
    """Re-read os.environ. Tests only — the running service reads env once
    at import; changing tuning values means restarting the container."""
    _apply(load_config())
    return _cfg


def current() -> ASRConfig:
    return _cfg


_apply(_cfg)

__all__ = [f.name for f in fields(ASRConfig)] + [
    "ASRConfig", "ConfigError", "load_config", "reload", "current",
    "ALIGNER_HARD_LIMIT_S",
]
__ASR_EOF__
  cat > "${dest}/app/gpu_metrics.py" <<'__ASR_EOF__'
"""Lightweight GPU/system telemetry + vLLM contention observation.

Samples `nvidia-smi` on a background thread every ASR_GPU_METRICS_INTERVAL_SECONDS
while enabled, and best-effort scrapes the two co-located vLLM services'
Prometheus /metrics endpoints (ports 8006/8007 by default) for their active
request count. Never raises into the request path — every public function
degrades to None/empty on any failure (missing nvidia-smi, vLLM down,
unreachable network, malformed output), per "do not make the service fail if
NVIDIA metrics are temporarily unavailable."

Contention policy (config.ASR_CONTENTION_POLICY):
  observe — sample and log only.
  warn    — additionally log a WARNING when ASR inference is slow (RTF above
            a threshold) while a vLLM service is showing active requests.
  throttle / exclusive — not implemented; config.py already logs that these
  fall back to 'warn' behavior. The hook point (`should_throttle()`) exists
  so a future patch can implement them without touching call sites.
"""
from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("asr.gpu_metrics")

_NVIDIA_SMI_FIELDS = (
    "utilization.gpu", "memory.used", "memory.total", "power.draw", "temperature.gpu"
)


@dataclass
class GPUSample:
    at: float
    gpu_utilization_pct: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    power_draw_w: float | None = None
    temperature_c: float | None = None


@dataclass
class VLLMSample:
    url: str
    active_requests: float | None = None
    reachable: bool = False


def sample_nvidia_smi(timeout_s: float = 3.0) -> GPUSample | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(_NVIDIA_SMI_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout_s, check=True,
        ).stdout.strip()
    except Exception as exc:
        log.debug("nvidia-smi unavailable: %s", exc)
        return None
    line = out.splitlines()[0] if out else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return None

    def f(v: str) -> float | None:
        try:
            return float(v)
        except ValueError:
            return None
    return GPUSample(
        at=time.time(), gpu_utilization_pct=f(parts[0]), memory_used_mb=f(parts[1]),
        memory_total_mb=f(parts[2]), power_draw_w=f(parts[3]), temperature_c=f(parts[4]),
    )


# vLLM's Prometheus exposition includes a line like:
#   vllm:num_requests_running{...} 3.0
_VLLM_RUNNING_RE = re.compile(r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.eE+-]+)")


def sample_vllm(url: str, timeout_s: float = 2.0) -> VLLMSample:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return VLLMSample(url=url, reachable=False)
    total = 0.0
    found = False
    for line in body.splitlines():
        m = _VLLM_RUNNING_RE.match(line)
        if m:
            try:
                total += float(m.group(1))
                found = True
            except ValueError:
                pass
    return VLLMSample(url=url, active_requests=total if found else None, reachable=True)


class GPUMetricsSampler:
    """Background sampler — call start()/stop(); read the latest sample with
    latest() or correlate a window with samples_between(t0, t1)."""

    def __init__(self, interval_s: float = 1.0, vllm_urls: tuple[str, ...] = (),
                enabled: bool = True, history_len: int = 3600) -> None:
        self.interval_s = interval_s
        self.vllm_urls = vllm_urls
        self.enabled = enabled
        self._history: list[tuple[GPUSample | None, list[VLLMSample]]] = []
        self._history_len = history_len
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="asr-gpu-metrics")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            gpu = sample_nvidia_smi()
            vllm = [sample_vllm(u) for u in self.vllm_urls]
            with self._lock:
                self._history.append((gpu, vllm))
                if len(self._history) > self._history_len:
                    self._history = self._history[-self._history_len:]
            self._stop.wait(self.interval_s)

    def latest(self) -> dict:
        with self._lock:
            if not self._history:
                return {"gpu": None, "vllm": []}
            gpu, vllm = self._history[-1]
        return {
            "gpu": None if gpu is None else vars(gpu),
            "vllm": [vars(v) for v in vllm],
        }

    def vllm_contention_active(self) -> bool:
        """True if any co-located vLLM service currently shows active requests."""
        with self._lock:
            if not self._history:
                return False
            _, vllm = self._history[-1]
        return any((v.active_requests or 0) > 0 for v in vllm)

    def summary_for_window(self, start_ts: float, end_ts: float) -> dict:
        """Peak/avg GPU utilization + memory across samples taken within
        [start_ts, end_ts] (wall-clock, time.time() units) — used to
        correlate a batch or benchmark run with what the GPU was doing."""
        with self._lock:
            in_window = [(g, vs) for g, vs in self._history if g and start_ts <= g.at <= end_ts]
            samples = [g for g, _ in in_window]
            vllm_hits = any((v.active_requests or 0) > 0 for _, vs in in_window for v in vs)
        if not samples:
            return {"peak_gpu_utilization_pct": None, "avg_gpu_utilization_pct": None,
                    "peak_memory_used_mb": None, "vllm_contention_observed": False}
        utils = [s.gpu_utilization_pct for s in samples if s.gpu_utilization_pct is not None]
        mems = [s.memory_used_mb for s in samples if s.memory_used_mb is not None]
        return {
            "peak_gpu_utilization_pct": max(utils) if utils else None,
            "avg_gpu_utilization_pct": round(sum(utils) / len(utils), 1) if utils else None,
            "peak_memory_used_mb": max(mems) if mems else None,
            "vllm_contention_observed": vllm_hits,
        }
__ASR_EOF__
  cat > "${dest}/app/main.py" <<'__ASR_EOF__'
"""ASR sidecar FastAPI app.

Endpoints (contract shared with the transcription app and the mock in
scripts/mock_asr_sidecar.py):

  GET  /healthz                   — legacy liveness + per-model load status (unchanged)
  GET  /health                    — basic process health
  GET  /ready                     — model loaded + GPU worker + queue + dirs writable
  GET  /metrics                   — Prometheus text exposition
  POST /v1/audio/transcriptions   — OpenAI-compatible batch (Qwen3-ASR); now also
                                     returns X-ASR-* timing headers (see timing.py)
                                     and accepts an optional form batch_size override
  GET  /v1/audio/jobs              — currently queued/running/recent jobs (progress)
  GET  /v1/audio/jobs/{job_id}     — one job's detail (per-chunk status)
  WS   /v1/audio/stream           — live PCM streaming (Nemotron) — unchanged

Models load in background threads at startup so the HTTP surface is up
immediately; endpoints return 503 until their engine is ready.

Backward compatibility: the request/response BODY shape of
/v1/audio/transcriptions is unchanged. Everything new is additive — response
headers a caller can ignore, and new endpoints a caller need not use. An
older client talking to this sidecar sees no difference; a newer client
talking to an older sidecar (no headers) should treat their absence as
"timing unavailable," not an error — see the main app's _remote_transcribe.
"""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from . import config, resume
from .batch_qwen import QwenBatchEngine
from .gpu_metrics import GPUMetricsSampler
from .stream_nemotron import NemotronStreamEngine
from .timing import RequestTiming, effective_mbps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("asr.main")

_batch_engine = QwenBatchEngine()
_stream_engine = NemotronStreamEngine()
_status = {"qwen3_asr": "disabled", "forced_aligner": "disabled", "nemotron_stream": "disabled"}
# Batch jobs are long; keep them off the event loop and at most 2 deep.
_BATCH_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="qwen-batch")
_gpu_sampler = GPUMetricsSampler(interval_s=config.ASR_GPU_METRICS_INTERVAL_SECONDS,
                                 vllm_urls=config.ASR_VLLM_METRICS_URLS,
                                 enabled=config.ASR_GPU_METRICS_ENABLED)

# ── Prometheus-style counters (hand-rolled — no prometheus_client dependency
# for a handful of counters/gauges; avoid high-cardinality labels like job IDs) ──
_METRICS = {
    "asr_jobs_total": 0, "asr_jobs_completed": 0, "asr_jobs_failed": 0,
    "asr_chunks_total": 0, "asr_chunks_completed": 0, "asr_chunks_failed": 0,
    "asr_requests_rejected_total": 0,
}
_METRICS_LOCK = threading.Lock()


def _bump(key: str, n: int = 1) -> None:
    with _METRICS_LOCK:
        _METRICS[key] = _METRICS.get(key, 0) + n


def _load(engine, keys: list[str]) -> None:
    for k in keys:
        _status[k] = "loading"
    try:
        engine.load()
        for k in keys:
            _status[k] = "loaded"
    except Exception as exc:
        log.exception("Engine load failed: %s", exc)
        for k in keys:
            _status[k] = f"error: {exc}"


def _load_all() -> None:
    """Load enabled engines one at a time.

    Sequential (not one thread per engine) on purpose: NeMo/transformers/qwen-asr
    use lazy module loaders that are not import-thread-safe, so concurrent loads
    surface bogus 'cannot import name X from transformers' errors. Loading in
    series also staggers CUDA allocation so peak GPU memory is one model, not two.
    """
    if config.ASR_VARIANT in ("all", "batch"):
        _load(_batch_engine, ["qwen3_asr", "forced_aligner"])
    if config.ASR_VARIANT in ("all", "stream"):
        _load(_stream_engine, ["nemotron_stream"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    for d in (config.ASR_TEMP_DIR, config.ASR_JOB_STATE_DIR, config.ASR_BENCHMARK_DIR):
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            log.warning("Could not create configured directory %s: %s", d, exc)
    _gpu_sampler.start()
    # One background thread keeps the HTTP surface (incl. /healthz) up while
    # models load; the engines within it load sequentially — see _load_all.
    threading.Thread(target=_load_all, daemon=True, name="load-asr-engines").start()
    yield
    _gpu_sampler.stop()
    _batch_engine.scheduler.shutdown(wait_s=30)


app = FastAPI(title="asr-sidecar", version=config.VERSION, lifespan=lifespan)


def _check_api_key(x_api_key: str | None) -> None:
    """Optional shared-network-only auth — see docs/asr-performance.md
    'Security' section. No-op (LAN-trust, current default) when ASR_API_KEY
    is unset, so existing deployments are unaffected."""
    if not config.ASR_API_KEY:
        return
    if x_api_key != config.ASR_API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.get("/healthz")
async def healthz() -> dict:
    gpu = ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {"status": "ok", "version": config.VERSION, "gpu": gpu, "models": dict(_status)}


@app.get("/health")
async def health() -> dict:
    """Basic process liveness — no model/GPU checks (use /ready for that)."""
    return {"status": "ok", "version": config.VERSION}


@app.get("/ready")
async def ready(response: Response) -> dict:
    checks = {
        "model_loaded": _batch_engine.loaded or config.ASR_VARIANT == "stream",
        "gpu_worker_running": _batch_engine.scheduler.is_ready(),
        "queue_accepting": _batch_engine.scheduler.is_accepting(),
        "state_dir_writable": _writable(config.ASR_JOB_STATE_DIR),
        "temp_dir_writable": _writable(config.ASR_TEMP_DIR),
    }
    ok = all(checks.values())
    response.status_code = 200 if ok else 503
    return {"ready": ok, "checks": checks}


def _writable(path: str) -> bool:
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        probe = p / f".write_probe_{uuid.uuid4().hex}"
        probe.write_text("ok")
        probe.unlink()
        return True
    except Exception:
        return False


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus text exposition. No job IDs or other high-cardinality
    labels — see the module docstring."""
    gpu_sample = _gpu_sampler.latest()
    lines = []

    def gauge(name: str, value, help_text: str) -> None:
        if value is None:
            return
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")

    def counter(name: str, value: int, help_text: str) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {value}")

    with _METRICS_LOCK:
        counter("asr_jobs_total", _METRICS["asr_jobs_total"], "ASR jobs submitted")
        counter("asr_jobs_completed", _METRICS["asr_jobs_completed"], "ASR jobs completed")
        counter("asr_jobs_failed", _METRICS["asr_jobs_failed"], "ASR jobs failed")
        counter("asr_chunks_total", _METRICS["asr_chunks_total"], "ASR chunks processed")
        counter("asr_chunks_completed", _METRICS["asr_chunks_completed"], "ASR chunks completed")
        counter("asr_chunks_failed", _METRICS["asr_chunks_failed"], "ASR chunks failed")
        counter("asr_requests_rejected_total", _METRICS["asr_requests_rejected_total"],
               "Requests rejected by a configured limit")
    gauge("asr_jobs_active", _batch_engine.scheduler.active_job_count(), "Active ASR jobs")
    gauge("asr_jobs_queued", _batch_engine.scheduler.queue_depth(), "Queued ASR jobs")
    gauge("asr_batch_size", config.ASR_BATCH_SIZE, "Configured ASR batch size")
    if gpu_sample.get("gpu"):
        g = gpu_sample["gpu"]
        gauge("asr_gpu_utilization_percent", g.get("gpu_utilization_pct"), "GPU utilization %")
        if g.get("memory_used_mb") is not None:
            gauge("asr_gpu_memory_used_bytes", g["memory_used_mb"] * 1024 * 1024, "GPU memory used, bytes")
    gauge("asr_vllm_contention_detected", 1 if _gpu_sampler.vllm_contention_active() else 0,
         "1 if a co-located vLLM service currently shows active requests")
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.get("/v1/audio/jobs")
async def list_jobs() -> dict:
    ids = _batch_engine.scheduler.list_job_ids(limit=50)
    return {"jobs": [_batch_engine.scheduler.get_job_status(jid) for jid in ids]}


@app.get("/v1/audio/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    status = _batch_engine.scheduler.get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    return status


@app.post("/v1/audio/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    ok = _batch_engine.scheduler.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found or already finished")
    return {"cancelled": True}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(""),
    response_format: str = Form("verbose_json"),
    language: str = Form(""),
    batch_size: int | None = Form(None),
    chunk_target_seconds: float | None = Form(None),
    bypass_resume: bool = Form(False),
    x_api_key: str | None = Header(None),
    x_asr_request_id: str | None = Header(None),
):
    _check_api_key(x_api_key)
    if _status["qwen3_asr"] == "loading":
        raise HTTPException(status_code=503, detail="model_loading")
    if not _batch_engine.loaded:
        raise HTTPException(status_code=503, detail=f"batch engine unavailable: {_status['qwen3_asr']}")

    request_id = x_asr_request_id or uuid.uuid4().hex
    timing = RequestTiming(request_id=request_id)
    t_recv = time.monotonic()
    timing.mark("request_received", t_recv)

    if batch_size is not None:
        if batch_size < 1 or batch_size > config.ASR_BATCH_SIZE_MAX:
            _bump("asr_requests_rejected_total")
            raise HTTPException(
                status_code=400,
                detail=f"batch_size must be between 1 and {config.ASR_BATCH_SIZE_MAX} (ASR_BATCH_SIZE_MAX)")
    if chunk_target_seconds is not None:
        if not (config.ASR_CHUNK_MIN_SECONDS <= chunk_target_seconds <= config.ASR_CHUNK_MAX_SECONDS):
            _bump("asr_requests_rejected_total")
            raise HTTPException(
                status_code=400,
                detail=f"chunk_target_seconds must be within "
                       f"[{config.ASR_CHUNK_MIN_SECONDS}, {config.ASR_CHUNK_MAX_SECONDS}] "
                       f"(ASR_CHUNK_MIN_SECONDS/ASR_CHUNK_MAX_SECONDS) — this request-time override "
                       f"cannot exceed the container's configured ceiling; restart with a higher "
                       f"ASR_CHUNK_MAX_SECONDS to benchmark past it")

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > config.ASR_MAX_UPLOAD_BYTES:
        _bump("asr_requests_rejected_total")
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds ASR_MAX_UPLOAD_BYTES ({config.ASR_MAX_UPLOAD_BYTES} bytes)")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    if suffix.lower() not in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"):
        _bump("asr_requests_rejected_total")
        raise HTTPException(status_code=415, detail=f"unsupported file extension {suffix!r}")
    # Never trust the client filename for a path — random suffix-only temp name.
    tmp = Path(tempfile.mkstemp(suffix=suffix, prefix="asr_up_", dir=config.ASR_TEMP_DIR)[1])

    body_bytes = 0
    try:
        timing.mark("request_body_read_started")
        data = await file.read()
        body_bytes = len(data)
        if body_bytes > config.ASR_MAX_UPLOAD_BYTES:
            _bump("asr_requests_rejected_total")
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds ASR_MAX_UPLOAD_BYTES ({config.ASR_MAX_UPLOAD_BYTES} bytes)")
        timing.mark("request_body_read_completed")
        tmp.write_bytes(data)
        del data
        timing.mark("temporary_file_written")

        _METRICS_LOCK.acquire()
        _METRICS["asr_jobs_total"] += 1
        _METRICS_LOCK.release()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _BATCH_EXECUTOR,
            lambda: _batch_engine.transcribe(
                tmp, language=language.strip() or None,
                want_words=response_format == "verbose_json",
                timing=timing, batch_size=batch_size, chunk_target_s=chunk_target_seconds,
                bypass_resume=bypass_resume,
            ),
        )
        _bump("asr_jobs_completed")
        _bump("asr_chunks_total", timing.chunk_count)
        _bump("asr_chunks_completed", timing.chunk_count)
    except HTTPException:
        _bump("asr_jobs_failed")
        raise
    except Exception as exc:
        _bump("asr_jobs_failed")
        log.exception("Batch transcription failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)

    timing.mark("response_serialization_started")
    if response_format == "text":
        body: object = result["text"]
        resp = PlainTextResponse(body)
    elif response_format == "json":
        resp = JSONResponse({"text": result["text"]})
    else:
        resp = JSONResponse(result)
    timing.mark("response_serialization_completed")
    timing.mark("response_sent")

    upload_ms = timing.span_ms("request_body_read_started", "request_body_read_completed")
    upload_mbps = effective_mbps(body_bytes, upload_ms / 1000 if upload_ms is not None else None)
    for k, v in timing.headers().items():
        resp.headers[k] = v
    if upload_mbps is not None:
        resp.headers["X-ASR-Upload-Mbps"] = str(upload_mbps)
    resp.headers["X-ASR-Bytes"] = str(body_bytes)

    log.info("asr_request_completed request_id=%s bytes=%d %s", request_id, body_bytes,
             " ".join(f"{k}={v}" for k, v in timing.durations_ms().items()))
    return resp


@app.websocket("/v1/audio/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    if not _stream_engine.loaded:
        await ws.send_text(json.dumps(
            {"type": "error", "message": f"stream engine unavailable: {_status['nemotron_stream']}"}
        ))
        await ws.close()
        return

    session = _stream_engine.new_session()
    loop = asyncio.get_event_loop()
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                text = await loop.run_in_executor(
                    None, _stream_engine.feed, session, msg["bytes"])
                if text:
                    await ws.send_text(json.dumps({"type": "partial", "text": text}))
                continue
            data = json.loads(msg.get("text") or "{}")
            mtype = data.get("type")
            if mtype == "start":
                await ws.send_text(json.dumps(
                    {"type": "ready", "engine": "nemotron-3.5-asr-streaming-0.6b"}))
            elif mtype == "flush":
                final = _stream_engine.flush(session)
                if final:
                    await ws.send_text(json.dumps({"type": "final", "text": final}))
            elif mtype == "stop":
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.exception("Stream session error")
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
            await ws.close()
        except Exception:
            pass
__ASR_EOF__
  cat > "${dest}/app/prefetch.py" <<'__ASR_EOF__'
"""Pre-download all sidecar models into the HF cache (idempotent).

Run inside the container with /models mounted. --entrypoint is required: the
image's ENTRYPOINT is the fixed uvicorn launch command, so without overriding
it here "python3 -m app.prefetch" would be appended as extra args to uvicorn
instead of replacing the entrypoint.
    docker run --rm --entrypoint python3 -v $DATA_DIR:/models -e HF_TOKEN $IMAGE -m app.prefetch
"""
from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

from . import config


def main() -> int:
    failed = 0
    for repo in (config.ASR_BATCH_MODEL, config.ASR_ALIGNER_MODEL, config.ASR_STREAM_MODEL):
        print(f"==> prefetch {repo}", flush=True)
        try:
            path = snapshot_download(repo_id=repo)
            print(f"    ok: {path}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"    FAILED: {exc}", file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
__ASR_EOF__
  cat > "${dest}/app/resume.py" <<'__ASR_EOF__'
"""Job resume support: persist enough metadata to skip already-completed
chunks when the same file is resubmitted (e.g. the client retried after a
network drop or the sidecar restarted mid-job).

A manifest is only reused when EVERY materially-relevant setting matches —
source file hash, model, sample rate, VAD settings, chunk boundaries,
overlap, language, and sidecar version. Any difference invalidates it (a
stale partial result is worse than reprocessing).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger("asr.resume")


def file_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_fingerprint(*, source_hash: str, model: str, sample_rate: int,
                        chunk_boundaries: list[tuple[float, float]],
                        min_silence_ms: int, speech_padding_ms: int,
                        chunk_overlap_s: float, language: str | None,
                        sidecar_version: str) -> str:
    payload = {
        "source_hash": source_hash,
        "model": model,
        "sample_rate": sample_rate,
        # rounded so float noise across runs of the same VAD plan doesn't
        # spuriously invalidate an otherwise-identical manifest
        "chunk_boundaries": [[round(a, 2), round(b, 2)] for a, b in chunk_boundaries],
        "min_silence_ms": min_silence_ms,
        "speech_padding_ms": speech_padding_ms,
        "chunk_overlap_s": round(chunk_overlap_s, 2),
        "language": language or "",
        "sidecar_version": sidecar_version,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def _manifest_path(state_dir: str, fingerprint: str) -> Path:
    return Path(state_dir) / f"{fingerprint}.json"


def load_manifest(state_dir: str, fingerprint: str) -> dict | None:
    path = _manifest_path(state_dir, fingerprint)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("asr_resume_manifest_unreadable path=%s error=%s", path, exc)
        return None


def save_manifest(state_dir: str, fingerprint: str, manifest: dict) -> None:
    """Atomic write (tmp file + rename) so a crash mid-write can't corrupt
    the manifest a later resume would read."""
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    path = _manifest_path(state_dir, fingerprint)
    fd, tmp_path = tempfile.mkstemp(dir=state_dir, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def new_manifest(fingerprint: str, chunk_count: int) -> dict:
    return {"fingerprint": fingerprint, "chunk_count": chunk_count, "chunks": {}}


def mark_chunk_complete(manifest: dict, chunk_id: str, cues: list[dict]) -> None:
    manifest["chunks"][chunk_id] = {"status": "completed", "cues": cues}


def completed_chunk_ids(manifest: dict | None) -> set[str]:
    if not manifest:
        return set()
    return {cid for cid, c in manifest.get("chunks", {}).items() if c.get("status") == "completed"}


def completed_cues(manifest: dict, chunk_id: str) -> list[dict] | None:
    entry = manifest.get("chunks", {}).get(chunk_id)
    return entry.get("cues") if entry and entry.get("status") == "completed" else None
__ASR_EOF__
  cat > "${dest}/app/scheduler.py" <<'__ASR_EOF__'
"""ASRInferenceScheduler — a single controlled GPU worker behind a job queue.

This replaces the old bare `with self._lock:` in batch_qwen.py, which
serialized every request's ENTIRE transcribe() call (including CPU-side
decode/VAD/chunking) behind one mutex with no visibility into what was
queued, no per-chunk retry, and no way to cancel. The scheduler keeps the
same single-GPU-worker safety property — see the fairness note below for why
it does NOT interleave chunks across jobs — while adding a real queue,
structured per-chunk state, retry, OOM batch-size backoff, and cancellation.

Fairness / concurrency model (read before changing):
  Jobs are served FIFO, one job's chunks fully processed before the next
  job starts. This is the "safe, preferred fallback" the design explicitly
  allows: true interleaved chunk-level fairness across concurrent jobs would
  need per-job result isolation inside a single mixed-job forward pass,
  which the underlying qwen-asr batched-transcribe call was never designed
  to guarantee (it returns a plain list positionally matched to its input
  list — safe within one job's chunks, unverified across two jobs' tensors
  sharing a batch). Job-level FIFO gets the real win (duration-aware
  batching + retry + observability) without that risk. A large job can
  still starve small jobs queued behind it; get_queue_position() exists so
  callers can at least surface "waiting behind N jobs" instead of silence.

Threading model: the worker runs on one dedicated background thread (model
inference is blocking CPU/GPU work, not async-friendly). Callers on the
asyncio side get a concurrent.futures.Future back from submit_job() and
`await asyncio.wrap_future(fut)` it.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from . import vad_chunker

log = logging.getLogger("asr.scheduler")


class ChunkStatus(str, Enum):
    PENDING = "pending"
    PREPARED = "prepared"
    QUEUED = "queued"
    BATCHED = "batched"
    PROCESSING = "processing"
    COMPLETED = "completed"
    RETRYING = "retrying"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ChunkRecord:
    chunk_id: str
    job_id: str
    sequence: int
    absolute_start_seconds: float
    absolute_end_seconds: float
    audio: object  # the decoded (samples, sr) piece — see JobSpec.audio_for()
    duration_seconds: float = 0.0
    speech_duration_seconds: float | None = None
    temporary_file: str | None = None
    status: ChunkStatus = ChunkStatus.PENDING
    attempts: int = 0
    batch_id: int | None = None
    queue_entered_at: float | None = None
    inference_started_at: float | None = None
    inference_completed_at: float | None = None
    inference_seconds: float | None = None
    text: str | None = None
    result: object = None
    error: str | None = None

    def to_public_dict(self) -> dict:
        """JSON-safe view for the progress/status endpoint — no raw audio."""
        return {
            "chunk_id": self.chunk_id, "job_id": self.job_id, "sequence": self.sequence,
            "absolute_start_seconds": self.absolute_start_seconds,
            "absolute_end_seconds": self.absolute_end_seconds,
            "duration_seconds": round(self.duration_seconds, 2),
            "speech_duration_seconds": self.speech_duration_seconds,
            "status": self.status.value, "attempts": self.attempts, "batch_id": self.batch_id,
            "inference_seconds": self.inference_seconds,
            "error": self.error,
        }


@dataclass
class JobSpec:
    job_id: str
    chunks: list[ChunkRecord]
    infer_batch: Callable[[list], list]        # [(audio,sr), ...] -> [result, ...]
    infer_one: Callable[[object, int], object]  # (audio, sr) -> result
    batch_size: int
    duration_bucketing: bool
    bucket_edges_s: tuple[float, ...]
    sample_rate: int
    on_chunk_done: Callable[[ChunkRecord], None] | None = None
    retry_count: int = 2
    retry_backoff_s: float = 2.0
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    status: JobStatus = JobStatus.QUEUED
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Future = field(default_factory=Future)
    batches_planned: int = 0
    batches_done: int = 0

    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.monotonic()
        return round(end - self.started_at, 2)

    def progress(self) -> dict:
        done = sum(1 for c in self.chunks if c.status == ChunkStatus.COMPLETED)
        failed = sum(1 for c in self.chunks if c.status == ChunkStatus.FAILED)
        total = len(self.chunks)
        pct = round(100.0 * done / total, 1) if total else 100.0
        elapsed = self.elapsed_s()
        remaining = None
        if done > 0 and done < total and elapsed > 0:
            remaining = round(elapsed / done * (total - done), 1)
        return {
            "job_id": self.job_id, "status": self.status.value,
            "chunks_total": total, "chunks_completed": done, "chunks_failed": failed,
            "batches_total": self.batches_planned, "batches_completed": self.batches_done,
            "percent_complete": pct, "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": remaining,
        }


class ASRInferenceScheduler:
    def __init__(self, min_batch_size: int = 1, max_queue_jobs: int = 8) -> None:
        self._queue: "queue.Queue[JobSpec]" = queue.Queue()
        self._jobs: dict[str, JobSpec] = {}
        self._jobs_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._accepting = True
        self._min_batch_size = min_batch_size
        self._max_queue_jobs = max_queue_jobs
        self._current_job_id: str | None = None

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._shutdown.clear()
        self._accepting = True
        self._worker_thread = threading.Thread(
            target=self.run_gpu_worker, daemon=True, name="asr-gpu-worker")
        self._worker_thread.start()

    def shutdown(self, wait_s: float = 30.0) -> None:
        """Stop accepting new jobs immediately; let the in-flight job drain,
        then stop the worker. Queued-but-not-started jobs are cancelled."""
        self._accepting = False
        with self._jobs_lock:
            for job in self._jobs.values():
                if job.status == JobStatus.QUEUED:
                    job.status = JobStatus.CANCELLED
                    job.cancel_event.set()
                    if not job.future.done():
                        job.future.set_exception(RuntimeError("scheduler shutting down"))
        self._shutdown.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=wait_s)

    # ── submission ───────────────────────────────────────────────────────
    def submit_job(self, job_id: str | None, chunk_specs: list[tuple[float, float, object]],
                   *, infer_batch: Callable, infer_one: Callable, batch_size: int,
                   duration_bucketing: bool, bucket_edges_s: tuple[float, ...],
                   sample_rate: int, on_chunk_done: Callable[[ChunkRecord], None] | None = None,
                   retry_count: int = 2, retry_backoff_s: float = 2.0) -> JobSpec:
        if not self._accepting:
            raise RuntimeError("scheduler is shutting down — not accepting new jobs")
        with self._jobs_lock:
            active = sum(1 for j in self._jobs.values()
                        if j.status in (JobStatus.QUEUED, JobStatus.RUNNING))
            if active >= self._max_queue_jobs:
                raise RuntimeError(
                    f"ASR job queue is full ({active}/{self._max_queue_jobs}) — try again shortly")
        job_id = job_id or uuid.uuid4().hex
        chunks = self.submit_chunks(job_id, chunk_specs)
        job = JobSpec(
            job_id=job_id, chunks=chunks, infer_batch=infer_batch, infer_one=infer_one,
            batch_size=batch_size, duration_bucketing=duration_bucketing,
            bucket_edges_s=bucket_edges_s, sample_rate=sample_rate,
            on_chunk_done=on_chunk_done, retry_count=retry_count, retry_backoff_s=retry_backoff_s,
        )
        with self._jobs_lock:
            self._jobs[job_id] = job
        self._queue.put(job)
        log.info("asr_job_queued job_id=%s chunks=%d queue_depth=%d",
                 job_id, len(chunks), self._queue.qsize())
        return job

    def submit_chunks(self, job_id: str,
                      chunk_specs: list[tuple[float, float, object]]) -> list[ChunkRecord]:
        """chunk_specs: [(start_s, end_s, (audio_samples, sr) or callable), ...]."""
        chunks = []
        for i, (start, end, audio) in enumerate(chunk_specs):
            chunks.append(ChunkRecord(
                chunk_id=f"chunk-{i:06d}", job_id=job_id, sequence=i,
                absolute_start_seconds=round(start, 3), absolute_end_seconds=round(end, 3),
                duration_seconds=max(0.0, end - start), audio=audio,
                status=ChunkStatus.PREPARED,
            ))
        return chunks

    # ── status / control ─────────────────────────────────────────────────
    def get_job_status(self, job_id: str) -> dict | None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        d = job.progress()
        d["chunks"] = [c.to_public_dict() for c in job.chunks]
        return d

    def get_queue_position(self, job_id: str) -> int | None:
        """0 = currently running; N = N jobs ahead of it in FIFO order."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
                return None
            if job.status == JobStatus.RUNNING:
                return 0
            ahead = sum(1 for j in self._jobs.values()
                       if j.status == JobStatus.QUEUED and j.created_at < job.created_at)
            return ahead + 1  # +1 for whatever's currently running (if anything)

    def cancel_job(self, job_id: str) -> bool:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return False
            job.cancel_event.set()
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                for c in job.chunks:
                    c.status = ChunkStatus.CANCELLED
                if not job.future.done():
                    job.future.set_exception(RuntimeError("cancelled"))
        log.info("asr_job_cancel_requested job_id=%s", job_id)
        return True

    def active_job_count(self) -> int:
        with self._jobs_lock:
            return sum(1 for j in self._jobs.values()
                      if j.status in (JobStatus.QUEUED, JobStatus.RUNNING))

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def is_ready(self) -> bool:
        return bool(self._worker_thread and self._worker_thread.is_alive())

    def is_accepting(self) -> bool:
        return self._accepting

    def list_job_ids(self, limit: int = 50) -> list[str]:
        with self._jobs_lock:
            return list(self._jobs.keys())[-limit:]

    # ── the worker ───────────────────────────────────────────────────────
    def run_gpu_worker(self) -> None:
        log.info("asr_gpu_worker_started")
        while not self._shutdown.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job.status == JobStatus.CANCELLED:
                continue
            self._run_job(job)
        log.info("asr_gpu_worker_stopped")

    def _run_job(self, job: JobSpec) -> None:
        self._current_job_id = job.job_id
        job.status = JobStatus.RUNNING
        job.started_at = time.monotonic()
        pending = [c for c in job.chunks if c.status == ChunkStatus.PREPARED]
        for c in pending:
            c.status = ChunkStatus.QUEUED
            c.queue_entered_at = time.monotonic()

        batches = vad_chunker.make_batches(
            pending, lambda c: c.duration_seconds, job.batch_size,
            job.duration_bucketing, job.bucket_edges_s)
        job.batches_planned = len(batches)

        try:
            for batch_id, batch in enumerate(batches):
                if job.cancel_event.is_set():
                    for c in batch:
                        c.status = ChunkStatus.CANCELLED
                    continue
                for c in batch:
                    c.batch_id = batch_id
                self._run_batch(job, batch_id, batch)
                job.batches_done += 1
                log.info("asr_batch_completed job_id=%s batch_id=%d of=%d chunks=%d",
                         job.job_id, batch_id + 1, job.batches_planned, len(batch))

            if job.cancel_event.is_set():
                job.status = JobStatus.CANCELLED
                if not job.future.done():
                    job.future.set_exception(RuntimeError("cancelled"))
            else:
                failed = [c for c in job.chunks if c.status == ChunkStatus.FAILED]
                job.status = JobStatus.FAILED if failed and len(failed) == len(job.chunks) else JobStatus.COMPLETED
                if not job.future.done():
                    job.future.set_result(job.chunks)
        except Exception as exc:  # never let one job's bug wedge the worker
            log.exception("asr_job_failed job_id=%s", job.job_id)
            job.status = JobStatus.FAILED
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
            job.finished_at = time.monotonic()
            self._current_job_id = None

    def _run_batch(self, job: JobSpec, batch_id: int, batch: list[ChunkRecord]) -> None:
        """Run one batch through the model, with retry + OOM size-halving.

        OOM handling never mutates job.batch_size (the configured value) —
        it locally shrinks just this failed batch and reports it, per the
        "do not permanently mutate the configured batch size" requirement.
        """
        for c in batch:
            c.status = ChunkStatus.BATCHED
        t_prep = time.monotonic()
        durations = [c.duration_seconds for c in batch]
        pad_eff = vad_chunker.padding_efficiency(durations)
        t0 = time.monotonic()
        self._infer_with_backoff(job, batch, attempt_batch_size=len(batch))
        t1 = time.monotonic()

        infer_s = round(t1 - t0, 3)
        actual_audio_s = round(sum(durations), 2)
        padded_equiv_s = round(max(durations) * len(durations), 2) if durations else 0.0
        batch_rtf = round(infer_s / actual_audio_s, 4) if actual_audio_s else None
        log.info(
            "asr_batch_stats job_id=%s batch_id=%d chunk_count=%d "
            "min_s=%.1f max_s=%.1f mean_s=%.1f actual_audio_s=%.1f "
            "padded_equivalent_s=%.1f padding_efficiency=%.3f "
            "prep_ms=%.1f inference_s=%.2f batch_rtf=%s configured_batch_size=%d",
            job.job_id, batch_id, len(batch),
            min(durations) if durations else 0.0, max(durations) if durations else 0.0,
            (actual_audio_s / len(durations)) if durations else 0.0,
            actual_audio_s, padded_equiv_s, pad_eff,
            (t0 - t_prep) * 1000, infer_s, batch_rtf, job.batch_size,
        )

    def _infer_with_backoff(self, job: JobSpec, batch: list[ChunkRecord],
                            attempt_batch_size: int) -> None:
        if not batch:
            return
        t0 = time.monotonic()
        for c in batch:
            c.status = ChunkStatus.PROCESSING
            c.inference_started_at = t0
            c.attempts += 1
        try:
            if len(batch) > 1:
                results = job.infer_batch([c.audio for c in batch])
                if not isinstance(results, (list, tuple)) or len(results) != len(batch):
                    raise RuntimeError("batched inference returned an unexpected shape")
            else:
                results = [job.infer_one(batch[0].audio, job.sample_rate)]
        except Exception as exc:
            if _looks_like_oom(exc) and len(batch) > 1:
                half = max(1, len(batch) // 2)
                log.warning("asr_batch_oom job_id=%s configured_batch_size=%d "
                           "failed_batch_size=%d retrying_with=%d",
                           job.job_id, job.batch_size, len(batch), half)
                _empty_cuda_cache()
                for c in batch:
                    c.status = ChunkStatus.RETRYING
                self._infer_with_backoff(job, batch[:half], half)
                self._infer_with_backoff(job, batch[half:], half)
                return
            self._retry_or_fail(job, batch, exc)
            return

        t1 = time.monotonic()
        for c, res in zip(batch, results):
            c.inference_completed_at = t1
            c.inference_seconds = round(t1 - c.inference_started_at, 3)
            c.result = res
            c.status = ChunkStatus.COMPLETED
            c.error = None
            if job.on_chunk_done:
                try:
                    job.on_chunk_done(c)
                except Exception:
                    log.exception("on_chunk_done callback failed for %s", c.chunk_id)

    def _retry_or_fail(self, job: JobSpec, batch: list[ChunkRecord], exc: Exception) -> None:
        retryable = _is_retryable(exc)
        remaining = job.retry_count - (batch[0].attempts - 1)
        if retryable and remaining > 0:
            backoff = job.retry_backoff_s * batch[0].attempts
            log.warning("asr_chunk_retry job_id=%s chunk_ids=%s attempt=%d backoff_s=%.1f error=%s",
                       job.job_id, [c.chunk_id for c in batch], batch[0].attempts, backoff, exc)
            for c in batch:
                c.status = ChunkStatus.RETRYING
            time.sleep(backoff)
            self._infer_with_backoff(job, batch, len(batch))
            return
        log.error("asr_chunk_failed job_id=%s chunk_ids=%s error=%s retryable=%s",
                 job.job_id, [c.chunk_id for c in batch], exc, retryable)
        for c in batch:
            c.status = ChunkStatus.FAILED
            c.error = f"{type(exc).__name__}: {exc}"
            if job.on_chunk_done:
                try:
                    job.on_chunk_done(c)
                except Exception:
                    log.exception("on_chunk_done callback failed for %s", c.chunk_id)


def _looks_like_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return "outofmemory" in name or "out of memory" in msg or "cuda oom" in msg


_NON_RETRYABLE_HINTS = (
    "invalid audio", "unsupported codec", "malformed", "invalid model",
    "unsupported sample", "corrupt",
)


def _is_retryable(exc: Exception) -> bool:
    """Transient errors (timeouts, transient CUDA errors, I/O hiccups, 5xx-ish
    internal errors) are retried; permanently-bad input is not. OOM is
    handled separately (batch-size backoff, not a plain retry)."""
    if _looks_like_oom(exc):
        return False
    msg = str(exc).lower()
    if any(hint in msg for hint in _NON_RETRYABLE_HINTS):
        return False
    retryable_hints = ("timeout", "timed out", "transient", "connection", "temporarily",
                       "cuda error", "i/o error", "ioerror", "internal server error")
    if any(hint in msg for hint in retryable_hints):
        return True
    # Default: unknown exceptions are retried once or twice rather than
    # failing a whole chunk on the first hiccup — permanently-bad input
    # typically re-raises the SAME exception, so the retry budget just
    # spends a couple of cheap attempts before falling through to FAILED.
    return True


def _empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
__ASR_EOF__
  cat > "${dest}/app/stream_nemotron.py" <<'__ASR_EOF__'
"""Nemotron 3.5 streaming ASR engine (NeMo cache-aware FastConformer-RNNT).

Two decode strategies, chosen at load time:

  * cache-aware step mode — the model's conformer_stream_step() with
    per-session encoder cache (true streaming, lowest latency);
  * rolling re-decode mode — if the cache-aware API surface is missing on
    the installed NeMo, fall back to re-transcribing the buffered audio of
    the current utterance on every step. Higher GPU cost, same protocol.

Heavy imports are deferred to load(); a global lock serializes decode steps
across sessions (the 0.6B model steps in a few ms; live sessions are few).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from . import config

log = logging.getLogger("asr.stream")


@dataclass
class StreamSession:
    """Per-WebSocket state: raw sample buffer + decode caches + text offset."""
    pending: list = field(default_factory=list)   # int16 samples not yet decoded
    utterance: list = field(default_factory=list)  # samples since last final (fallback mode)
    committed_text: str = ""                       # text already sent as final
    hypothesis: str = ""                           # current utterance hypothesis
    cache: dict | None = None                      # cache-aware tensors
    prev_hyp: object = None                        # NeMo previous_hypotheses


class NemotronStreamEngine:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._cache_aware = False
        self._chunk_samples = int(config.SAMPLE_RATE * config.ASR_STREAM_CHUNK_MS / 1000)

    def load(self) -> None:
        import nemo.collections.asr as nemo_asr

        log.info("Loading %s ...", config.ASR_STREAM_MODEL)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=config.ASR_STREAM_MODEL)
        model = model.cuda().eval()
        try:
            model.encoder.setup_streaming_params()
            self._cache_aware = hasattr(model, "conformer_stream_step")
        except Exception as exc:
            log.warning("Cache-aware streaming setup failed (%s); "
                        "using rolling re-decode mode.", exc)
            self._cache_aware = False
        self._model = model
        log.info("Nemotron streaming loaded (cache_aware=%s).", self._cache_aware)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def new_session(self) -> StreamSession:
        return StreamSession()

    def feed(self, session: StreamSession, pcm_int16) -> str | None:
        """Feed Int16 samples; returns an updated hypothesis when one decoded."""
        import numpy as np

        samples = np.frombuffer(pcm_int16, dtype=np.int16)
        session.pending.extend(samples.tolist())
        session.utterance.extend(samples.tolist())
        if len(session.pending) < self._chunk_samples:
            return None
        chunk = np.array(session.pending[:self._chunk_samples], dtype=np.int16)
        del session.pending[:self._chunk_samples]

        with self._lock:
            if self._cache_aware:
                try:
                    text = self._step_cache_aware(session, chunk)
                except Exception as exc:
                    # Self-heal: the prompt-based model's step API may differ
                    # from the classic cache-aware surface. Downgrade once to
                    # rolling re-decode instead of killing live sessions.
                    log.warning("cache-aware step failed (%s); switching to "
                                "rolling re-decode mode.", exc)
                    self._cache_aware = False
                    text = self._redecode_utterance(session)
            else:
                text = self._redecode_utterance(session)
        if text is not None and text != session.hypothesis:
            session.hypothesis = text
            return text
        return None

    def flush(self, session: StreamSession) -> str:
        """Commit the current utterance: return its text and reset for the next.

        The encoder cache is kept across finals (acoustic context carries
        over); only the text bookkeeping resets.
        """
        final = session.hypothesis.strip()
        if final:
            session.committed_text = (session.committed_text + " " + final).strip()
        session.hypothesis = ""
        session.utterance.clear()
        if self._cache_aware:
            # Text offset reset: future hypotheses are diffed against the
            # full committed text (see _step_cache_aware).
            pass
        return final

    # ── decode strategies ────────────────────────────────────────────────────

    def _step_cache_aware(self, session: StreamSession, chunk) -> str | None:
        import numpy as np
        import torch

        model = self._model
        audio = torch.from_numpy(chunk.astype(np.float32) / 32768.0).unsqueeze(0).cuda()
        length = torch.tensor([audio.shape[1]], device=audio.device)

        if session.cache is None:
            c_ch, c_t, c_len = model.encoder.get_initial_cache_state(batch_size=1)
            session.cache = {"ch": c_ch, "t": c_t, "len": c_len}

        processed, processed_len = model.preprocessor(input_signal=audio, length=length)
        with torch.no_grad():
            (pred_out, transcribed, c_ch, c_t, c_len, prev_hyp) = model.conformer_stream_step(
                processed_signal=processed,
                processed_signal_length=processed_len,
                cache_last_channel=session.cache["ch"],
                cache_last_time=session.cache["t"],
                cache_last_channel_len=session.cache["len"],
                keep_all_outputs=False,
                previous_hypotheses=session.prev_hyp,
                previous_pred_out=None,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )
        session.cache = {"ch": c_ch, "t": c_t, "len": c_len}
        session.prev_hyp = prev_hyp

        if not transcribed:
            return None
        hyp = transcribed[0]
        text = str(getattr(hyp, "text", hyp) or "").strip()
        # The stream decodes cumulatively; subtract what was already committed.
        if session.committed_text and text.startswith(session.committed_text):
            text = text[len(session.committed_text):].strip()
        return text or None

    def _redecode_utterance(self, session: StreamSession) -> str | None:
        import numpy as np

        if len(session.utterance) < config.SAMPLE_RATE // 4:
            return None
        audio = np.array(session.utterance, dtype=np.float32) / 32768.0
        try:
            # The prompt-based nemotron models take a language prompt.
            out = self._model.transcribe([audio], verbose=False, target_lang="auto")
        except TypeError:
            out = self._model.transcribe([audio], verbose=False)
        if not out:
            return None
        hyp = out[0]
        return str(getattr(hyp, "text", hyp) or "").strip() or None
__ASR_EOF__
  cat > "${dest}/app/timing.py" <<'__ASR_EOF__'
"""Phase-level request timing + real-time-factor math.

Pure, dependency-free, and unit-tested on its own — every other module
(batch_qwen, scheduler, main) records into a RequestTiming and reads its
durations/headers rather than hand-rolling timestamps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

# Every phase name the spec calls out, in the order a request normally
# passes through them. main.py/batch_qwen.py mark whichever ones apply to a
# given code path — any phase never marked contributes 0ms, not an error.
PHASES: tuple[str, ...] = (
    "request_received",
    "request_body_read_started",
    "request_body_read_completed",
    "temporary_file_written",
    "audio_decode_started",
    "audio_decode_completed",
    "audio_normalization_started",
    "audio_normalization_completed",
    "vad_started",
    "vad_completed",
    "chunk_creation_started",
    "chunk_creation_completed",
    "queue_entered",
    "gpu_worker_acquired",
    "batch_preparation_started",
    "batch_preparation_completed",
    "model_inference_started",
    "first_result_generated",
    "model_inference_completed",
    "chunk_merge_started",
    "chunk_merge_completed",
    "response_serialization_started",
    "response_serialization_completed",
    "response_sent",
)

# (start_mark, end_mark) -> duration name, for the derived spans the spec asks for.
_SPANS: tuple[tuple[str, str, str], ...] = (
    ("request_body_read_started", "request_body_read_completed", "upload_ms"),
    ("request_body_read_completed", "temporary_file_written", "file_write_ms"),
    ("audio_decode_started", "audio_decode_completed", "decode_ms"),
    ("audio_normalization_started", "audio_normalization_completed", "normalize_ms"),
    ("vad_started", "vad_completed", "vad_ms"),
    ("chunk_creation_started", "chunk_creation_completed", "chunking_ms"),
    ("queue_entered", "gpu_worker_acquired", "queue_ms"),
    ("batch_preparation_started", "batch_preparation_completed", "batch_prepare_ms"),
    ("model_inference_started", "model_inference_completed", "inference_ms"),
    ("chunk_merge_started", "chunk_merge_completed", "merge_ms"),
    ("response_serialization_started", "response_serialization_completed", "serialize_ms"),
    ("request_received", "response_sent", "total_ms"),
)


@dataclass
class RequestTiming:
    """Records wall-clock marks for one HTTP request and derives spans/headers."""
    request_id: str
    marks: dict[str, float] = field(default_factory=dict)
    audio_duration_s: float | None = None
    speech_duration_s: float | None = None
    chunk_count: int = 0
    batch_count: int = 0
    batch_size: int = 0
    chunk_target_s: float = 0.0
    model: str = ""

    def mark(self, phase: str, at: float | None = None) -> float:
        if phase not in PHASES:
            raise ValueError(f"Unknown timing phase {phase!r} — add it to timing.PHASES")
        t = at if at is not None else time.monotonic()
        # First mark wins — a retried sub-phase shouldn't erase the original
        # request_received, but callers that legitimately re-enter a phase
        # (e.g. queue_entered on a requeue) should use mark(phase, force=...)
        # semantics via a fresh RequestTiming per attempt instead.
        self.marks.setdefault(phase, t)
        return t

    def remark(self, phase: str, at: float | None = None) -> float:
        """Like mark(), but overwrites — for phases legitimately re-entered."""
        t = at if at is not None else time.monotonic()
        self.marks[phase] = t
        return t

    def span_ms(self, start_phase: str, end_phase: str) -> float | None:
        a, b = self.marks.get(start_phase), self.marks.get(end_phase)
        if a is None or b is None:
            return None
        return round((b - a) * 1000, 1)

    def durations_ms(self) -> dict[str, float]:
        out = {}
        for start, end, name in _SPANS:
            v = self.span_ms(start, end)
            if v is not None:
                out[name] = v
        return out

    def total_seconds(self) -> float | None:
        v = self.span_ms("request_received", "response_sent")
        return None if v is None else v / 1000.0

    def inference_seconds(self) -> float | None:
        v = self.span_ms("model_inference_started", "model_inference_completed")
        return None if v is None else v / 1000.0

    def rtf(self, processing_seconds: float | None = None) -> float | None:
        """RTF = processing_time / audio_duration. Defaults to total wall time;
        pass inference_seconds() for GPU-only RTF."""
        secs = processing_seconds if processing_seconds is not None else self.total_seconds()
        if secs is None or not self.audio_duration_s:
            return None
        return round(secs / self.audio_duration_s, 4)

    def speed_realtime(self, processing_seconds: float | None = None) -> float | None:
        """times_faster_than_realtime = audio_duration / processing_time."""
        secs = processing_seconds if processing_seconds is not None else self.total_seconds()
        if not secs or not self.audio_duration_s:
            return None
        return round(self.audio_duration_s / secs, 2)

    def headers(self) -> dict[str, str]:
        """X-ASR-* response headers (see docs/asr-performance.md). Every value
        is a plain string/number — never job text, never a stack trace."""
        d = self.durations_ms()
        h: dict[str, str] = {"X-ASR-Request-Id": self.request_id}

        def put(name: str, value) -> None:
            if value is not None:
                h[name] = str(value)

        put("X-ASR-Queue-Ms", d.get("queue_ms"))
        put("X-ASR-Upload-Ms", d.get("upload_ms"))
        put("X-ASR-File-Write-Ms", d.get("file_write_ms"))
        put("X-ASR-Decode-Ms", d.get("decode_ms"))
        put("X-ASR-Normalize-Ms", d.get("normalize_ms"))
        put("X-ASR-VAD-Ms", d.get("vad_ms"))
        put("X-ASR-Chunking-Ms", d.get("chunking_ms"))
        put("X-ASR-Batch-Prepare-Ms", d.get("batch_prepare_ms"))
        put("X-ASR-Inference-Ms", d.get("inference_ms"))
        put("X-ASR-Merge-Ms", d.get("merge_ms"))
        put("X-ASR-Serialize-Ms", d.get("serialize_ms"))
        put("X-ASR-Total-Ms", d.get("total_ms"))
        put("X-ASR-Audio-Duration-S", self.audio_duration_s)
        put("X-ASR-Speech-Duration-S", self.speech_duration_s)
        put("X-ASR-Chunk-Count", self.chunk_count)
        put("X-ASR-Batch-Count", self.batch_count)
        put("X-ASR-Batch-Size", self.batch_size)
        put("X-ASR-Chunk-Target-S", self.chunk_target_s)
        put("X-ASR-RTF", self.rtf())
        put("X-ASR-Speed-Realtime", self.speed_realtime())
        put("X-ASR-Model", self.model)
        return h


def effective_mbps(byte_count: int, seconds: float | None) -> float | None:
    """effective_mbps = bytes*8 / seconds / 1_000_000. None if seconds is 0/None
    (avoids a division that would otherwise read as an infeasible transfer rate)."""
    if not seconds or seconds <= 0:
        return None
    return round(byte_count * 8 / seconds / 1_000_000, 2)


def rtf(processing_seconds: float, audio_duration_seconds: float) -> float | None:
    if not audio_duration_seconds:
        return None
    return round(processing_seconds / audio_duration_seconds, 4)


def speed_realtime(processing_seconds: float, audio_duration_seconds: float) -> float | None:
    if not processing_seconds:
        return None
    return round(audio_duration_seconds / processing_seconds, 2)
__ASR_EOF__
  cat > "${dest}/app/vad_chunker.py" <<'__ASR_EOF__'
"""Long-audio chunk planning + duration-aware batch bucketing.

`pack_regions` / `bucket_chunks` / `padding_efficiency` are pure and unit
tested on any host; `plan_chunks` needs silero-vad + torch and only runs in
the container.

Config knobs (asr_sidecar/app/config.py):
  ASR_CHUNK_TARGET_SECONDS  preferred chunk length (was hard-coded ~280s)
  ASR_CHUNK_MIN_SECONDS     below this, prefer merging into a neighbor
  ASR_CHUNK_MAX_SECONDS     hard ceiling (was ASR_MAX_CHUNK_S) — the
                            ForcedAligner's ~5min/call limit lives here
  ASR_CHUNK_OVERLAP_SECONDS overlap added to fixed-window fallback slices,
                            so the boundary-merge step (batch_qwen.py) has
                            text to reconcile instead of a hard mid-word cut
  ASR_MIN_SILENCE_MS        VAD: minimum silence gap treated as a real pause
  ASR_SPEECH_PADDING_MS     VAD: padding kept around detected speech
"""
from __future__ import annotations

from dataclasses import dataclass


def pack_regions(
    regions: list[tuple[float, float]],
    total_dur: float,
    max_chunk_s: float = 280.0,
    pad_s: float = 0.2,
    target_chunk_s: float | None = None,
    min_chunk_s: float = 0.0,
    overlap_s: float = 1.0,
) -> list[tuple[float, float]]:
    """Pack VAD speech regions into transcription windows.

    *regions* are (start, end) speech spans in seconds, ascending. Consecutive
    regions are merged into one window until adding the next would exceed
    *target_chunk_s* (preferred) — falling back to *max_chunk_s* if no target
    is given, so old call sites (target_chunk_s=None) behave exactly as
    before. A window that's still under *min_chunk_s* when the input runs out
    is merged into the previous window rather than shipped as a tiny tail
    chunk. A single region longer than max_chunk_s is sliced at fixed size
    with *overlap_s* overlap (no silence available to cut at). Windows are
    padded by pad_s on each side and clamped to [0, total_dur].
    """
    total_dur = max(0.0, float(total_dur))
    if total_dur == 0.0:
        return []
    split_at = target_chunk_s if target_chunk_s is not None else max_chunk_s
    if not regions:
        return _fixed_windows(0.0, total_dur, max_chunk_s, overlap_s)

    windows: list[tuple[float, float]] = []
    cur_start: float | None = None
    cur_end = 0.0

    def close_window() -> None:
        if cur_start is None:
            return
        start = max(0.0, cur_start - pad_s)
        end = min(total_dur, cur_end + pad_s)
        if end - start > max_chunk_s:
            windows.extend(_fixed_windows(start, end, max_chunk_s, overlap_s))
        elif end > start:
            if windows and min_chunk_s > 0 and (end - start) < min_chunk_s:
                # Tiny tail/lead window — merge into the previous one instead
                # of shipping a sub-min_chunk_s chunk (wasted forward pass).
                p_start, p_end = windows[-1]
                windows[-1] = (p_start, max(p_end, end))
            else:
                windows.append((start, end))

    for r_start, r_end in regions:
        r_start, r_end = float(r_start), float(r_end)
        if r_end <= r_start:
            continue
        if cur_start is None:
            cur_start, cur_end = r_start, r_end
        elif r_end - cur_start <= split_at:
            cur_end = r_end
        else:
            close_window()
            cur_start, cur_end = r_start, r_end
    close_window()
    return windows


def _fixed_windows(start: float, end: float, max_chunk_s: float,
                   overlap_s: float = 1.0) -> list[tuple[float, float]]:
    """Fixed-size fallback slicing when no silence gap is available."""
    windows = []
    step = max(1.0, max_chunk_s - overlap_s)
    t = start
    while t < end:
        windows.append((t, min(end, t + max_chunk_s)))
        t += step
    return windows


def plan_chunks(wav_path: str, total_dur: float,
                max_chunk_s: float = 280.0,
                target_chunk_s: float | None = None,
                min_chunk_s: float = 0.0,
                overlap_s: float = 1.0,
                min_silence_ms: int = 500,
                speech_padding_ms: int = 250) -> list[tuple[float, float]]:
    """VAD-based chunk plan for *wav_path* (16 kHz mono WAV).

    Falls back to fixed windows when silero-vad is unavailable or errors —
    a degraded cut point is better than a failed transcription.
    """
    try:
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

        model = load_silero_vad()
        audio = read_audio(wav_path, sampling_rate=16000)
        stamps = get_speech_timestamps(
            audio, model, sampling_rate=16000,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_padding_ms,
        )
        regions = [(s["start"] / 16000.0, s["end"] / 16000.0) for s in stamps]
        return pack_regions(regions, total_dur, max_chunk_s=max_chunk_s,
                            target_chunk_s=target_chunk_s, min_chunk_s=min_chunk_s,
                            overlap_s=overlap_s)
    except Exception:
        return _fixed_windows(0.0, total_dur, max_chunk_s, overlap_s)


# ── Duration-aware batch bucketing ──────────────────────────────────────────
# The model pads every sample in a batch to the longest one — grouping a
# 280s chunk with a 75s chunk wastes ~73% of the short chunk's forward-pass
# compute on padding. Bucketing groups same-ish-duration chunks first so
# batches are built from a bucket (falling back to whatever's left over).

@dataclass(frozen=True)
class Bucket:
    lo: float
    hi: float  # inclusive upper edge; float("inf") for the overflow bucket


def build_buckets(edges_s: tuple[float, ...]) -> list[Bucket]:
    """edges_s=(60,90,120,150,180) -> [0-60],[60-90],[90-120],[120-150],[150-180],[180-inf]."""
    edges = sorted(edges_s)
    lo = 0.0
    buckets = []
    for hi in edges:
        buckets.append(Bucket(lo, hi))
        lo = hi
    buckets.append(Bucket(lo, float("inf")))
    return buckets


def bucket_index(duration_s: float, buckets: list[Bucket]) -> int:
    for i, b in enumerate(buckets):
        if duration_s <= b.hi:
            return i
    return len(buckets) - 1


def bucket_chunks(items: list, duration_fn, edges_s: tuple[float, ...]) -> list[list]:
    """Group *items* (anything with a duration, via duration_fn) into
    same-bucket groups, each internally sorted by duration ascending and in
    original relative order across buckets (bucket 0's items all come before
    bucket 1's) — a stable, deterministic ordering so results are easy to
    reason about and test."""
    buckets = build_buckets(edges_s)
    grouped: list[list] = [[] for _ in buckets]
    for item in items:
        grouped[bucket_index(duration_fn(item), buckets)].append(item)
    for g in grouped:
        g.sort(key=duration_fn)
    return [g for g in grouped if g]


def make_batches(items: list, duration_fn, batch_size: int,
                 duration_bucketing: bool, edges_s: tuple[float, ...]) -> list[list]:
    """Split *items* into batches of at most batch_size, optionally
    duration-bucketed first so each batch pads as little as possible."""
    if batch_size < 1:
        batch_size = 1
    if duration_bucketing:
        ordered: list = []
        for group in bucket_chunks(items, duration_fn, edges_s):
            ordered.extend(group)
    else:
        ordered = list(items)
    return [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]


def padding_efficiency(durations_s: list[float]) -> float:
    """padding_efficiency = sum(durations) / (max(durations) * count).

    1.0 = every chunk in the batch is the same length (no wasted padding);
    lower = more compute spent on padding than real audio. Returns 1.0 for
    an empty/degenerate input (nothing to pad).
    """
    if not durations_s:
        return 1.0
    total = sum(durations_s)
    longest = max(durations_s)
    if longest <= 0:
        return 1.0
    return round(total / (longest * len(durations_s)), 4)
__ASR_EOF__
}
# __EMBEDDED_ASR_SIDECAR_END__

# ─────────────────────────────── banner ──────────────────────────────────────
printf '%s\n' "${c_b}install_asr_gb10.sh v${INSTALLER_VERSION} (${INSTALLER_UPDATED})${c_0}"
log "image=${IMAGE}  container=${CONTAINER_NAME}  port=${PORT}  variant=${VARIANT}"

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
  if [ -n "${REPO_GIT_URL}" ]; then
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
    log "Detached run — writing embedded sidecar source → ${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    write_embedded_source "${BUILD_DIR}"
  fi
else
  log "Building from synced repo at ${BUILD_DIR}"
fi
cd "${BUILD_DIR}"
log "base image:  ${BASE_IMAGE}"
log "torch:       ${TORCH_COMMAND}"
log "nemo:        ${NEMO_COMMAND}"
log "variant:     ${VARIANT}"
${SUDO} docker build --pull \
  -f Dockerfile \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "TORCH_COMMAND=${TORCH_COMMAND}" \
  --build-arg "NEMO_COMMAND=${NEMO_COMMAND}" \
  -t "${IMAGE}" . \
  || die "Image build failed — scroll up for the first real error.
    • A dependency failed building from source (arm64 has no wheel for it,
      e.g. 'ModuleNotFoundError' during metadata generation)? Add it to the
      no-build-isolation shim in the Dockerfile — or sidestep every NeMo
      build issue at once with NVIDIA's prebuilt NeMo base image:
      ASR_BASE_IMAGE=nvcr.io/nvidia/nemo:25.04 ASR_TORCH_COMMAND=true ./install_asr_gb10.sh
    • 'pip check' / the import smoke test failed (NeMo × qwen-asr version
      conflict)? This deployment only loads the batch (Qwen3-ASR) engine at
      runtime (see the VARIANT comment above) — NeMo is still built into the
      image for the smoke test / future streaming work, so a conflict here
      is a build-time issue, not something ASR_VARIANT can route around
      anymore. Options: pin transformers to a version both stacks accept, or
      drop NEMO_COMMAND/the NeMo RUN steps from the Dockerfile entirely if
      streaming won't be revisited soon."
log "Built ${IMAGE}."

# ─────────────────────────── 7. pre-fetch models ─────────────────────────────
step "7/9  Pre-fetch HF models into ${DATA_DIR} (idempotent)"
mkdir -p "${DATA_DIR}"
if [ "${DOWNLOAD_MODELS}" != "0" ]; then
  [ -z "${HF_TOKEN}" ] && warn "HF_TOKEN not set — public repos still work; set it if a download 401s/429s."
  # --entrypoint python3: the image's ENTRYPOINT is the fixed uvicorn launch
  # command (exec form) — without this, "python3 -m app.prefetch" below is
  # appended as extra CLI ARGS to uvicorn itself instead of replacing the
  # entrypoint, which uvicorn's arg parser rejects ("Error: No such option
  # '-m'"). This overrides the entrypoint for this one-off run only; the
  # main container in step 8 is unaffected.
  ${SUDO} docker run --rm \
    --entrypoint python3 \
    -v "${DATA_DIR}:/models" \
    ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
    "${IMAGE}" -m app.prefetch \
    || warn "Model pre-fetch reported failures — the container will retry lazily at startup."
else
  log "ASR_DOWNLOAD_MODELS=0 — skipping model pre-fetch."
fi

# ──────────────────────────────── 8. run ─────────────────────────────────────
step "8/9  (Re)start the container"
ensure_env_file
if ${SUDO} docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  log "Removing previous container '${CONTAINER_NAME}' …"
  ${SUDO} docker rm -f "${CONTAINER_NAME}" >/dev/null
fi
# Also stop/remove ANY other container already publishing this host port —
# a leftover from a manual test or an earlier install under a different
# name would otherwise keep answering requests on $PORT after this script
# reports success, silently serving old code with nothing in the install
# log to explain it (this is exactly what made an earlier run look like it
# deployed new code when it hadn't).
OTHER_ON_PORT="$(${SUDO} docker ps -a --format '{{.ID}}\t{{.Names}}\t{{.Ports}}' \
  | grep -E "(^|[^0-9])${PORT}->" | awk -F'\t' -v n="${CONTAINER_NAME}" '$2 != n {print $1}')"
if [ -n "${OTHER_ON_PORT}" ]; then
  for cid in ${OTHER_ON_PORT}; do
    cname="$(${SUDO} docker inspect -f '{{.Name}}' "${cid}" 2>/dev/null | sed 's#^/##')"
    warn "Container '${cname:-$cid}' is already publishing port ${PORT} — stopping and removing it."
    ${SUDO} docker rm -f "${cid}" >/dev/null
  done
fi
# --shm-size: torch's DataLoader-style multi-worker paths use /dev/shm; the
# 64m default is too small once batching runs several chunks concurrently.
# --log-opt: batch runs can log a lot over hours-long uptime; cap it instead
# of letting the default json-file driver grow unbounded.
${SUDO} docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --gpus all \
  -p "0.0.0.0:${PORT}:8790" \
  -v "${DATA_DIR}:/models" \
  --env-file "${ENV_FILE}" \
  -e "ASR_VARIANT=${VARIANT}" \
  --shm-size=2g \
  --log-opt max-size=10m --log-opt max-file=5 \
  ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
  "${IMAGE}" >/dev/null
log "Container started (tuning config: ${ENV_FILE})."

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
  READY_OK=0
  if curl -fsS "http://127.0.0.1:${PORT}/ready" 2>/dev/null | grep -q '"ready":true'; then
    READY_OK=1
    log "/ready: all checks passing (model, GPU worker, queue, disk) ✅"
  else
    warn "/ready is not all-green yet — check: ${SUDO:+sudo }docker logs ${CONTAINER_NAME}"
  fi
  SMOKE_RTF=""
  SMOKE_SPEED=""
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
    SMOKE_HEADERS=/tmp/asr_test_headers_$$.txt
    if curl -fsS -D "${SMOKE_HEADERS}" -F "file=@${TESTWAV}" -F "response_format=verbose_json" \
         "http://127.0.0.1:${PORT}/v1/audio/transcriptions" | grep -q '"segments"'; then
      log "Batch transcription endpoint responds ✅"
      # Headers arrive lowercase on the wire; a 2s tone is far too short to be
      # a meaningful RTF sample (fixed model-load overhead dominates) — this
      # only proves the timing plumbing works end-to-end, not real throughput.
      SMOKE_RTF="$(grep -i '^x-asr-rtf:' "${SMOKE_HEADERS}" | tr -d '\r' | cut -d' ' -f2)"
      SMOKE_SPEED="$(grep -i '^x-asr-speed-realtime:' "${SMOKE_HEADERS}" | tr -d '\r' | cut -d' ' -f2)"
      [ -n "${SMOKE_RTF}" ] && log "Smoke-test RTF=${SMOKE_RTF} speed=${SMOKE_SPEED}x realtime (2s tone — not representative; run a real benchmark, see below)."
    else
      warn "Batch endpoint didn't return segments — check: ${SUDO:+sudo }docker logs ${CONTAINER_NAME}"
    fi
    rm -f "${TESTWAV}" "${SMOKE_HEADERS}"
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
BATCH_SIZE_SUMMARY="$(grep '^ASR_BATCH_SIZE=' "${ENV_FILE}" 2>/dev/null | tail -1 | cut -d= -f2)"
CHUNK_TARGET_SUMMARY="$(grep '^ASR_CHUNK_TARGET_SECONDS=' "${ENV_FILE}" 2>/dev/null | tail -1 | cut -d= -f2)"
BUCKETING_SUMMARY="$(grep '^ASR_DURATION_BUCKETING=' "${ENV_FILE}" 2>/dev/null | tail -1 | cut -d= -f2)"
GPU_METRICS_SUMMARY="$(grep '^ASR_GPU_METRICS_ENABLED=' "${ENV_FILE}" 2>/dev/null | tail -1 | cut -d= -f2)"
cat <<EOF

${c_b}Done.${c_0}
  Local:            http://127.0.0.1:${PORT}/healthz
$( [ -n "${IP}" ] && printf '  LAN:              http://%s:%s   (use this URL from the app hosts)\n' "${IP}" "${PORT}" )
  Model:            qwen3-asr-1.7b (variant=${VARIANT})
  Batch size:       ${BATCH_SIZE_SUMMARY:-unknown}  (max: $(grep '^ASR_BATCH_SIZE_MAX=' "${ENV_FILE}" 2>/dev/null | tail -1 | cut -d= -f2))
  Chunk target:     ${CHUNK_TARGET_SUMMARY:-unknown}s
  Duration bucketing: ${BUCKETING_SUMMARY:-unknown}
  GPU metrics:      ${GPU_METRICS_SUMMARY:-unknown}
  /ready:           $( [ "${READY_OK:-0}" -eq 1 ] && echo "passing ✅" || echo "NOT all-green — check logs" )
  Smoke-test RTF:   ${SMOKE_RTF:-n/a} (speed ${SMOKE_SPEED:-n/a}x realtime — 2s tone, not a benchmark)
  Tuning config:    ${ENV_FILE}  (hand-editable; re-run this script to add new keys without losing edits)
  Logs:             ${SUDO:+sudo }docker logs -f ${CONTAINER_NAME}
  Stop:             ${SUDO:+sudo }docker rm -f ${CONTAINER_NAME}
  Update:           re-run this script (rebuilds with --pull, recreates the container)
  Models:           ${DATA_DIR}  (HF cache; safe to keep across rebuilds)

${c_b}Run a real throughput benchmark${c_0} (against this running container, from any host that can
reach it — needs a representative audio file, ideally the one that was slow):
  python3 -m asr_sidecar.benchmark --url http://${IP:-127.0.0.1}:${PORT} \\
    --audio /path/to/meeting.m4a --output /tmp/asr_bench
  (defaults to the standard chunk-target × batch-size matrix; add --chunk-targets /
   --batch-sizes to narrow it. Writes .json/.csv/.md summaries with RTF per config.)

${c_b}Point the transcription app at this box${c_0} (on the app host, e.g. ai2-strixhelo):
  Option A — AI Providers page: add an 'openai_compatible' connection with
    base_url http://${IP:-<dgx-ip>}:${PORT} named "ASR Sidecar (DGX)", then route
    task 'transcription' → qwen3-asr-1.7b. This deployment does not load the
    Nemotron streaming engine (batch-only, see VARIANT in this script) — leave
    'live_transcription' unrouted so live mic capture uses the local
    faster-whisper fallback.
  Option B — env vars on the app host:
    ASR_SIDECAR_URL=http://${IP:-<dgx-ip>}:${PORT}
    LIVE_ASR_WS_URL=ws://${IP:-<dgx-ip>}:${PORT}/v1/audio/stream
  Quality gate tuning (app host): ASR_QUALITY_THRESHOLD=0.90 (default),
    ASR_DUAL_TRANSCRIPTION=1 to archive whisper output alongside every run.

${c_b}Upgrading from a pre-tuning-config install?${c_0} This run created ${ENV_FILE} with
defaults and wired it into the container via --env-file — nothing you had running changes
behavior. To roll back to the old fixed-batch-size behavior, stop the container and re-run
the previous version of this script, or set ASR_DURATION_BUCKETING=false and
ASR_CHUNK_RETRY_COUNT=0 in ${ENV_FILE} and restart.

If generation errors with "no kernel image available for execution on the device",
the GB10 needs newer kernels than cu128 stable — re-run with the nightly wheel:
  ASR_TORCH_COMMAND="pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128" ${SUDO:+sudo }./install_asr_gb10.sh
EOF
