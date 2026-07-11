#!/usr/bin/env bash
#    By: Christopher Gray
#    Version: 0.1.8
#    Updated: 7/11/2026
#
#    This script installs the ASR sidecar on an NVIDIA DGX Spark / GB10 (arm64 + Blackwell).
#
#    curl -sSL https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_asr_gb10.sh | bash
#
#   Installer:
#     ./install_asr_gb10.sh        (from a synced checkout — preferred)
#
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
INSTALLER_VERSION="0.1.8"
INSTALLER_UPDATED="7/11/2026"

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

VOLUME ["/models"]
EXPOSE 8790
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
"""
from __future__ import annotations

import logging
import math
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

from . import config, vad_chunker

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


class QwenBatchEngine:
    """Singleton-style wrapper around Qwen3-ASR (+ ForcedAligner)."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()  # serialize GPU inference across requests

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        log.info("Loading %s (+aligner %s) ...", config.ASR_BATCH_MODEL, config.ASR_ALIGNER_MODEL)
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

    def transcribe(self, audio_path: Path, language: str | None = None,
                   want_words: bool = False) -> dict:
        """Full-file transcription → OpenAI verbose_json-style dict.

        Long audio is VAD-chunked under the aligner's 5-minute limit; each
        chunk contributes its confidence (exp(mean token logprob) when the
        qwen-asr result exposes it, else null) to every cue it produced.
        """
        if self._model is None:
            raise RuntimeError("model not loaded")

        wav = _to_16k_mono(audio_path)
        duration = _wav_duration(wav)
        chunks = vad_chunker.plan_chunks(str(wav), duration, config.ASR_MAX_CHUNK_S)
        log.info("Transcribing %.1fs of audio in %d chunk(s).", duration, len(chunks))

        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(str(wav), dtype="float32")
        segments: list[dict] = []
        detected_language = None

        with self._lock:
            for c_start, c_end in chunks:
                piece = audio[int(c_start * sr):int(c_end * sr)]
                if len(piece) < sr // 10:
                    continue
                results = self._model.transcribe(
                    audio=(piece, sr),
                    language=language,
                    return_time_stamps=True,
                )
                res = results[0] if isinstance(results, (list, tuple)) else results
                text = str(getattr(res, "text", "") or "").strip()
                detected_language = detected_language or getattr(res, "language", None)
                confidence = self._extract_confidence(res)
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
                        cue["words"] = [w for w in words
                                        if cue["start"] <= w["start"] < cue["end"]]
                segments.extend(cues)

        if wav != audio_path:
            wav.unlink(missing_ok=True)

        for i, seg in enumerate(segments):
            seg["id"] = i
        confidences = [s["confidence"] for s in segments if s.get("confidence") is not None]
        return {
            "task": "transcribe",
            "language": str(detected_language or language or ""),
            "duration": round(duration, 3),
            "text": " ".join(s["text"] for s in segments).strip(),
            "segments": segments,
            "x_engine": "qwen3-asr-1.7b",
            "x_confidence_overall": (
                round(sum(confidences) / len(confidences), 4) if confidences else None
            ),
        }

    @staticmethod
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
__ASR_EOF__
  cat > "${dest}/app/config.py" <<'__ASR_EOF__'
"""Environment-driven configuration for the ASR sidecar."""
from __future__ import annotations

import os

ASR_PORT = int(os.environ.get("ASR_PORT", "8790"))

# Batch engine (uploaded WAVs / completed meetings)
ASR_BATCH_MODEL = os.environ.get("ASR_BATCH_MODEL", "Qwen/Qwen3-ASR-1.7B")
ASR_ALIGNER_MODEL = os.environ.get("ASR_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
# The ForcedAligner supports up to 5 minutes per call; chunk under it.
ASR_MAX_CHUNK_S = float(os.environ.get("ASR_MAX_CHUNK_S", "280"))

# Streaming engine (live microphone captions)
ASR_STREAM_MODEL = os.environ.get("ASR_STREAM_MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b")
# Cache-aware decode step size; valid model configs range 80 ms – 1.12 s.
ASR_STREAM_CHUNK_MS = int(os.environ.get("ASR_STREAM_CHUNK_MS", "160"))

# Which engines to load: all | batch | stream. Lets one image run as two
# containers if the NeMo and qwen-asr dependency stacks ever conflict.
ASR_VARIANT = os.environ.get("ASR_VARIANT", "all").strip().lower()

SAMPLE_RATE = 16000
VERSION = "0.1.1"
__ASR_EOF__
  cat > "${dest}/app/main.py" <<'__ASR_EOF__'
"""ASR sidecar FastAPI app.

Endpoints (contract shared with the transcription app and the mock in
scripts/mock_asr_sidecar.py):

  GET  /healthz                   — liveness + per-model load status
  POST /v1/audio/transcriptions   — OpenAI-compatible batch (Qwen3-ASR)
  WS   /v1/audio/stream           — live PCM streaming (Nemotron)

Models load in background threads at startup so the HTTP surface is up
immediately; endpoints return 503 until their engine is ready.
"""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from . import config
from .batch_qwen import QwenBatchEngine
from .stream_nemotron import NemotronStreamEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("asr.main")

_batch_engine = QwenBatchEngine()
_stream_engine = NemotronStreamEngine()
_status = {"qwen3_asr": "disabled", "forced_aligner": "disabled", "nemotron_stream": "disabled"}
# Batch jobs are long; keep them off the event loop and at most 2 deep.
_BATCH_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="qwen-batch")


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.ASR_VARIANT in ("all", "batch"):
        threading.Thread(target=_load, args=(_batch_engine, ["qwen3_asr", "forced_aligner"]),
                         daemon=True, name="load-qwen").start()
    if config.ASR_VARIANT in ("all", "stream"):
        threading.Thread(target=_load, args=(_stream_engine, ["nemotron_stream"]),
                         daemon=True, name="load-nemotron").start()
    yield


app = FastAPI(title="asr-sidecar", version=config.VERSION, lifespan=lifespan)


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


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(""),
    response_format: str = Form("verbose_json"),
    language: str = Form(""),
):
    if _status["qwen3_asr"] == "loading":
        raise HTTPException(status_code=503, detail="model_loading")
    if not _batch_engine.loaded:
        raise HTTPException(status_code=503, detail=f"batch engine unavailable: {_status['qwen3_asr']}")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp = Path(tempfile.mkstemp(suffix=suffix, prefix="asr_up_")[1])
    try:
        tmp.write_bytes(await file.read())
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _BATCH_EXECUTOR,
            lambda: _batch_engine.transcribe(
                tmp,
                language=language.strip() or None,
                want_words=response_format == "verbose_json",
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Batch transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)

    if response_format == "text":
        return PlainTextResponse(result["text"])
    if response_format == "json":
        return {"text": result["text"]}
    return result


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

Run inside the container with /models mounted:
    docker run --rm -v $DATA_DIR:/models -e HF_TOKEN $IMAGE python3 -m app.prefetch
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
  cat > "${dest}/app/vad_chunker.py" <<'__ASR_EOF__'
"""Long-audio chunk planning for the batch engine.

The ForcedAligner handles at most ~5 minutes per call, so long recordings
are split into windows that (a) stay under ASR_MAX_CHUNK_S and (b) cut at
silence rather than mid-word. `pack_regions` is pure and unit-tested on
any host; `plan_chunks` needs silero-vad + torch and only runs in the
container.
"""
from __future__ import annotations


def pack_regions(
    regions: list[tuple[float, float]],
    total_dur: float,
    max_chunk_s: float = 280.0,
    pad_s: float = 0.2,
) -> list[tuple[float, float]]:
    """Pack VAD speech regions into transcription windows ≤ max_chunk_s.

    *regions* are (start, end) speech spans in seconds, ascending. Consecutive
    regions are merged into one window until adding the next region would
    exceed max_chunk_s — i.e. windows always split inside a silence gap. A
    single region longer than max_chunk_s is sliced at fixed size with 1 s
    overlap (no silence available to cut at). Windows are padded by pad_s on
    each side and clamped to [0, total_dur].
    """
    total_dur = max(0.0, float(total_dur))
    if total_dur == 0.0:
        return []
    if not regions:
        return _fixed_windows(0.0, total_dur, max_chunk_s)

    windows: list[tuple[float, float]] = []
    cur_start: float | None = None
    cur_end = 0.0

    def close_window() -> None:
        if cur_start is None:
            return
        start = max(0.0, cur_start - pad_s)
        end = min(total_dur, cur_end + pad_s)
        if end - start > max_chunk_s:
            windows.extend(_fixed_windows(start, end, max_chunk_s))
        elif end > start:
            windows.append((start, end))

    for r_start, r_end in regions:
        r_start, r_end = float(r_start), float(r_end)
        if r_end <= r_start:
            continue
        if cur_start is None:
            cur_start, cur_end = r_start, r_end
        elif r_end - cur_start <= max_chunk_s:
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
                max_chunk_s: float = 280.0) -> list[tuple[float, float]]:
    """VAD-based chunk plan for *wav_path* (16 kHz mono WAV).

    Falls back to fixed windows when silero-vad is unavailable or errors —
    a degraded cut point is better than a failed transcription.
    """
    try:
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

        model = load_silero_vad()
        audio = read_audio(wav_path, sampling_rate=16000)
        stamps = get_speech_timestamps(audio, model, sampling_rate=16000)
        regions = [(s["start"] / 16000.0, s["end"] / 16000.0) for s in stamps]
        return pack_regions(regions, total_dur, max_chunk_s=max_chunk_s)
    except Exception:
        return _fixed_windows(0.0, total_dur, max_chunk_s)
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
      conflict)? Build two single-engine containers instead:
      ASR_VARIANT=batch  ASR_CONTAINER_NAME=asr-gb10-batch  ./install_asr_gb10.sh
      ASR_VARIANT=stream ASR_CONTAINER_NAME=asr-gb10-stream ASR_PORT=8791 ./install_asr_gb10.sh"
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
