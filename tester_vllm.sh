#!/usr/bin/env python3
# Christopher Gray - @c2theg/ai  |  Version: 1.0.0  |  Update: 8/9/2026
# vLLM smoke test — auto-discovers every running vLLM instance (ports + models)
#                   and runs the full smoke test against each one.
# Includes 21 auto-graded model-quality tests (reasoning, math, summarization,
# counting, PDF extraction, table lookup, multi-turn, negation, unit conversion,
# date math, JSON extraction, code bug-fixing, instruction-following, code,
# factual, long-context, translation, sentiment, vision/OCR, audio/ASR) plus a
# 7-language timed code-generation suite with model self-graded correctness,
# and a per-instance capability scorecard.
#
#
# Update Yourself:
#  curl -fsSL -H 'Cache-Control: no-cache' -H 'Pragma: no-cache' -o 'tester_vllm.py' "https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/tester_vllm.py?nocache=$(date +%s)" && chmod u+x tester_vllm.py
#
#
# Usage: ./tester_llm.py [HOST] [PORT]
#   No args     -> auto-discover ALL local vLLM instances and test each.
#   HOST        -> discover instances on that host (local discovery only).
#   HOST PORT   -> test only that specific host:port (skips discovery).
#
# Requirements:
#   - python3 (>=3.8), pip install rich   (required)
#   - pdftotext (poppler-utils)           (optional — enables the PDF
#                                           extraction test; skips gracefully
#                                           if it's absent)
#   - nvidia-smi / rocm-smi               (optional — enables GPU stats)
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
import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
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
SCRIPT_VERSION = "1.0.0"
SCRIPT_UPDATED = "8/9/2026"

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


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────
def http_get(url: str, timeout: int = 10) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def http_get_status(url: str, timeout: int = 10) -> tuple[Optional[int], str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return None, ""


def http_post_json(url: str, obj: dict, timeout: int = 300) -> Optional[str]:
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
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
    console.print()
    console.print(Rule(f"[bold {color}]{title}[/bold {color}]", style=color, align="left"))


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


def discover_instances(host: str) -> list[str]:
    ps_out = run_cmd(["ps", "aux"], timeout=5)
    proc_lines = [
        ln
        for ln in ps_out.splitlines()
        if re.search(r"vllm serve|vllm[._]entrypoints[._]openai|[Vv]llm.*api_server", ln)
        and "grep" not in ln
    ]

    if not proc_lines:
        cwarn("No running vLLM processes found on this host")
        return []

    cinfo(f"Found {len(proc_lines)} vLLM-related process(es); resolving ports...")
    target_ports: list[str] = []
    for line in proc_lines:
        cols = line.split()
        if len(cols) < 2:
            continue
        pid = cols[1]
        host_bind_m = re.search(r"--host[= ](\S+)", line)
        model_m = re.search(r"--served-model-name[= ](\S+)", line) or re.search(r"vllm serve (\S+)", line)
        port_flag_m = re.search(r"--port[= ](\d+)", line)

        ports_found: list[str] = []
        if port_flag_m:
            ports_found.append(port_flag_m.group(1))
        else:
            ports_found = listen_ports_for_pid(pid) or ["8000"]

        host_bind = host_bind_m.group(1) if host_bind_m else "0.0.0.0"
        model_name = model_m.group(1) if model_m else "(from /v1/models)"

        for p in ports_found:
            console.print(f"  PID {pid}  |  port={p}  host={host_bind}  model={model_name}")
            target_ports.append(p)

    # Dedup while preserving numeric order.
    unique_ports = sorted(set(target_ports), key=lambda p: int(p))
    return unique_ports


# ─────────────────────────────────────────────────────────────────────────────
# Per-instance test suite
# ─────────────────────────────────────────────────────────────────────────────
def run_instance_tests(host: str, port: str, color: str) -> InstanceResult:
    base_url = f"http://{host}:{port}"
    result = InstanceResult(base_url=base_url)
    start_time = time.perf_counter()
    quality = QualityTracker()

    console.print()
    console.print(Panel(f"Instance target: [bold]{base_url}[/bold]", style=color, box=box.DOUBLE))

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
        task = progress.add_task(f"[{color}]Testing {base_url}", total=TOTAL_TEST_STEPS)
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
            progress.update(task, completed=TOTAL_TEST_STEPS)
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
                    console.print(f"  • {m.get('id')}  (owned_by: {m.get('owned_by', 'n/a')})")
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
                headers={"Content-Type": "application/json"},
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
        else:
            reason = "no model discovered" if not first_model else "model does not support chat/completions (embedding-only model)"
            cwarn(f"Skipping capability tests 10-37 — {reason}")
            progress.update(task, completed=TOTAL_TEST_STEPS)
            progress.refresh()

        # ── Capability scorecard ────────────────────────────────────────────
        section(f"Capability Scorecard — {base_url}", color)
        if first_model:
            cinfo(f"Model : {first_model}")
            if quality.total:
                pct = quality.passed * 100 // quality.total
                cinfo(f"Quality score : {quality.passed}/{quality.total} graded checks passed ({pct}%)")
            else:
                cwarn("No graded checks were run")
        else:
            cwarn("No model discovered — nothing to score")

    result.quality_pass = quality.passed
    result.quality_total = quality.total
    result.duration_s = time.perf_counter() - start_time
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Roll-up report
# ─────────────────────────────────────────────────────────────────────────────
def print_rollup_report(results: list[InstanceResult]) -> None:
    section("Roll-Up Report")
    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Model", style="bold")
    table.add_column("Instance")
    table.add_column("Duration", justify="right")
    table.add_column("Capabilities")
    table.add_column("Failures", justify="right")
    table.add_column("Quality Score", justify="right")

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
        table.add_row(
            f"[{color}]{res.model}[/{color}]",
            res.base_url,
            f"{res.duration_s:.1f}s",
            caps_s,
            f"[{fail_style}]{res.failures}[/{fail_style}]",
            quality_s,
        )
    console.print(table)
    cinfo(f"Total wall-clock time across all instances: {total_duration:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM smoke test with a Rich UI")
    parser.add_argument("host", nargs="?", default="localhost")
    parser.add_argument("port", nargs="?", default=None)
    args = parser.parse_args()

    console.print()
    console.print(Rule(f"[bold cyan]tester_vllm.py[/bold cyan]", style="cyan", align="left"))
    cinfo(f"Author  : {SCRIPT_AUTHOR}")
    cinfo(f"Version : {SCRIPT_VERSION}")
    cinfo(f"Updated : {SCRIPT_UPDATED}")

    print_system_hardware()

    section("0. vLLM Instance Discovery")
    if args.port:
        cinfo(f"Explicit port supplied — testing only {args.host}:{args.port}")
        target_ports = [args.port]
    else:
        target_ports = discover_instances(args.host)

    if not target_ports:
        cfail("No vLLM ports discovered — nothing to test.")
        console.print(f"  Hint: start vLLM, or pass an explicit target: {sys.argv[0]} {args.host} <port>")
        return 1

    cinfo(f"Will test {len(target_ports)} instance(s) on {args.host}: ports {', '.join(target_ports)}")

    results: list[InstanceResult] = []
    failures_total = 0
    for i, port in enumerate(target_ports):
        color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        res = run_instance_tests(args.host, port, color)
        results.append(res)
        failures_total += res.failures

    print_rollup_report(results)

    section("Summary")
    cinfo(f"Tested {len(target_ports)} instance(s) on {args.host}: ports {', '.join(target_ports)}")
    if failures_total == 0:
        cpass("All critical checks passed across all instances")
    else:
        cfail(f"{failures_total} critical check(s) failed across all instances")

    return failures_total


if __name__ == "__main__":
    sys.exit(main())
