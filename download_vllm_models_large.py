#!/usr/bin/env python3
"""Downloads vLLM models (full Hugging Face repos, e.g. NVFP4 safetensors) onto an Ubuntu box.

Author: Christopher Gray
Updated: 10/8/2026
Version: 0.0.11

Download:
    wget https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/download_vllm_models_large.py && chmod +x download_vllm_models_large.py


Companion to download_llama_cpp_models.sh, but for whole-repo vLLM models that are
hundreds of GB across many shard files. Python 3.8+, standard library only.

Robust: resumes partial files (HTTP Range), skips files already complete, retries
every file with exponential backoff (the retry budget resets whenever a retry makes
progress, so a flaky link still finishes), downloads several shards in parallel,
and SHA-256-verifies every LFS file against the hash HF publishes for it -- safe
to run by hand or from cron. Re-running only fetches what is missing or changed.

Target: /opt/ai/models/<repo-name>/   (override: MODELS_DIR env or 1st arg)
Source: Hugging Face (public repos need no token; set HF_TOKEN for gated ones)

Requested models (the Hugging Face repo page URL is all that is needed):
  https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4
  https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4
  https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4
  https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4
  https://huggingface.co/unsloth/gemma-4-26B-A4B-it-NVFP4

Target hardware: 2x NVIDIA DGX Spark (128 GB unified memory each, 256 GB total, linked by
ConnectX-7). vLLM runs multi-node with tensor parallel 2, and every node needs its own
copy of the weights at the same path. So either run this script on both Sparks, or run it
on one and let it rsync to the other:  SYNC_TO=spark2 ./download_vllm_models_large.py
Or download first and copy later:      ./download_vllm_models_large.py --sync-only spark2
The picker flags models that are too big for one Spark (needs both) or for the pair.

Usage:
    ./download_vllm_models_large.py [MODELS_DIR] [--add REPO_OR_URL ...] [--sync-to HOST]

Examples:
    CHECK=1 ./download_vllm_models_large.py                  # sizes + status only, downloads nothing
    ./download_vllm_models_large.py                          # menu: pick models, Enter to start
    ALL=1 nohup ./download_vllm_models_large.py > download.log 2>&1 &   # overnight, no menu
    ./download_vllm_models_large.py --add nvidia/Some-Model-NVFP4       # one extra repo

    # Copy the finished models to the 2nd Spark over the ConnectX-7 link: give the address
    # of the 200 Gb/s port (ip -br addr), not the LAN hostname. Needs ssh keys + rsync.
    ./download_vllm_models_large.py --sync-only <user>@192.168.100.11
    # Uses 4 parallel rsyncs + a fast cipher by default; SYNC_JOBS=8 for more. Ends with a size check.
    # Or download and sync each model as it finishes:
    SYNC_TO=<user>@192.168.100.11 ALL=1 ./download_vllm_models_large.py

On a terminal a checkbox menu lets you pick models (Up/Down or j/k move, Space
toggles, a = all/none, Enter = start, q = quit). Models not complete on disk, or
changed on HF, are pre-checked. With no terminal (cron, pipes) every model is
downloaded.

Update check: each file's size and (for LFS files) SHA-256 are compared with what
HF publishes. Hashes of local files are cached in $MODELS_DIR/model_hashes_vllm.json
so each file is hashed only once (re-hashed if its size or mtime changes).

Environment:
    MODELS_DIR   Where to save models      (default: /opt/ai/models, or 1st arg)
    FORCE=1      Re-download everything even if complete
    ALL=1        Skip the menu and download every model
    CHECK=1      Only report complete / outdated / partial / missing; download nothing
    QUICK=1      Update check by file size only (skip hashing existing files)
    WORKERS      Parallel file downloads (default: 4)
    RETRIES      Consecutive no-progress retries per file before giving up (default: 20)
    SYNC_TO      After each model finishes, rsync it to this host (ssh name of the
                 second Spark, e.g. "spark2" or "user@10.0.0.2"), same MODELS_DIR path.
                 Use the ConnectX-7 link address for 200 Gb/s transfers.
    SYNC_JOBS    Parallel rsyncs when copying to the 2nd Spark (default: 4)
    REMOTE_MODELS_DIR  Models dir on the 2nd Spark if different (default: same as here)
    RSYNC_RSH    ssh command for the sync (default: ssh with the fast aes128-gcm cipher)
    SPARK_GB     Unified memory per Spark (default: 128); used for the fit hints
    HF_TOKEN     Hugging Face access token (gated/restricted repos, higher rate limits).
                 If unset, read from $HF_TOKEN_FILE, ~/.cache/huggingface/token (what
                 `huggingface-cli login` writes), ~/.hf_token, or a .env beside the script
                 (HF_TOKEN=...). Never put the token in this
                 script -- it lives in a synced folder. Create a read token at
                 https://huggingface.co/settings/tokens and accept each gated model's terms
                 on its page while logged in.
"""
import argparse 
import concurrent.futures as cf
import hashlib
import http.client
import json
import os
import shlex
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

VERSION = "0.0.1"
HF = "https://huggingface.co"
REVISION = "main"

# Repos to offer. Add more here, or pass --add on the command line.
MODELS = [
    "nvidia/GLM-5.3-Flash-NVFP4",
    "nvidia/Qwen3.8-Flash-Next-NVFP4",
    "nvidia/DeepSeek-V4.1-Flash-NVFP4",
    "nvidia/Qwen3.8-27B-NVFP4",
    "unsloth/gemma-4-26B-A4B-it-NVFP4",
]

# Optional hint shown next to a model in the picker.
HINTS = {
    "nvidia/Qwen3.8-27B-NVFP4": " - dense 27B, fits one DGX Spark",
    "unsloth/gemma-4-26B-A4B-it-NVFP4": " - MoE 26B (4B active)",
}

SPARK_GB = float(os.environ.get("SPARK_GB", "128"))
SPARKS = 2
# Fraction of memory usable for weights; the rest is OS + KV cache + activations.
WEIGHT_FRACTION = 0.8


def fit_note(total_bytes):
    gb = total_bytes / 1e9
    if gb <= SPARK_GB * WEIGHT_FRACTION:
        return "fits 1 Spark"
    if gb <= SPARK_GB * SPARKS * WEIGHT_FRACTION:
        return "needs both Sparks (TP=2)"
    return "TOO BIG even for 2 Sparks"


def _sync_buckets(files, jobs):
    """Split [(relpath, size)] into `jobs` lists of similar total size (biggest first)."""
    buckets = [[0, []] for _ in range(jobs)]
    for rel, size in sorted(files, key=lambda x: -x[1]):
        b = min(buckets, key=lambda x: x[0])
        b[0] += size
        b[1].append(rel)
    return [b[1] for b in buckets if b[1]]


def sync_repo(repo, models_dir, host):
    """Copy a finished model dir to the second Spark with several parallel rsyncs (a single
    rsync is CPU-bound well below the ConnectX-7 link speed), then verify. Resumable.
    Returns error|None."""
    import tempfile
    name = local_dir_name(repo)
    src = os.path.join(models_dir, name)
    rdir = os.path.join(os.environ.get("REMOTE_MODELS_DIR", models_dir), name)
    rsh = os.environ.get("RSYNC_RSH") or "ssh -c aes128-gcm@openssh.com -o Compression=no"
    files = []
    for root, _, names in os.walk(src):
        for n in names:
            if not n.endswith(".part"):
                full = os.path.join(root, n)
                files.append((os.path.relpath(full, src), os.path.getsize(full)))
    if not files:
        return f"nothing to sync in {src}"
    total = sum(sz for _, sz in files)
    jobs = max(1, min(int(os.environ.get("SYNC_JOBS", "4")), len(files)))
    say(f"   [sync] {name}: {len(files)} files, {human(total)} -> {host}:{rdir}  ({jobs} parallel rsync)")

    # Make sure the remote dir exists; fall back to sudo (passwordless) if the parent isn't writable.
    q = shlex.quote(rdir)
    mk = f'mkdir -p {q} 2>/dev/null || {{ sudo -n mkdir -p {q} && sudo -n chown "$(id -u):$(id -g)" {q}; }}'
    ok = subprocess.run(shlex.split(rsh) + [host, mk], stderr=subprocess.DEVNULL).returncode == 0
    if not ok and sys.stdin.isatty():
        # sudo needs a password: let the user type it once (ssh -t gives sudo a terminal).
        say(f"   [sync] {rdir} needs sudo on {host}; enter its password if asked:")
        sudo = f'sudo mkdir -p {q} && sudo chown "$(id -u):$(id -g)" {q}'
        ok = subprocess.run(shlex.split(rsh) + ["-t", host, sudo]).returncode == 0
    if not ok:
        return (f"can't create {rdir} on {host}. Run once on {host}:  "
                f"sudo mkdir -p {os.path.dirname(rdir)} && sudo chown $USER {os.path.dirname(rdir)}")

    tmpdir = tempfile.mkdtemp(prefix="vllm_sync_")
    try:
        def listfile(rels, i):
            path = os.path.join(tmpdir, f"files{i}.txt")
            with open(path, "w") as f:
                f.write("\n".join(rels) + "\n")
            return path

        base = ["rsync", "-a", "--partial", "--inplace", "-e", rsh]
        t0 = time.time()
        procs = []
        for i, rels in enumerate(_sync_buckets(files, jobs)):
            extra = ["--info=progress2"] if jobs == 1 else []
            procs.append(subprocess.Popen(base + extra + [f"--files-from={listfile(rels, i)}", src + "/", f"{host}:{rdir}/"]))
        rcs = [p.wait() for p in procs]
        if any(rcs):
            return f"rsync failed (exit codes {rcs}); re-run to resume"
        secs = max(time.time() - t0, 1)
        say(f"   [sync] copied in {fmt_time(secs)} (~{human(total / secs)}/s); verifying sizes...")

        # Verify: a dry run must have nothing left to transfer.
        out = subprocess.run(["rsync", "-rn", "--size-only", "--out-format=%n", "-e", rsh,
                              f"--files-from={listfile([r for r, _ in files], 'all')}", src + "/", f"{host}:{rdir}/"],
                             capture_output=True, text=True)
        diff = [l for l in out.stdout.splitlines() if l.strip()]
        if out.returncode != 0 or diff:
            return f"verify failed ({len(diff)} file(s) differ, e.g. {diff[:2]}); re-run to resume"
        say(f"   [sync] verified: {len(files)} files match on {host}")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


CHUNK = 4 * 1024 * 1024
SOCKET_TIMEOUT = 60          # seconds without data before a read is retried
HASH_CHUNK = 16 * 1024 * 1024



def load_token():
    """HF access token: $HF_TOKEN, else the first token file found. Returns (token, source)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip(), "HF_TOKEN env"
    candidates = [os.environ.get("HF_TOKEN_FILE"),
                  os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "token"),
                  os.path.expanduser("~/.hf_token")]
    for path in filter(None, candidates):
        try:
            with open(path) as f:
                tok = f.read().strip()
            if tok:
                return tok, path
        except OSError:
            pass
    # A .env next to this script (or in the current dir), as the other projects use.
    for d in dict.fromkeys([os.path.dirname(os.path.abspath(__file__)), os.getcwd()]):
        try:
            with open(os.path.join(d, ".env")) as f:
                for line in f:
                    k, _, v = line.strip().partition("=")
                    if k in ("HF_TOKEN", "HUGGINGFACE_ACCESS_TOKEN", "HF_ACCESS_TOKEN") and v.strip(" \"'"):
                        return v.strip(" \"'"), os.path.join(d, ".env")
        except OSError:
            pass
    return None, None


TOKEN, TOKEN_SOURCE = load_token()
TTY = sys.stdout.isatty()

print_lock = threading.Lock()


def say(msg=""):
    with print_lock:
        if TTY:
            sys.stdout.write("\r\033[K")
        print(msg, flush=True)


def human(n):
    n = float(n)
    for unit in "BKMGTP":
        if n < 1024 or unit == "P":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def fmt_time(s):
    s = int(s)
    if s >= 3600:
        return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def normalize_repo(s):
    """Accept 'owner/name' or any huggingface.co URL (tree/blob/resolve) for it."""
    s = s.strip()
    if s.startswith("http"):
        parts = urllib.parse.urlparse(s).path.strip("/").split("/")
        s = "/".join(parts[:2])
    return s


def local_dir_name(repo):
    return repo.split("/", 1)[1]


# -------------------------------------------------------------
# HTTP
# -------------------------------------------------------------

class _StripAuthOnCrossHost(urllib.request.HTTPRedirectHandler):
    """HF redirects LFS files to a CDN whose presigned URLs reject an Authorization header."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers = {k: v for k, v in new.headers.items() if k.lower() != "authorization"}
            new.unredirected_hdrs = {k: v for k, v in new.unredirected_hdrs.items() if k.lower() != "authorization"}
        return new


OPENER = urllib.request.build_opener(_StripAuthOnCrossHost)
NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException, ConnectionError, socket.timeout, TimeoutError, OSError)


class Fatal(Exception):
    """Not worth retrying (auth, missing file, disk full)."""


def request(url, headers=None, method="GET"):
    h = {"User-Agent": f"download_vllm_models_large/{VERSION}"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    h.update(headers or {})
    return OPENER.open(urllib.request.Request(url, headers=h, method=method), timeout=SOCKET_TIMEOUT)


def with_retries(fn, what, tries=6):
    """Run a small API call, retrying transient failures."""
    for i in range(tries):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise Fatal(f"HTTP {e.code} for {what}" + (" (gated? set HF_TOKEN)" if e.code in (401, 403) else ""))
            err = e
        except NETWORK_ERRORS as e:
            if isinstance(getattr(e, "reason", e), ssl.SSLCertVerificationError):
                raise Fatal(f"{what}: TLS certificate error ({e}); install ca-certificates / set SSL_CERT_FILE")
            err = e
        if i == tries - 1:
            raise Fatal(f"{what}: {err}")
        time.sleep(min(30, 2 ** (i + 1)))


def list_repo_files(repo):
    """[{path, size, sha256|None}] for every file in the repo (paginated tree API)."""
    url = f"{HF}/api/models/{repo}/tree/{REVISION}?recursive=1&expand=1"
    files = []
    while url:
        def fetch(u=url):
            with request(u) as r:
                return json.load(r), r.headers.get("Link", "")
        entries, link = with_retries(fetch, f"file list of {repo}")
        for e in entries:
            if e.get("type") != "file":
                continue
            lfs = e.get("lfs") or {}
            files.append({"path": e["path"], "size": int(lfs.get("size") or e.get("size") or 0),
                          "sha256": lfs.get("oid")})
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.index("<") + 1:part.index(">")]
    return files


# -------------------------------------------------------------
# Hash cache / status
# -------------------------------------------------------------

class HashDB:
    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        try:
            with open(path) as f:
                self.data = json.load(f)
        except (OSError, ValueError):
            self.data = {}

    def get(self, key, size, mtime):
        e = self.data.get(key) or {}
        return e.get("sha256") if e.get("size") == size and e.get("mtime") == mtime else None

    def set(self, key, sha, size, mtime):
        with self.lock:
            self.data[key] = {"sha256": sha, "size": size, "mtime": mtime,
                              "hashed": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, self.path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def local_sha256(db, key, path):
    st = os.stat(path)
    mtime = int(st.st_mtime)
    sha = db.get(key, st.st_size, mtime)
    if not sha:
        say(f"   [hash] hashing {key} ({human(st.st_size)}) - only needed once...")
        sha = sha256_file(path)
        db.set(key, sha, st.st_size, mtime)
    return sha


def file_state(db, ddir, relkey, f, quick):
    """complete | outdated | partial | missing  for one remote file."""
    dest = os.path.join(ddir, f["path"])
    if os.path.isfile(dest):
        if os.path.getsize(dest) != f["size"]:
            return "outdated"
        if f["sha256"] and not quick and local_sha256(db, relkey, dest) != f["sha256"]:
            return "outdated"
        return "complete"
    return "partial" if os.path.exists(dest + ".part") else "missing"


def repo_status(db, models_dir, repo, files, quick, force):
    """Overall: complete | outdated | partial | missing, plus the files still needed."""
    ddir = os.path.join(models_dir, local_dir_name(repo))
    todo, states = [], set()
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda f: file_state(db, ddir, f"{local_dir_name(repo)}/{f['path']}", f, quick), files))
    for f, s in zip(files, results):
        states.add(s)
        if force or s != "complete":
            todo.append(f)
    if not todo:
        return "complete", todo
    if "outdated" in states:
        return "outdated", todo
    if states == {"missing"}:
        return "missing", todo
    return "partial", todo


# -------------------------------------------------------------
# Download engine
# -------------------------------------------------------------

class Progress:
    """Thread-safe per-file byte counters + a one-line aggregate bar."""

    def __init__(self, total):
        self.total, self.bytes, self.lock = total, {}, threading.Lock()
        self.active, self.stop = set(), threading.Event()

    def set(self, key, n):
        with self.lock:
            self.bytes[key] = n

    def add(self, key, n):
        with self.lock:
            self.bytes[key] = self.bytes.get(key, 0) + n

    def done_bytes(self):
        with self.lock:
            return sum(self.bytes.values())

    def monitor(self):
        width, rate, prev, tick = 30, 0.0, self.done_bytes(), 1.0
        while not self.stop.wait(tick):
            cur = self.done_bytes()
            rate = (rate * 2 + (cur - prev) / tick) / 3
            prev = cur
            pct = min(100, int(cur * 100 / self.total)) if self.total else 0
            filled = pct * width // 100
            eta = fmt_time((self.total - cur) / rate) if rate > 1 else "--:--"
            line = (f"   {'█' * filled}{'░' * (width - filled)} {pct:3d}%  {human(cur)} / {human(self.total)}"
                    f"  {human(rate)}/s  ETA {eta}  [{len(self.active)} active]")
            with print_lock:
                sys.stdout.write("\r\033[K" + line)
                sys.stdout.flush()
        with print_lock:
            sys.stdout.write("\r\033[K")


def stream_once(repo, f, part, prog, key):
    """One connection: resume `part` and append until the server closes. Returns when it ends."""
    url = f"{HF}/{repo}/resolve/{REVISION}/{urllib.parse.quote(f['path'])}"
    cur = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={cur}-"} if cur else {}
    try:
        resp = request(url, headers)
    except urllib.error.HTTPError as e:
        if e.code == 416:                      # asked past the end: .part is complete or oversized
            return
        raise
    with resp:
        if cur and resp.status != 206:         # server ignored Range: start over
            cur = 0
        with open(part, "r+b" if cur else "wb") as out:
            out.seek(cur)
            prog.set(key, cur)
            while chunk := resp.read(CHUNK):
                out.write(chunk)
                prog.add(key, len(chunk))


def download_file(repo, f, models_dir, prog, db, retries):
    """Download one file to <dir>/<path> with resume + retries. Returns (path, error|None)."""
    rel = f["path"]
    key = f"{local_dir_name(repo)}/{rel}"
    dest = os.path.join(models_dir, local_dir_name(repo), rel)
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    size = f["size"]
    prog.active.add(rel)
    try:
        attempts = 0
        while True:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if have > size > 0:
                os.remove(part)
                have = 0
            prog.set(key, have)
            if size and have == size:
                break
            try:
                stream_once(repo, f, part, prog, key)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 404):
                    return rel, f"HTTP {e.code}" + (" (gated? set HF_TOKEN)" if e.code in (401, 403) else "")
                err = f"HTTP {e.code}"
            except OSError as e:
                if getattr(e, "errno", None) in (28, 122):   # ENOSPC / EDQUOT
                    return rel, "disk full"
                err = str(e)
            except NETWORK_ERRORS as e:
                err = str(e)
            else:
                err = None
            now = os.path.getsize(part) if os.path.exists(part) else 0
            if size and now == size:
                break
            if not size and err is None and now > 0:
                break                          # size unknown: a clean finish is done
            attempts = 0 if now > have else attempts + 1
            if attempts >= retries:
                return rel, f"stalled at {human(now)} of {human(size)} ({err or 'short read'})"
            wait = min(60, 2 ** min(attempts, 6))
            say(f"   [warn] {rel}: {err or 'connection closed early'} at {human(now)}; retry {attempts + 1}/{retries} in {wait}s")
            time.sleep(wait)

        if f["sha256"]:
            prog.active.discard(rel)
            sha = sha256_file(part)
            if sha != f["sha256"]:
                os.remove(part)                # a bad .part would just be resumed again
                return rel, "SHA-256 mismatch; partial deleted, re-run to download fresh"
        else:
            sha = None
        os.replace(part, dest)
        if sha:
            st = os.stat(dest)
            db.set(key, sha, st.st_size, int(st.st_mtime))
        return rel, None
    except Fatal as e:
        return rel, str(e)
    finally:
        prog.active.discard(rel)


def download_repo(repo, todo, models_dir, db, workers, retries):
    """Fetch the files in `todo`. Returns list of (path, error) failures."""
    ddir = os.path.join(models_dir, local_dir_name(repo))
    os.makedirs(ddir, exist_ok=True)
    # Biggest first keeps the workers busy to the end.
    todo = sorted(todo, key=lambda f: -f["size"])
    prog = Progress(sum(f["size"] for f in todo))
    mon = threading.Thread(target=prog.monitor, daemon=True)
    if TTY:
        mon.start()
    failures, finished = [], 0
    try:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(download_file, repo, f, models_dir, prog, db, retries) for f in todo]
            for fut in cf.as_completed(futs):
                rel, err = fut.result()
                finished += 1
                if err:
                    failures.append((rel, err))
                    say(f"   [fail] {rel} - {err}")
                else:
                    say(f"   [done] ({finished}/{len(todo)}) {rel}")
    finally:
        prog.stop.set()
        if TTY:
            mon.join()
    return failures


# -------------------------------------------------------------
# Model picker (checkbox menu, interactive terminals only)
# -------------------------------------------------------------

def read_key():
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select_models(repos, status, sizes, force, totals):
    n, cur, drawn = len(repos), 0, False
    checked = [force or status[r] != "complete" for r in repos]
    sys.stdout.write("\033[?25l")
    try:
        while True:
            if drawn:
                sys.stdout.write(f"\033[{n + 2}A")
            drawn = True
            sys.stdout.write("\r\033[K Select models to download  (Space toggle, a all/none, Enter start, q quit)\n")
            for i, r in enumerate(repos):
                note = HINTS.get(r, "")
                st = status[r]
                note += {"complete": "   (on disk, up to date)", "outdated": "   (on disk, UPDATE AVAILABLE)",
                         "partial": "   (partially downloaded - will resume)", "missing": ""}.get(st, "   (not checked)")
                note += f"   [{human(totals[r])}, {fit_note(totals[r])}]"
                sys.stdout.write(f"\r\033[K  {'>' if i == cur else ' '} [{'x' if checked[i] else ' '}] {r}{note}\n")
            sys.stdout.write("\r\033[K\n")
            sys.stdout.flush()
            key = read_key()
            if key in ("\x1b[A", "k"):
                cur = (cur - 1) % n
            elif key in ("\x1b[B", "j"):
                cur = (cur + 1) % n
            elif key == " ":
                checked[cur] = not checked[cur]
            elif key in ("a", "A"):
                checked = [not all(checked)] * n
            elif key in ("\r", "\n"):
                break
            elif key in ("q", "Q", "\x03", "\x04"):
                sys.stdout.write("\033[?25h Cancelled.\n")
                sys.exit(0)
    finally:
        sys.stdout.write("\033[?25h")
    return [r for r, c in zip(repos, checked) if c]


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Download large vLLM models from Hugging Face (resume + retry).")
    ap.add_argument("models_dir", nargs="?", default=os.environ.get("MODELS_DIR", "/opt/ai/models"))
    ap.add_argument("--sync-to", default=os.environ.get("SYNC_TO"), metavar="HOST",
                    help="rsync each finished model to this host (the second DGX Spark)")
    ap.add_argument("--sync-only", metavar="HOST",
                    help="don't download; just rsync the models already on disk to HOST (the second Spark)")
    ap.add_argument("--add", action="append", default=[], metavar="REPO_OR_URL",
                    help="extra repo (owner/name or huggingface.co URL); repeatable")
    a = ap.parse_args()

    models_dir = a.models_dir
    force = os.environ.get("FORCE") == "1"
    quick = os.environ.get("QUICK") == "1"
    workers = max(1, int(os.environ.get("WORKERS", "4")))
    retries = max(1, int(os.environ.get("RETRIES", "20")))
    repos = list(dict.fromkeys(normalize_repo(r) for r in MODELS + a.add))

    print(f"\n vLLM large-model downloader  v{VERSION}\n")
    if a.sync_only:
        bad = 0
        for r in repos:
            if not os.path.isdir(os.path.join(models_dir, local_dir_name(r))):
                print(f"   [skip] not on disk: {r}")
                continue
            err = sync_repo(r, models_dir, a.sync_only)
            bad += bool(err)
            print(f"   [fail] {r}: {err}" if err else f"   [ok  ] {r}")
        return 1 if bad else 0
    os.makedirs(models_dir, exist_ok=True)
    if TOKEN:
        try:
            with request(f"{HF}/api/whoami-v2") as r:
                print(f" Hugging Face login: {json.load(r).get('name', '?')}  (token from {TOKEN_SOURCE})\n")
        except urllib.error.HTTPError as e:
            print(f" [warn] token from {TOKEN_SOURCE} was rejected (HTTP {e.code}); continuing anonymously-limited\n")
        except NETWORK_ERRORS:
            print(f" Using Hugging Face token from {TOKEN_SOURCE}\n")
    else:
        print(" No Hugging Face token (public repos only). For restricted models: export HF_TOKEN=hf_... \n")
    db = HashDB(os.path.join(models_dir, "model_hashes_vllm.json"))

    # Remote file lists + local status.
    print(f" Checking {len(repos)} models against Hugging Face...")
    remote, status, todo, errors = {}, {}, {}, {}
    for r in repos:
        try:
            remote[r] = list_repo_files(r)
        except Fatal as e:
            errors[r] = str(e)
            print(f"   [warn] {r}: {e}")
            continue
        if not remote[r]:
            errors[r] = "repo has no files"
            print(f"   [warn] {r}: repo has no files")
            continue
        status[r], todo[r] = repo_status(db, models_dir, r, remote[r], quick, force)
        if status[r] == "outdated":
            print(f"   [new ] update available on HF: {r}")
    print()
    repos = [r for r in repos if r in remote and r not in errors]
    sizes = {r: sum(f["size"] for f in todo[r]) for r in repos}
    totals = {r: sum(f["size"] for f in remote[r]) for r in repos}

    if os.environ.get("CHECK") == "1":
        print(f" Model status ({models_dir}):")
        for r in repos:
            print(f"   {status[r]:<9} {r}   (to fetch: {human(sizes[r])})")
        for r, e in errors.items():
            print(f"   {'error':<9} {r}   {e}")
        return 1 if errors else 0

    if sys.stdin.isatty() and TTY and os.environ.get("ALL") != "1" and repos:
        repos = select_models(repos, status, sizes, force, totals)
    else:
        repos = [r for r in repos if force or status[r] != "complete"] if not force else repos
    if not repos:
        print(" Nothing to download." if not errors else " Nothing to download (see warnings above).")
        return 1 if errors else 0

    need = sum(sizes[r] for r in repos)
    free = shutil.disk_usage(models_dir).free
    print("==============================================")
    print(" vLLM models - large downloader")
    print("==============================================")
    print(f"Target directory  : {models_dir}")
    print(f"Models            : {len(repos)}")
    print(f"Parallel files    : {workers}     Retries/file: {retries}")
    print(f"To download       : {human(need)}   (free: {human(free)})")
    print(f"Second Spark sync : {a.sync_to or 'off (run this script on the other Spark too, or set SYNC_TO)'}")
    for r in repos:
        print(f"  {r}: {human(totals[r])} - {fit_note(totals[r])}")
    print()
    if need > free:
        print(f"ERROR: not enough free disk space ({human(need)} needed, {human(free)} free).")
        return 1

    failed = {}
    for r in repos:
        print(f"[{r}]  {len(todo[r])} file(s), {human(sizes[r])}")
        fails = download_repo(r, todo[r], models_dir, db, workers, retries) if todo[r] else []
        if not todo[r]:
            print("   [skip] up to date")
        if not fails and a.sync_to:
            err = sync_repo(r, models_dir, a.sync_to)
            if err:
                fails = [("(sync)", err)]
        if fails:
            failed[r] = fails
        elif todo[r]:
            print(f"   [ok  ] {r} -> {os.path.join(models_dir, local_dir_name(r))}")
        print()

    print("==============================================")
    print(" Download summary")
    print("==============================================")
    print(f"  target : {models_dir}")
    for r in repos:
        mark = "FAILED" if r in failed else "ok"
        print(f"  {mark:<7} {r}")
        for rel, err in failed.get(r, []):
            print(f"            - {rel}: {err}")
    print()
    if failed or errors:
        print(f" Completed with problems. Re-run to resume:\n    {sys.argv[0]} {models_dir}")
        return 1
    print(" All models downloaded.\n Serve one with:  vllm serve <MODELS_DIR>/<model-name>")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n Interrupted -- partial files are kept; re-run to resume.")
        sys.exit(130)
