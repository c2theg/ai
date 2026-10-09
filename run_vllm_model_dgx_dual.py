#!/usr/bin/env python3
"""Launch ONE vLLM model across BOTH DGX Sparks (tensor parallel 2) over ConnectX-7 / RoCE.

Author: Chris Gray
Updated: 10/9/2026
Version: 0.0.9
 
Install:
     wget https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/run_vllm_model_dgx_dual.py && chmod +x run_vllm_model_dgx_dual.py


Re-implements the multi-node path of https://github.com/eugr/spark-vllm-docker
(launch-cluster.sh + recipes/) in a single stdlib-only script: it starts that
project's Docker image on both Sparks (privileged, host network, /dev/infiniband),
points NCCL / UCX / Gloo at the ConnectX-7 port, then runs `vllm serve` on the head
(rank 0) and as a headless worker (rank 1) over SSH -- eugr's default "no-Ray" mode.
The OpenAI-compatible API comes up on the HEAD node (the Spark you run this on).

Models are read from MODELS_DIR (default /opt/ai/models) on each Spark -- the layout
download_vllm_models_large.py produces -- and mounted read-only at /models.

Every model folder under MODELS_DIR (anything with a config.json) is offered: the ones in
PROFILES use eugr's tuned recipes, any other gets an automatic generic profile. If a model is
on one Spark but missing/incomplete on the other, `start` copies it over the ConnectX link
first (parallel rsync, with progress) and says what it is doing. Before launching it also stops
every other vLLM on the nodes it uses (containers, `vllm serve`, Ray); --keep-others disables that.

One-time prerequisites (see eugr's docs/NETWORKING.md):
  * The two Sparks are cabled by ConnectX-7 with an IP each on that link (MTU 9000 is best).
  * Passwordless SSH from this node to the other Spark; your user can run `docker` on both.
  * The Docker image on both nodes:  ./run_vllm_model_dgx_dual.py setup
  * The model folder on at least one node (download_vllm_models_large.py); the other gets a copy.

Usage:
    ./run_vllm_model_dgx_dual.py                       # menu: pick a model, then launch
    ./run_vllm_model_dgx_dual.py start qwen3.8-27b     # launch a profile (see `list`)
    ./run_vllm_model_dgx_dual.py start glm-5.3-flash --max-model-len 131072 -- --enable-expert-parallel
    ./run_vllm_model_dgx_dual.py check                 # preflight only: network, RoCE, image, models
    ./run_vllm_model_dgx_dual.py status | logs [-f] [--worker] | stop | list | setup
    ./run_vllm_model_dgx_dual.py start <profile> --dry-run   # print every command, run nothing

Anything after `--` is appended to `vllm serve` last, so it overrides the profile
(e.g. `-- --load-format auto --quantization modelopt`).

Config (environment or a .env beside this script; same names as install_ai_spark_vllm.sh):
    SPARK_WORKER_SSH   ssh target of the other Spark, e.g. cgray@10.13.1.21   (required)
    SPARK_WORKER_IP    its IP on the ConnectX link   (default: read from its ConnectX port)
    SPARK_HEAD_IP      this node's ConnectX IP       (default: read from the port)
    SPARK_IFACE        ConnectX netdev on both nodes, e.g. enp1s0f0np0 (default: ibdev2netdev)
    SPARK_IB_HCA       RoCE devices for NCCL, e.g. rocep1s0f0,roceP2p1s0f0 (default: all Up)
    EXTRA_KILL_CONTAINERS  extra container names to stop before a launch (default: qwen38-flash);
                       any container whose name/image/command mentions vllm is always stopped
    SYNC_JOBS          parallel rsyncs when copying a model to the other Spark (4)
    MODELS_DIR         /opt/ai/models        PORT  8000        MASTER_PORT  29501
    CONTAINER_NAME     vllm_node             HEALTH_TIMEOUT  seconds to wait for the API (3600)
    SETUP_DIR          ~/spark-vllm-docker   where `setup` clones eugr's repo
"""
import argparse 
import getpass
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

VERSION = "0.0.1"
EUGR_REPO = "https://github.com/eugr/spark-vllm-docker.git"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
CACHE_DIRS = [".cache/vllm", ".cache/flashinfer", ".cache/b12x", ".triton", ".tilelang", ".cache/huggingface"]
DRY = False


def load_env_file():
    """Fill os.environ from a .env beside this script (real environment wins)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path) as f:
            for line in f:
                k, _, v = line.strip().partition("=")
                if k.startswith(("SPARK_", "MODELS_DIR", "PORT", "MASTER_PORT", "CONTAINER_NAME",
                                 "HEALTH_TIMEOUT", "SETUP_DIR")) and k not in os.environ:
                    os.environ[k] = v.strip(" \"'")
    except OSError:
        pass


def cfg(key, default=""):
    return os.environ.get(key) or default


# -------------------------------------------------------------
# Model profiles (from eugr's recipes/*.yaml, pointed at the local model folders)
# -------------------------------------------------------------

def spec(d):
    return json.dumps(d, separators=(",", ":"))


def args_qwen38_27b(tp, models_dir):
    a = ["--trust-remote-code", "--kv-cache-dtype", "fp8", "--gpu-memory-utilization", "0.7",
         "--max-model-len", "262144", "--max-num-seqs", "8", "--max-num-batched-tokens", "16384",
         "--enable-chunked-prefill", "--async-scheduling", "--enable-prefix-caching",
         "--load-format", "instanttensor", "--reasoning-parser", "qwen3",
         "--tool-call-parser", "qwen3_xml", "--enable-auto-tool-choice"]
    # DFlash2 draft model: used only if it was downloaded next to the main model.
    if os.path.isdir(os.path.join(models_dir, "Qwen3.8-27B-DFlash2")):
        a += ["--speculative-config", spec({"method": "dflash", "model": "/models/Qwen3.8-27B-DFlash2",
                                            "num_speculative_tokens": 8, "draft_tensor_parallel_size": tp})]
    return a


def args_qwen38_flash(tp, models_dir):
    return ["--trust-remote-code", "--pipeline-parallel-size", "1", "--mamba-cache-mode", "align",
            "--enable-prefix-caching", "--enable-chunked-prefill", "--dtype", "bfloat16",
            "--kv-cache-dtype", "fp8", "--quantization", "modelopt_mixed", "--block-size", "16",
            "--load-format", "b12x", "--max-model-len", "262144", "--max-num-seqs", "16",
            "--max-num-batched-tokens", "4096",
            "--speculative-config", spec({"method": "mtp", "num_speculative_tokens": 4}),
            "--gdn-decode-kernel", "b12x", "--linear-backend", "b12x", "--moe-backend", "b12x",
            "--no-enable-flashinfer-autotune", "--mm-encoder-tp-mode", "data",
            "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_xml", "--enable-auto-tool-choice",
            "--compilation-config", spec({"pass_config": {"fuse_act_quant": True}}),
            "--gpu-memory-utilization", "0.7"]


def args_glm53_flash(tp, models_dir):
    return ["--trust-remote-code", "--pipeline-parallel-size", "1", "--decode-context-parallel-size", "1",
            "--mamba-cache-mode", "align", "--enable-prefix-caching", "--enable-chunked-prefill",
            "--dtype", "bfloat16", "--kv-cache-dtype", "fp8", "--quantization", "modelopt_mixed",
            "--attention-backend", "B12X", "--block-size", "256", "--moe-backend", "b12x",
            "--linear-backend", "b12x", "--no-enable-flashinfer-autotune", "--load-format", "b12x",
            "--max-model-len", "500000", "--max-num-seqs", "4", "--max-num-batched-tokens", "4096",
            "--speculative-config", spec({"method": "mtp", "num_speculative_tokens": 5,
                                          "moe_backend": "humming", "attention_backend": "B12X"}),
            "--reasoning-parser", "glm45", "--tool-call-parser", "glm47", "--enable-auto-tool-choice",
            "--gpu-memory-utilization", "0.87"]


def args_deepseek(tp, models_dir):
    return ["--trust-remote-code", "--kv-cache-dtype", "fp8", "--block-size", "256",
            "--max-model-len", "500000", "--max-num-seqs", "4", "--max-num-batched-tokens", "8192",
            "--gpu-memory-utilization", "0.8", "--enable-prefix-caching",
            "--speculative-config", spec({"method": "mtp", "num_speculative_tokens": 2}),
            "--tokenizer-mode", "deepseek_v4", "--tool-call-parser", "deepseek_v4",
            "--enable-auto-tool-choice", "--reasoning-parser", "deepseek_v4",
            "--reasoning-config", spec({"reasoning_parser": "deepseek_v4", "reasoning_start_str": "",
                                        "reasoning_end_str": ""}),
            "--default-chat-template-kwargs.thinking=true",
            "--default-chat-template-kwargs.reasoning_effort=high",
            "--load-format", "instanttensor"]


def args_deepseek_0731(tp, models_dir):
    return ["--trust-remote-code", "--kv-cache-dtype", "fp8", "--block-size", "256",
            "--max-model-len", "auto", "--max-num-seqs", "8", "--max-num-batched-tokens", "8192",
            "--gpu-memory-utilization", "0.85", "--enable-prefix-caching",
            "--tokenizer-mode", "deepseek_v4", "--tool-call-parser", "deepseek_v4",
            "--enable-auto-tool-choice", "--reasoning-parser", "deepseek_v4",
            "--reasoning-config", spec({"reasoning_parser": "deepseek_v4", "reasoning_start_str": "",
                                        "reasoning_end_str": ""}),
            "--default-chat-template-kwargs.thinking=true",
            "--default-chat-template-kwargs.reasoning_effort=high",
            "--load-format", "b12x", "--moe-backend", "b12x", "--linear-backend", "b12x",
            "--attention-backend", "B12X", "--max-cudagraph-capture-size", "48",
            "--compilation-config", spec({"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"]}),
            "--speculative-config", spec({"method": "dspark", "num_speculative_tokens": 5,
                                          "draft_sample_method": "probabilistic", "attention_backend": "B12X"})]


def args_qwen36_35b(tp, models_dir):
    return ["--trust-remote-code", "--kv-cache-dtype", "fp8", "--attention-backend", "flashinfer",
            "--moe-backend", "marlin", "--gpu-memory-utilization", "0.4", "--max-model-len", "262144",
            "--max-num-seqs", "4", "--max-num-batched-tokens", "8192", "--enable-chunked-prefill",
            "--async-scheduling", "--enable-prefix-caching",
            "--speculative-config", spec({"method": "mtp", "num_speculative_tokens": 3, "moe_backend": "triton"}),
            "--load-format", "fastsafetensors", "--reasoning-parser", "qwen3",
            "--tool-call-parser", "qwen3_xml", "--enable-auto-tool-choice"]


def args_gemma4(tp, models_dir):
    return ["--max-model-len", "262144", "--gpu-memory-utilization", "0.7", "--load-format", "instanttensor",
            "--enable-prefix-caching", "--enable-auto-tool-choice", "--tool-call-parser", "gemma4",
            "--reasoning-parser", "gemma4", "--kv-cache-dtype", "fp8", "--max-num-batched-tokens", "8192"]


B12X_ENV = {"CUTE_DSL_ARCH": "sm_121a", "SAFETENSORS_FAST_GPU": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_SSM_CONV_STATE_LAYOUT": "DS", "VLLM_USE_AOT_COMPILE": "1", "VLLM_USE_MEGA_AOT_ARTIFACT": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1", "B12X_POLICY_MODE": "auto",
            "VLLM_ENABLE_ROCE_ALLREDUCE": "1", "VLLM_ROCE_ALLREDUCE_MAX_SIZE": "2MB"}

PROFILES = {
    "qwen3.8-27b": dict(
        repo="nvidia/Qwen3.8-27B-NVFP4", image="vllm-node", args=args_qwen38_27b, env={},
        note="eugr recipe qwen3.8-27b-nvfp4-dflash2 (DFlash2 draft used if Qwen3.8-27B-DFlash2 is on disk)"),
    "qwen3.8-flash-next": dict(
        repo="nvidia/Qwen3.8-Flash-Next-NVFP4", image="vllm-node-b12x", args=args_qwen38_flash, env=B12X_ENV,
        note="eugr recipe qwen3.8-flash-next-nvfp4-cluster. It targets local-inference-lab's checkpoint; "
             "if the nvidia one won't load try:  -- --load-format auto --quantization modelopt"),
    "glm-5.3-flash": dict(
        repo="nvidia/GLM-5.3-Flash-NVFP4", image="vllm-node-b12x", args=args_glm53_flash,
        env=dict(B12X_ENV, VLLM_ENABLE_PCIE_ALLREDUCE="0", INSTANTTENSOR_BACKEND="BUFFERED",
                 INSTANTTENSOR_BUFFER_SIZE="67108864", INSTANTTENSOR_CHUNK_SIZE="8388608",
                 INSTANTTENSOR_CONCURRENCY="1", INSTANTTENSOR_IO_DEPTH="3"),
        note="eugr recipe glm-5.3-flash. It targets local-inference-lab's Spark checkpoint; "
             "if the nvidia one won't load try:  -- --load-format auto --quantization modelopt "
             "--attention-backend auto"),
    "deepseek-v4-flash": dict(
        repo="deepseek-ai/DeepSeek-V4-Flash", image="vllm-node", args=args_deepseek,
        env={"DG_JIT_USE_NVRTC": "0", "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "VLLM_USE_BREAKABLE_CUDAGRAPH": "1"},
        note="eugr recipe deepseek-v4-flash"),
    "deepseek-v4-flash-0731": dict(
        repo="deepseek-ai/DeepSeek-V4-Flash-0731", image="vllm-node-b12x", args=args_deepseek_0731,
        env={"CUTE_DSL_ARCH": "sm_121a", "VLLM_USE_AOT_COMPILE": "1", "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
             "VLLM_USE_MEGA_AOT_ARTIFACT": "1", "VLLM_MEMORY_PROFILE_INCLUDE_ATTN": "1",
             "VLLM_USE_FLASHINFER_SAMPLER": "1", "VLLM_USE_B12X_WO_PROJECTION": "1", "VLLM_USE_B12X_MHC": "1",
             "VLLM_USE_B12X_FP8_GEMM": "1", "VLLM_USE_B12X_MOE": "1", "VLLM_USE_B12X_SPARSE_INDEXER": "1",
             "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_MOE_SKIP_PADDING": "0", "B12X_MLA_SM120_UNIFIED": "1",
             "B12X_MOE_FORCE_A8": "1"},
        note="eugr recipe deepseek-v4-flash-0731 (B12X stack, DSpark speculative decoding)"),
    "qwen3.6-35b-a3b": dict(
        repo="nvidia/Qwen3.6-35B-A3B-NVFP4", image="vllm-node", args=args_qwen36_35b,
        env={"VLLM_MARLIN_USE_ATOMIC_ADD": "1"},
        note="eugr recipe qwen3.6-35b-a3b-nvfp4 (Marlin MoE, MTP)"),
    "deepseek-v4.1-flash": dict(
        repo="nvidia/DeepSeek-V4.1-Flash-NVFP4", image="vllm-node", args=args_deepseek,
        env={"DG_JIT_USE_NVRTC": "0", "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "VLLM_USE_BREAKABLE_CUDAGRAPH": "1"},
        note="adapted from eugr's deepseek-v4-flash recipe (no V4.1 recipe exists yet)"),
    "gemma-4-26b": dict(
        repo="unsloth/gemma-4-26B-A4B-it-NVFP4", image="vllm-node", args=args_gemma4, env={}, solo=True,
        note="eugr's gemma4 recipe is solo-only: runs on THIS Spark only (TP=1), the other stays free"),
}


# -------------------------------------------------------------
# Nodes (this machine = head, plus the worker over ssh)
# -------------------------------------------------------------

class Node:
    def __init__(self, label, ssh=None):
        self.label, self.ssh = label, ssh
        self.iface = self.ip = self.hcas = None

    def run(self, cmd, tty=False, timeout=None, capture=True):
        """Run a shell command on this node. Returns CompletedProcess (never raises)."""
        if DRY:
            print(f"[dry-run] {self.label}$ {cmd}")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        argv = (["ssh", *SSH_OPTS] + (["-t"] if tty else []) + [self.ssh, cmd]) if self.ssh else ["bash", "-c", cmd]
        try:
            return subprocess.run(argv, capture_output=capture, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, 124, "", "timeout")

    def ok(self, cmd, **kw):
        return self.run(cmd, **kw).returncode == 0

    def out(self, cmd, **kw):
        return self.run(cmd, **kw).stdout.strip()


def detect_net(node):
    """Fill node.iface / ip / hcas from ibdev2netdev (same rules as eugr's autodiscover.sh)."""
    iface, hcas, ip = cfg("SPARK_IFACE"), cfg("SPARK_IB_HCA"), cfg("SPARK_HEAD_IP" if not node.ssh else "SPARK_WORKER_IP")
    if DRY:
        node.iface, node.hcas, node.ip = iface or "<iface>", hcas or "<hca>", ip or f"<{node.label}-ip>"
        return
    pairs = []   # "rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)"
    for line in node.out("ibdev2netdev 2>/dev/null").splitlines():
        p = line.split()
        if "(Up)" in line and len(p) >= 5:
            pairs.append((p[0], p[4]))
    if not iface:
        with_ip = [n for _, n in pairs if node.out(f"ip -4 -o addr show dev {shlex.quote(n)}")]
        iface = next((n for n in with_ip if "P" not in n), with_ip[0] if with_ip else "")
    if not hcas:
        hcas = ",".join(h for h, _ in pairs)
    if iface and not ip:
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", node.out(f"ip -4 -o addr show dev {shlex.quote(iface)}"))
        ip = m.group(1) if m else ""
    node.iface, node.hcas, node.ip = iface, hcas, ip


def make_nodes(need_worker=True):
    ssh = cfg("SPARK_WORKER_SSH")
    if not ssh and cfg("SPARK_WORKER_IP"):
        ssh = f"{getpass.getuser()}@{cfg('SPARK_WORKER_IP')}"
    if need_worker and not ssh:
        sys.exit("SPARK_WORKER_SSH is not set (e.g. SPARK_WORKER_SSH=cgray@10.13.1.21 in .env or the environment).")
    head, worker = Node("head"), Node("worker", ssh or None)
    nodes = [head] + ([worker] if need_worker else [])
    for n in nodes:
        detect_net(n)
    return head, worker if need_worker else None


# -------------------------------------------------------------
# Preflight
# -------------------------------------------------------------

def preflight(nodes, prof, models_dir, solo):
    """Returns (errors, warnings) as lists of strings. Model folders are handled by ensure_model()."""
    errs, warns = [], []
    for n in nodes:
        t = f"[{n.label}]"
        if not n.ok("true", timeout=20):
            errs.append(f"{t} can't reach {n.ssh} over ssh (passwordless key needed)")
            continue
        di = n.run("docker info 2>&1 >/dev/null | head -3", timeout=30)
        if di.returncode != 0 or di.stdout.strip():
            why = di.stdout.strip().splitlines()[0] if di.stdout.strip() else "unknown error"
            fix = ("add the user to the docker group (`sudo usermod -aG docker $USER`, then log in again)"
                   if "permission denied" in why.lower() else "start Docker (`sudo systemctl start docker`)"
                   if "cannot connect" in why.lower() else "check Docker")
            errs.append(f"{t} `docker info` fails: {why[:140]} -> {fix}")
        if not n.ok(f"docker image inspect {shlex.quote(prof['image'])} >/dev/null 2>&1"):
            errs.append(f"{t} Docker image '{prof['image']}' missing: run `./run_vllm_model_dgx_dual.py setup`")
        if not n.ok("nvidia-smi -L >/dev/null 2>&1"):
            errs.append(f"{t} nvidia-smi fails: no GPU / driver")
        if solo:
            continue
        if not n.iface or not n.ip:
            errs.append(f"{t} no ConnectX-7 port with an IPv4 found (ibdev2netdev); set SPARK_IFACE / IPs")
            continue
        if not n.hcas:
            errs.append(f"{t} no RoCE device is Up (ibdev2netdev); check the cable")
        mtu = n.out(f"cat /sys/class/net/{shlex.quote(n.iface)}/mtu")
        if mtu.isdigit() and int(mtu) < 9000:
            warns.append(f"{t} {n.iface} MTU is {mtu}; 9000 gives noticeably better throughput (netplan mtu: 9000)")
    if solo or errs:
        return errs, warns
    head, worker = nodes
    if head.ip == worker.ip:
        errs.append("head and worker report the same ConnectX IP")
    for a, b in ((head, worker), (worker, head)):
        if not a.ok(f"ping -c1 -W2 {shlex.quote(b.ip)} >/dev/null 2>&1", timeout=15):
            errs.append(f"[{a.label}] can't ping {b.label} {b.ip} over the ConnectX link")
    return errs, warns


# -------------------------------------------------------------
# Model discovery + copying a model to the other Spark
# -------------------------------------------------------------

def slug(name):
    return re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")


def human(n):
    n = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def scan_dirs(node, models_dir):
    """Names of the model folders (those holding a config.json) under models_dir on a node."""
    if node.ssh is None:
        try:
            return sorted(d for d in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, d, "config.json")))
        except OSError:
            return []
    out = node.out(f"find {shlex.quote(models_dir)} -mindepth 2 -maxdepth 2 -name config.json -printf '%h\\n' 2>/dev/null")
    return sorted({os.path.basename(l) for l in out.splitlines() if l})


def stats(node, path):
    """(file count, total bytes) of a model folder on a node, ignoring .part files; (0, 0) if absent."""
    if node.ssh is None:
        c = b = 0
        for root, _, names in os.walk(path):
            for n in names:
                if not n.endswith(".part"):
                    c += 1
                    b += os.path.getsize(os.path.join(root, n))
        return c, b
    out = node.out(f"find {shlex.quote(path)} -type f ! -name '*.part' -printf '%s\\n' 2>/dev/null "
                   f"| awk '{{s+=$1;c++}} END{{print c+0, s+0}}'")
    try:
        c, b = out.split()
        return int(c), int(b)
    except ValueError:
        return 0, 0


def file_list(node, path):
    """[(relative path, size)] of the finished files in a model folder on a node."""
    if node.ssh is None:
        return [(os.path.relpath(os.path.join(r, n), path), os.path.getsize(os.path.join(r, n)))
                for r, _, ns in os.walk(path) for n in ns if not n.endswith(".part")]
    out = node.out(f"cd {shlex.quote(path)} && find . -type f ! -name '*.part' -printf '%P\\t%s\\n'")
    return [(l.split("\t")[0], int(l.split("\t")[1])) for l in out.splitlines() if "\t" in l]


def free_bytes(node, path):
    if node.ssh is None:
        import shutil
        return shutil.disk_usage(path if os.path.isdir(path) else os.path.dirname(path)).free
    out = node.out(f"df -B1 --output=avail {shlex.quote(os.path.dirname(path))} | tail -1")
    return int(out) if out.isdigit() else 0


def link_host(worker):
    """ssh target to use for rsync: the worker's ConnectX IP (fast link), else the configured ssh target."""
    user = worker.ssh.split("@")[0] if "@" in worker.ssh else getpass.getuser()
    cand = f"{user}@{worker.ip}"
    try:
        ok = subprocess.run(["ssh", *SSH_OPTS, cand, "true"], capture_output=True, timeout=20).returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    if ok:
        return cand, f"the ConnectX link ({worker.ip})"
    return worker.ssh, f"{worker.ssh} (ssh to the ConnectX IP {worker.ip} failed, so this may be slower)"


def make_remote_dir(node, path):
    q = shlex.quote(path)
    if node.ok(f"mkdir -p {q}"):
        return True
    print(f"  {path} needs sudo on {node.label}; enter its password if asked:")
    sudo = f'sudo mkdir -p {q} && sudo chown "$(id -u):$(id -g)" {q}'
    return node.run(sudo, tty=True, capture=False).returncode == 0


def sync_model(head, worker, dirname, models_dir, src):
    """Copy models_dir/dirname from `src` to the other node with parallel rsyncs. Exits on failure."""
    dst = worker if src is head else head
    path = f"{models_dir}/{dirname}"
    files = file_list(src, path)
    total = sum(sz for _, sz in files)
    host, route = link_host(worker)
    print(f"  Copying {dirname} from {src.label} to {dst.label}: {len(files)} files, {human(total)}, over {route}.")
    if free_bytes(dst, path) < total * 1.02:
        sys.exit(f"  Not enough free disk space on {dst.label} for {human(total)}.")
    if not make_remote_dir(dst, path):
        sys.exit(f"  Can't create {path} on {dst.label}. Run there once:  sudo mkdir -p {models_dir} && sudo chown $USER {models_dir}")
    jobs = max(1, min(int(cfg("SYNC_JOBS", "4")), len(files)))
    rsh = os.environ.get("RSYNC_RSH") or "ssh -c aes128-gcm@openssh.com -o Compression=no -o StrictHostKeyChecking=no -o BatchMode=yes"
    buckets = [[0, []] for _ in range(jobs)]
    for rel, sz in sorted(files, key=lambda x: -x[1]):
        b = min(buckets, key=lambda x: x[0])
        b[0] += sz
        b[1].append(rel)
    import tempfile
    tmp = tempfile.mkdtemp(prefix="vllm_sync_")
    procs = []
    try:
        for i, (_, rels) in enumerate(b for b in buckets if b[1]):
            lf = os.path.join(tmp, f"files{i}.txt")
            with open(lf, "w") as f:
                f.write("\n".join(rels) + "\n")
            ends = [f"{path}/", f"{host}:{path}/"] if src is head else [f"{host}:{path}/", f"{path}/"]
            procs.append(subprocess.Popen(["rsync", "-a", "--partial", "--inplace", "-e", rsh,
                                           f"--files-from={lf}", *ends]))
        print(f"  Started {len(procs)} parallel rsyncs. The model can't start until this finishes; it resumes if interrupted.")
        t0, last, prev = time.time(), time.time(), 0
        while any(p.poll() is None for p in procs):
            time.sleep(1)
            if time.time() - last >= 15:
                done = stats(dst, path)[1]
                rate = (done - prev) / (time.time() - last)
                last, prev = time.time(), done
                eta = f", ~{int((total - done) / rate / 60)} min left" if rate > 1 else ""
                print(f"    {human(min(done, total))} / {human(total)} ({100 * min(done, total) // max(total, 1)}%)  {human(max(rate, 0))}/s{eta}")
        if any(p.returncode for p in procs):
            sys.exit(f"  rsync failed (exit codes {[p.returncode for p in procs]}); run again to resume.")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    c, b = stats(dst, path)
    if c < len(files) or b < total:
        sys.exit(f"  Copy verification failed on {dst.label}: {c} files / {human(b)}, expected {len(files)} / {human(total)}.")
    print(f"  Copied in {int(time.time() - t0)}s (~{human(total / max(time.time() - t0, 1))}/s) and verified.")


def ensure_model(head, worker, dirname, models_dir, need_both):
    """Make sure the model folder is complete where it's needed, copying it over if not."""
    path = f"{models_dir}/{dirname}"
    hs = stats(head, path)
    ws = stats(worker, path) if worker else (0, 0)
    if hs == ws and hs[0] > 0:
        print(f"  Model {dirname}: present and identical on both Sparks ({hs[0]} files, {human(hs[1])}).")
        return
    if hs[0] == 0 and ws[0] == 0:
        sys.exit(f"  Model {dirname} not found in {models_dir} on either Spark. Download it first "
                 f"(download_vllm_models_large.py).")
    if not need_both and hs[0] > 0:
        return
    src = head if hs[1] >= ws[1] else worker
    other = worker if src is head else head
    have = stats(other, path)
    why = "is missing" if have[0] == 0 else f"is incomplete or different ({have[0]} files, {human(have[1])})"
    print(f"  Model {dirname} {why} on {other.label}; it's on {src.label} ({hs[0] if src is head else ws[0]} files).")
    sync_model(head, worker, dirname, models_dir, src)


# -------------------------------------------------------------
# Profiles for models found on disk but not in PROFILES
# -------------------------------------------------------------

def generic_args(dirname, models_dir):
    mx = 32768
    try:
        with open(os.path.join(models_dir, dirname, "config.json")) as f:
            c = json.load(f)
        v = c.get("max_position_embeddings") or (c.get("text_config") or {}).get("max_position_embeddings")
        if isinstance(v, int) and v > 0:
            mx = min(v, 131072)
    except (OSError, ValueError):
        pass
    return ["--trust-remote-code", "--enable-prefix-caching", "--enable-chunked-prefill", "--max-model-len", str(mx),
            "--max-num-seqs", "8", "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.8"]


def generic_profile(dirname):
    return dict(repo=f"local/{dirname}", served=dirname, image="vllm-node", env={},
                args=lambda tp, md, d=dirname: generic_args(d, md),
                note="auto profile for a model without a tuned recipe (quantization is read from its config.json); "
                     "add flags after `--` if it needs them")


def build_catalog(head, worker, models_dir):
    """key -> {prof, dir, head, worker}: every tuned profile, plus a generic one for each other model on disk."""
    hd = set(scan_dirs(head, models_dir))
    wd = set(scan_dirs(worker, models_dir)) if worker else None
    cat = {}
    for k, p in PROFILES.items():
        d = p["repo"].split("/")[1]
        cat[k] = dict(prof=p, dir=d, head=d in hd, worker=None if wd is None else d in wd)
    known = {c["dir"] for c in cat.values()}
    for d in sorted((hd | (wd or set())) - known):
        key, i = slug(d), 2
        while key in cat:
            key, i = f"{slug(d)}-{i}", i + 1
        cat[key] = dict(prof=generic_profile(d), dir=d, head=d in hd, worker=None if wd is None else d in wd)
    return cat


def where(c):
    if c["head"] and c["worker"] is not False and (c["worker"] or c["worker"] is None):
        return "on both" if c["worker"] else "on head"
    if c["head"]:
        return "head only - will copy"
    if c["worker"]:
        return "worker only - will copy"
    return "not downloaded"


# -------------------------------------------------------------
# Container + vllm serve
# -------------------------------------------------------------

def node_env(node, prof, solo, args):
    if solo:
        env = {"VLLM_HOST_IP": node.ip or "127.0.0.1", "NCCL_SOCKET_IFNAME": "lo", "NCCL_IB_DISABLE": "1",
               "GLOO_SOCKET_IFNAME": "lo", "TP_SOCKET_IFNAME": "lo"}
    else:   # exactly the variables eugr's launch-cluster.sh injects
        env = {"VLLM_HOST_IP": node.ip, "MN_IF_NAME": node.iface, "UCX_NET_DEVICES": node.iface,
               "NCCL_SOCKET_IFNAME": node.iface, "NCCL_IB_HCA": node.hcas, "NCCL_IB_DISABLE": "0",
               "OMPI_MCA_btl_tcp_if_include": node.iface, "GLOO_SOCKET_IFNAME": node.iface,
               "TP_SOCKET_IFNAME": node.iface}
    env.update({"NCCL_IGNORE_CPU_AFFINITY": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    env.update(prof["env"])
    if not args.online:
        env["HF_HUB_OFFLINE"] = "1"
    if args.nccl_debug:
        env["NCCL_DEBUG"] = args.nccl_debug
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v
    return env


def docker_run_cmd(node, prof, solo, args, name, models_dir):
    mounts = [f"-v {shlex.quote(models_dir)}:/models:ro"]
    mounts += [f'-v "$HOME/{d}:/root/{d}"' for d in CACHE_DIRS]
    envs = " ".join(f"-e {shlex.quote(f'{k}={v}')}" for k, v in node_env(node, prof, solo, args).items())
    mk = "mkdir -p " + " ".join(f'"$HOME/{d}"' for d in CACHE_DIRS)
    run = (f"docker run -d --rm --gpus all --privileged --ipc=host --network host "
           f"--ulimit nofile=1048576:1048576 --entrypoint= --name {shlex.quote(name)} "
           f"{envs} {' '.join(mounts)} {shlex.quote(prof['image'])} sleep infinity")
    return f"{mk} && {run}"


def serve_cmd(prof, tp, args, models_dir, head_ip, rank, nnodes):
    dirname = prof["repo"].split("/")[1]
    cmd = ["vllm", "serve", f"/models/{dirname}", "--served-model-name", prof.get("served", prof["repo"]),
           "--host", "0.0.0.0", "--port", str(cfg("PORT", "8000"))]
    cmd += prof["args"](tp, models_dir)
    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.gpu_memory_utilization:
        cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    cmd += ["--tensor-parallel-size", str(tp)] + args.extra
    if nnodes > 1:
        cmd += ["--nnodes", str(nnodes), "--node-rank", str(rank), "--master-addr", head_ip,
                "--master-port", cfg("MASTER_PORT", "29501")]
        if rank > 0:
            cmd += ["--headless"]
    return shlex.join(cmd)


def docker_exec_bg(node, name, cmd):
    # Output goes to PID 1's stdout so `docker logs` shows it.
    inner = f"{cmd} >> /proc/1/fd/1 2>&1"
    return node.run(f"docker exec -d {shlex.quote(name)} bash -c {shlex.quote(inner)}")


def stop_all(nodes, name):
    for n in nodes:
        n.run(f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1; true")


VLLM_PROC_PATTERN = "[v]llm serve|[v]llm.entrypoints|[V]LLM::|[r]ay::|[r]aylet|[g]cs_server"


def kill_other_vllm(nodes, name):
    """Stop every other vLLM on these nodes: Docker containers (ours included), host `vllm serve`
    processes, their VLLM:: workers and Ray. Says what it found. Runs before the new model starts."""
    extra = {c.strip() for c in cfg("EXTRA_KILL_CONTAINERS", "qwen38-flash").split(",") if c.strip()}
    for n in nodes:
        found = []
        fmt = "'{{.Names}}\\t{{.Image}}\\t{{.Command}}'"
        for line in n.out(f"docker ps --format {fmt} 2>/dev/null").splitlines():
            cname = line.split("\t")[0]
            if cname == name or cname in extra or "vllm" in line.lower():
                found.append(cname)
        if found:
            print(f"  [{n.label}] stopping containers: {', '.join(found)}")
            n.run("docker rm -f " + " ".join(shlex.quote(c) for c in found) + " >/dev/null 2>&1; true")
        count = n.out(f"pgrep -fc '{VLLM_PROC_PATTERN}' 2>/dev/null")
        if count.isdigit() and int(count) > 0:
            print(f"  [{n.label}] killing {count} vLLM/Ray host process(es)")
        # sudo -n covers processes started by another user (e.g. root); harmless if it isn't allowed.
        n.run(f"pkill -TERM -f '{VLLM_PROC_PATTERN}'; sleep 3; pkill -KILL -f '{VLLM_PROC_PATTERN}'; "
              f"sudo -n pkill -KILL -f '{VLLM_PROC_PATTERN}' 2>/dev/null; true")
        if not found and not (count.isdigit() and int(count) > 0):
            print(f"  [{n.label}] no other vLLM running")
    if DRY:
        return
    for n in nodes:   # wait for the GPU memory to be released
        for _ in range(20):
            if n.run(f"pgrep -f '{VLLM_PROC_PATTERN}' >/dev/null 2>&1").returncode != 0:
                break
            time.sleep(1)
        else:
            print(f"  [warn] [{n.label}] vLLM processes are still alive (owned by another user? try: sudo pkill -9 -f vllm)")


def head_health(head, port):
    try:
        with urllib.request.urlopen(f"http://{head.ip or '127.0.0.1'}:{port}/health", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_ready(nodes, name, port, timeout):
    head = nodes[0]
    t0, last = time.time(), 0
    print(f"Waiting for the API on http://{head.ip}:{port} (first load of a big model can take many minutes)...")
    while time.time() - t0 < timeout:
        if head_health(head, port):
            print(f"\nReady after {int(time.time() - t0)}s.")
            return True
        for n in nodes:
            if not n.ok(f"docker ps -q -f name=^{shlex.quote(name)}$ | grep -q ."):
                print(f"\n[{n.label}] container exited.")
                return False
            # pgrep exit 1 = vllm gone; 127 (pgrep missing in image) is ignored.
            if n.run(f"docker exec {shlex.quote(name)} pgrep -f 'vllm serve' >/dev/null 2>&1").returncode == 1:
                print(f"\n[{n.label}] the vllm process died.")
                return False
        if time.time() - last >= 30:
            last = time.time()
            line = head.out(f"docker logs --tail 1 {shlex.quote(name)} 2>&1")[-150:]
            print(f"  {int(last - t0):>5}s  {line}")
        time.sleep(5)
    print("\nTimed out waiting for the API.")
    return False


def dump_logs(nodes, name, n_lines=40):
    for n in nodes:
        print(f"\n----- last {n_lines} log lines [{n.label}] -----")
        print(n.out(f"docker logs --tail {n_lines} {shlex.quote(name)} 2>&1"))


# -------------------------------------------------------------
# Commands
# -------------------------------------------------------------

def pick_profile(args, cat):
    key = getattr(args, "profile", None)
    if not key and sys.stdin.isatty():
        keys = list(cat)
        print("\nModels:")
        for i, k in enumerate(keys, 1):
            c = cat[k]
            tags = ("[solo: 1 Spark] " if c["prof"].get("solo") else "") + f"({where(c)})"
            print(f"  {i:>2}) {k:<30} {c['dir']:<38} {tags}")
        ans = input("Pick a number (q to quit): ").strip()
        if not ans.isdigit() or not 1 <= int(ans) <= len(keys):
            sys.exit(0)
        key = keys[int(ans) - 1]
    if key not in cat:   # also accept a folder name or a Hugging Face repo id
        match = [k for k, c in cat.items() if key in (c["dir"], c["prof"]["repo"])]
        if not match:
            sys.exit(f"Unknown model '{key}'. Run `list` to see them.")
        key = match[0]
    return key, cat[key]


def open_nodes():
    configured = bool(cfg("SPARK_WORKER_SSH") or cfg("SPARK_WORKER_IP"))
    return make_nodes(need_worker=configured)


def cmd_start(args):
    models_dir, name, port = cfg("MODELS_DIR", "/opt/ai/models"), cfg("CONTAINER_NAME", "vllm_node"), cfg("PORT", "8000")
    head, worker = open_nodes()
    key, c = pick_profile(args, build_catalog(head, worker, models_dir))
    prof, solo = c["prof"], c["prof"].get("solo", False)
    if not solo and worker is None:
        sys.exit("SPARK_WORKER_SSH is not set (e.g. SPARK_WORKER_SSH=cgray@10.13.1.21 in .env or the environment).")
    nodes = [head] if solo else [head, worker]
    tp = 1 if solo else 2
    print(f"\n{key}: {c['dir']}   image={prof['image']}   TP={tp} on {'1 Spark' if solo else '2 Sparks'}")
    print(f"  note: {prof['note']}")
    for n in nodes:
        print(f"  {n.label:<6} iface={n.iface} ip={n.ip} hca={n.hcas}")
    if not args.skip_checks and not DRY:
        errs, warns = preflight(nodes, prof, models_dir, solo)
        for w in warns:
            print(f"  [warn] {w}")
        if errs:
            for e in errs:
                print(f"  [FAIL] {e}")
            sys.exit(1)
        print("  preflight OK")
    if DRY:
        print(f"[dry-run] would verify {models_dir}/{c['dir']} on both Sparks and copy it where it's missing")
    else:
        ensure_model(head, worker, c["dir"], models_dir, need_both=not solo)

    if args.keep_others:
        stop_all(nodes, name)
    else:
        print("Stopping any other vLLM on " + " and ".join(n.label for n in nodes) + "...")
        kill_other_vllm(nodes, name)
    for n in nodes:                                  # head first, like launch-cluster.sh
        print(f"Starting container on {n.label}...")
        if n.run(docker_run_cmd(n, prof, solo, args, name, models_dir)).returncode != 0:
            stop_all(nodes, name)
            sys.exit(f"docker run failed on {n.label}")
    if not solo:                                     # worker (rank 1) first, then head (rank 0)
        print("Launching vllm worker (rank 1)...")
        docker_exec_bg(worker, name, serve_cmd(prof, tp, args, models_dir, head.ip, 1, 2))
    print("Launching vllm serve on head (rank 0)...")
    docker_exec_bg(head, name, serve_cmd(prof, tp, args, models_dir, head.ip, 0, 1 if solo else 2))
    if DRY:
        return
    if not wait_ready(nodes, name, port, int(cfg("HEALTH_TIMEOUT", "3600"))):
        dump_logs(nodes, name)
        stop_all(nodes, name)
        sys.exit(1)
    print(f"\nServing {prof.get('served', prof['repo'])} on {'1 Spark' if solo else '2 Sparks'}.\n  API   : http://{head.ip}:{port}/v1\n  Models: curl http://{head.ip}:{port}/v1/models"
          f"\n  Logs  : {sys.argv[0]} logs -f\n  Stop  : {sys.argv[0]} stop")


def cmd_stop(args):
    name = cfg("CONTAINER_NAME", "vllm_node")
    head, worker = make_nodes(need_worker=bool(cfg("SPARK_WORKER_SSH") or cfg("SPARK_WORKER_IP")))
    stop_all([n for n in (head, worker) if n], name)
    print("Stopped.")


def cmd_status(args):
    name, port = cfg("CONTAINER_NAME", "vllm_node"), cfg("PORT", "8000")
    head, worker = make_nodes(need_worker=bool(cfg("SPARK_WORKER_SSH") or cfg("SPARK_WORKER_IP")))
    for n in (head, worker):
        if n:
            st = n.out(f"docker ps --filter name=^{shlex.quote(name)}$ --format '{{{{.Status}}}}'") or "not running"
            print(f"  {n.label:<6} {n.ip or '?':<16} container: {st}")
    print(f"  API    {'healthy' if head_health(head, port) else 'not responding'}  (http://{head.ip}:{port})")


def cmd_logs(args):
    name = cfg("CONTAINER_NAME", "vllm_node")
    head, worker = make_nodes(need_worker=args.worker)
    node = worker if args.worker else head
    cmd = f"docker logs {'-f ' if args.follow else ''}--tail {args.tail} {shlex.quote(name)}"
    sys.exit(subprocess.call((["ssh", *SSH_OPTS, "-t", node.ssh, cmd]) if node.ssh else ["bash", "-c", cmd]))


def cmd_list(args):
    models_dir = cfg("MODELS_DIR", "/opt/ai/models")
    head, worker = open_nodes()
    cat = build_catalog(head, worker, models_dir)
    print(f"\n{'name':<30} {'folder':<38} {'image':<16} nodes  status")
    for k, c in cat.items():
        print(f"{k:<30} {c['dir']:<38} {c['prof']['image']:<16} {'1' if c['prof'].get('solo') else '2':<6} {where(c)}")
    print()


def cmd_check(args):
    models_dir = cfg("MODELS_DIR", "/opt/ai/models")
    head, worker = open_nodes()
    key, c = pick_profile(args, build_catalog(head, worker, models_dir))
    solo = c["prof"].get("solo", False)
    nodes = [head] if solo else [head, worker]
    if not solo and worker is None:
        sys.exit("SPARK_WORKER_SSH is not set.")
    for n in nodes:
        print(f"  {n.label:<6} iface={n.iface} ip={n.ip} hca={n.hcas}")
    errs, warns = preflight(nodes, c["prof"], models_dir, solo)
    path = f"{models_dir}/{c['dir']}"
    hs, ws = stats(head, path), stats(worker, path) if worker else (0, 0)
    print(f"  model {c['dir']}: head {hs[0]} files / {human(hs[1])}" + ("" if solo else f", worker {ws[0]} files / {human(ws[1])}"))
    if hs[0] == 0 and ws[0] == 0:
        errs.append(f"model not found on either Spark ({path}); download it first")
    elif not solo and hs != ws:
        warns.append("model differs between the Sparks: `start` will copy it over the ConnectX link first")
    for w in warns:
        print(f"  [warn] {w}")
    for e in errs:
        print(f"  [FAIL] {e}")
    print("  preflight OK" if not errs else "")
    sys.exit(1 if errs else 0)


def cmd_setup(args):
    """Clone eugr/spark-vllm-docker and run its build-and-copy.sh (pulls the images and copies them to the worker)."""
    d = os.path.expanduser(cfg("SETUP_DIR", "~/spark-vllm-docker"))
    if os.path.isdir(os.path.join(d, ".git")):
        subprocess.call(["git", "-C", d, "pull", "--ff-only"])
    elif subprocess.call(["git", "clone", EUGR_REPO, d]) != 0:
        sys.exit("git clone failed")
    # build-and-copy.sh ssh's as $USER (root here) unless told: give it the worker's user and ConnectX IP.
    ssh, wip = cfg("SPARK_WORKER_SSH"), cfg("SPARK_WORKER_IP")
    if not wip and ssh:
        _, w = make_nodes(need_worker=True)      # read its ConnectX IP
        wip = w.ip if w else ""
    copy = ["-c", wip] if wip else ["-c"]
    user_flag = ["-u", ssh.split("@")[0]] if "@" in ssh else []
    builds = [[]] if args.regular_only else [["--exp-b12x"]] if args.b12x_only else [[], ["--exp-b12x"]]
    for flags in builds:
        cmd = ["./build-and-copy.sh", *flags, *copy, "--copy-parallel", *user_flag]
        print(f"\n$ {' '.join(cmd)}   (in {d})")
        if subprocess.call(cmd, cwd=d) != 0:
            sys.exit("build-and-copy.sh failed; see eugr's docs/NETWORKING.md for the cluster/SSH setup it expects")


def main():
    global DRY
    load_env_file()
    ap = argparse.ArgumentParser(description="Run one vLLM model across both DGX Sparks over ConnectX-7 / RoCE.")
    sub = ap.add_subparsers(dest="cmd")

    def add_profile(p):
        p.add_argument("profile", nargs="?", help="model profile (see `list`)")

    p = sub.add_parser("start", help="launch a model on both Sparks")
    add_profile(p)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--gpu-memory-utilization", type=float)
    p.add_argument("-e", "--env", action="append", default=[], metavar="K=V", help="extra container env var")
    p.add_argument("--nccl-debug", choices=["VERSION", "WARN", "INFO", "TRACE"])
    p.add_argument("--online", action="store_true", help="allow Hugging Face lookups (default: offline)")
    p.add_argument("--skip-checks", action="store_true")
    p.add_argument("--keep-others", action="store_true",
                   help="don't stop other vLLM instances/containers first (default: stop them)")
    p.add_argument("--dry-run", action="store_true", help="print every command, run nothing")
    p.set_defaults(fn=cmd_start)
    p = sub.add_parser("check", help="preflight only")
    add_profile(p)
    p.set_defaults(fn=cmd_check)
    sub.add_parser("stop", help="stop the containers on both Sparks").set_defaults(fn=cmd_stop)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("list", help="show model profiles").set_defaults(fn=cmd_list)
    p = sub.add_parser("logs")
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("--worker", action="store_true", help="logs of the worker node instead of the head")
    p.add_argument("--tail", type=int, default=100)
    p.set_defaults(fn=cmd_logs)
    p = sub.add_parser("setup", help="clone eugr's repo and pull/copy the Docker images to both Sparks")
    p.add_argument("--b12x-only", action="store_true")
    p.add_argument("--regular-only", action="store_true")
    p.set_defaults(fn=cmd_setup)

    argv = sys.argv[1:] or ["start"]
    extra = []
    if "--" in argv:                       # everything after `--` goes to `vllm serve` untouched
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    if args.cmd is None:
        args = ap.parse_args(["start"])
    if args.cmd == "start":
        args.extra = extra
        DRY = args.dry_run
    args.fn(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted (containers keep running; use `stop` to stop them).")
        sys.exit(130)
