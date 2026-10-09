#!/usr/bin/env python3
# cx7-speedtest.py - Christopher Gray - 10/2026
#
#  Install:
#    wget https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/cx7-speedtest.py && chmod +x cx7-speedtest.py
#
# Usage:
#   python3 cx7-speedtest.py --ssh user@10.13.1.21
#
# True link speed test between two DGX Sparks cabled directly over ConnectX-7 (the 4 netdevs configured by
# configure-cx7-direct-dgx-spark.sh: 10.200.{1..4}.{1,2}/30).  Python 3 stdlib only, nice colour output.
#
#   Peer needs:  sudo apt install iperf3 perftest          (perftest = ib_write_bw, only for --rdma)
#
#   Easiest (one command, needs passwordless ssh to the peer's management IP or name):
#       ./cx7-speedtest.py --ssh user@10.13.1.21              # TCP: each link alone, then all 4 together
#       ./cx7-speedtest.py --ssh user@10.13.1.21 --rdma       # + RDMA (ib_write_bw) = the real wire speed
#       ./cx7-speedtest.py --ssh user@10.13.1.21 --reverse    # also test the other direction
#
#   No ssh between the Sparks?  On the peer run:   ./cx7-speedtest.py server
#   then on this one:                              ./cx7-speedtest.py
#
#   Options: --time 10  --streams 4  --base 10.200  --json out.json  --demo (fake numbers, to preview the layout)
#
# Reading the numbers: each physical 200G port is two PCIe x4 halves (~100 Gbit/s each), so the bars are scaled
# to 100 Gbit/s per link and the full two-cable total is ~400 Gbit/s. Kernel TCP is CPU-bound and usually lands
# lower than RDMA; NCCL/RDMA is the number that matters for multi-Spark inference/training.
#
# NOTE: always run the client on one Spark and the server on the OTHER. Testing against your own IP goes over
# loopback and never touches the cables (this tool picks the peer's address automatically to avoid that).

import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

IFACES = ["enp1s0f0np0", "enP2p1s0f0np0", "enp1s0f1np1", "enP2p1s0f1np1"]
CABLE = ["A", "A", "B", "B"]
TCP_PORT = 5200      # + link number
RDMA_PORT = 18515    # + link number
HALF = 100.0         # Gbit/s a single PCIe half can carry
W = 80

TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else str(s)


bold = lambda s: c("1", s)
dim = lambda s: c("2", s)
red = lambda s: c("31", s)
green = lambda s: c("32", s)
yellow = lambda s: c("33", s)
cyan = lambda s: c("36", s)


def tone(g, scale):
    r = g / scale
    return "32" if r >= 0.8 else "33" if r >= 0.4 else "31"


def bar(g, scale=HALF, w=26):
    n = max(0, min(w, round(g / scale * w)))
    return c(tone(g, scale), "█" * n) + dim("░" * (w - n))


def section(title):
    print("\n" + bold(cyan(("━━ " + title + " ").ljust(W, "━"))))


def sh(cmd, timeout=None):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return subprocess.CompletedProcess(cmd, 255, "", str(e))


def rsh(host, cmd, timeout=20):
    return sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, cmd], timeout)


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


# ---------------------------------------------------------------- discovery
def rdma_dev(ifn):
    for d in glob.glob("/sys/class/infiniband/*"):
        if read(f"{d}/ports/1/gid_attrs/ndevs/0") == ifn:
            return os.path.basename(d)
    return None


def detect(base):
    links = []
    for i, ifn in enumerate(IFACES, 1):
        out = sh(["ip", "-4", "-o", "addr", "show", "dev", ifn]).stdout
        m = re.search(rf"inet ({re.escape(base)}\.{i}\.(\d+))/", out)
        ip, node = (m.group(1), int(m.group(2))) if m else (None, None)
        speed = read(f"/sys/class/net/{ifn}/speed")
        links.append(dict(
            i=i, ifn=ifn, cable=CABLE[i - 1], ip=ip, node=node,
            peer=f"{base}.{i}.{3 - node}" if node in (1, 2) else None,
            speed=int(speed) // 1000 if speed and speed.lstrip("-").isdigit() and int(speed) > 0 else None,
            mtu=int(read(f"/sys/class/net/{ifn}/mtu") or 0),
            carrier=read(f"/sys/class/net/{ifn}/carrier") == "1",
            rdma=rdma_dev(ifn)))
    return links


def rtt_us(l):
    p = sh(["ping", "-c", "3", "-W", "1", "-q", "-I", l["ifn"], l["peer"]])
    m = re.search(r"= [\d.]+/([\d.]+)/", p.stdout)
    return float(m.group(1)) * 1000 if m else None


def jumbo_ok(l):
    return sh(["ping", "-M", "do", "-s", str(l["mtu"] - 28), "-c", "1", "-W", "1", "-I", l["ifn"], l["peer"]]).returncode == 0


GID_SRC = r'''
import os, sys
dev, ip = sys.argv[1], sys.argv[2]
base = "/sys/class/infiniband/%s/ports/1" % dev
a, b, c, d = map(int, ip.split("."))
tail = "ffff:%02x%02x:%02x%02x" % (a, b, c, d)
try:
    idxs = sorted(os.listdir(base + "/gids"), key=int)
except OSError:
    idxs = []
for idx in idxs:
    try:
        g = open("%s/gids/%s" % (base, idx)).read().strip()
        t = open("%s/gid_attrs/types/%s" % (base, idx)).read().strip()
    except OSError:
        continue
    if g.endswith(tail) and "v2" in t:
        print(idx)
        sys.exit(0)
print(-1)
'''


def gid_index(dev, ip, host=None):
    if host:
        p = rsh(host, f"python3 -c {shlex.quote(GID_SRC)} {dev} {ip}")
    else:
        p = sh([sys.executable, "-c", GID_SRC, dev, ip])
    try:
        return int(p.stdout.strip())
    except ValueError:
        return -1


# ---------------------------------------------------------------- running tests
def run_parallel(fns, label, secs):
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    t0 = time.time()
    with cf.ThreadPoolExecutor(len(fns)) as ex:
        futs = [ex.submit(f) for f in fns]
        k = 0
        while not all(f.done() for f in futs):
            if TTY:
                sys.stdout.write(f"\r  {cyan(frames[k % 10])} {label}  {int(time.time() - t0)}s / ~{secs}s ")
                sys.stdout.flush()
            k += 1
            time.sleep(0.1)
        if TTY:
            sys.stdout.write("\r" + " " * (W - 2) + "\r")
        return [f.result() for f in futs]


def iperf(l, a, reverse=False):
    cmd = ["iperf3", "-c", l["peer"], "-p", str(TCP_PORT + l["i"]), "-B", l["ip"], "-P", str(a.streams),
           "-t", str(a.time), "-O", "1", "-J"]
    if reverse:
        cmd.append("-R")
    p = sh(cmd, timeout=a.time + 30)
    try:
        d = json.loads(p.stdout)
    except ValueError:
        return dict(error=(p.stderr or p.stdout or "iperf3 failed").strip()[:80])
    if "error" in d:
        return dict(error=d["error"][:80])
    e = d["end"]
    return dict(gbps=e["sum_received"]["bits_per_second"] / 1e9,
                retr=e["sum_sent"].get("retransmits"),
                cpu=e["cpu_utilization_percent"]["host_total"])


def rdma_one(l, a):
    dev = l["rdma"]
    if not dev:
        return dict(error="no RoCE device mapped to this netdev")
    lidx = gid_index(dev, l["ip"])
    ridx = gid_index(dev, l["peer"], a.ssh)
    if lidx < 0 or ridx < 0:
        return dict(error=f"no RoCEv2 GID for {l['ip']} / {l['peer']} on {dev}")
    port = RDMA_PORT + l["i"]
    opts = ["-d", dev, "-p", str(port), "-q", "4", "-s", "1048576", "-D", str(a.time), "--report_gbits", "-F"]
    srv = subprocess.Popen(["ssh", "-o", "BatchMode=yes", a.ssh, "ib_write_bw " + " ".join(opts + ["-x", str(ridx)])],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    p = sh(["ib_write_bw"] + opts + ["-x", str(lidx), l["peer"]], timeout=a.time + 30)
    try:
        srv.wait(timeout=5)
    except subprocess.TimeoutExpired:
        srv.terminate()
    rows = re.findall(r"^\s*\d+\s+\d+\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s*$", p.stdout, re.M)
    if not rows:
        return dict(error=(p.stderr or p.stdout or "ib_write_bw failed").strip().splitlines()[-1][:80])
    return dict(gbps=float(rows[-1]))


# ---------------------------------------------------------------- output
def row(l, r):
    name = f"  {l['i']}  {l['ifn']:<14} {dim('cable ' + l['cable'])}  "
    if "error" in r:
        print(name + red("✗ " + r["error"]))
        return
    extra = ""
    if r.get("retr") is not None:
        extra += f"  retr {r['retr']}"
    if r.get("cpu") is not None:
        extra += f"  cpu {r['cpu']:.0f}%"
    g = r["gbps"]
    print(f"{name}{bar(g)} {c(tone(g, HALF), f'{g:6.1f}')} Gbit/s {dim(f'{g / HALF * 100:3.0f}%')}{dim(extra)}")


def preflight_table(links, pings):
    print(dim("  #  NIC             cable  local → peer                  speed   MTU  RoCE dev        RTT      jumbo"))
    for l, (rt, jb) in zip(links, pings):
        sp = f"{l['speed']}G" if l["speed"] else red("down")
        rd = l["rdma"] or dim("—")
        rts = f"{rt:5.0f} µs" if rt else red("  fail ")
        print(f"  {l['i']}  {l['ifn']:<14}  {l['cable']}      {l['ip']:>10} → {l['peer']:<12}  {sp:>5}  {l['mtu']:>5}  "
              f"{rd:<14}  {rts}  {green('✓') if jb else red('✗')}")


def summary_line(name, res):
    g = [r.get("gbps", 0) for r in res]
    ca, cb = g[0] + g[1], g[2] + g[3]
    tot = ca + cb
    cells = "".join(f"{x:7.1f}" for x in g)
    print(f"  {name:<22}{cells}  {dim('│')} A {ca:6.1f}  B {cb:6.1f} {dim('│')} {bold(c(tone(tot, 400), f'{tot:6.1f}'))}")
    return tot


def finish(results, a):
    section("Summary")
    print(dim(f"  {'test':<22}{'link1':>7}{'link2':>7}{'link3':>7}{'link4':>7}  │ cable totals     │ total Gbit/s"))
    best = (None, 0)
    for name, res in results.items():
        if all("gbps" in r for r in res):
            if name.endswith("(all links)"):
                t = summary_line(name, res)
            else:
                g = [r["gbps"] for r in res]
                print(f"  {name:<22}" + "".join(f"{x:7.1f}" for x in g))
                t = sum(g)
            if t > best[1] and name.endswith("(all links)"):
                best = (name, t)
    if best[0]:
        print(f"\n  best aggregate  {bar(best[1], 400, 40)}  {bold(f'{best[1]:.1f}')} of ~400 Gbit/s ({best[1] / 4:.0f}%)")
    print(dim("\n  bars: % of 100 Gbit/s per PCIe half | green ≥80%  yellow ≥40%  red <40%"))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2)
        print(dim(f"  saved {a.json}"))


# ---------------------------------------------------------------- server mode
def serve(a):
    links = detect(a.base)
    procs = []
    section("iperf3 servers")
    for l in links:
        if not l["ip"]:
            print(red(f"  {l['ifn']}: no {a.base}.{l['i']}.x address - run configure-cx7-direct-dgx-spark.sh first"))
            continue
        p = subprocess.Popen(["iperf3", "-s", "-B", l["ip"], "-p", str(TCP_PORT + l["i"])],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
        print(f"  {green('●')} {l['ifn']:<14} listening on {l['ip']}:{TCP_PORT + l['i']}")
    time.sleep(0.5)
    if any(p.poll() is not None for p in procs):
        print(red("  a server exited - is iperf3 installed and is something else on those ports?"))
    print(dim("\n  Now run ./cx7-speedtest.py on the other Spark.  Ctrl-C to stop."))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        for p in procs:
            p.terminate()


# ---------------------------------------------------------------- client mode
def demo_links(base):
    return [dict(i=i, ifn=IFACES[i - 1], cable=CABLE[i - 1], ip=f"{base}.{i}.1", node=1, peer=f"{base}.{i}.2",
                 speed=200, mtu=9000, carrier=True,
                 rdma=["rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1", "roceP2p1s0f1"][i - 1]) for i in range(1, 5)]


def client(a):
    t_all = time.time()
    links = demo_links(a.base) if a.demo else detect(a.base)
    section("DGX Spark ConnectX-7 direct-link speed test")
    me = socket.gethostname()
    peer_name = "peer"
    if a.ssh and not a.demo:
        r = rsh(a.ssh, "hostname")
        if r.returncode != 0:
            sys.exit(red(f"\n  ssh to {a.ssh} failed (needs key-based login): {r.stderr.strip()[:100]}"))
        peer_name = r.stdout.strip()
    print(f"  {bold(me)} ⇄ {bold(peer_name)}   {dim(f'{a.streams} streams × {a.time}s per test, base {a.base}')}")

    bad = [l for l in links if not l["ip"]]
    if bad:
        sys.exit(red(f"\n  no {a.base}.x address on: {', '.join(l['ifn'] for l in bad)}  (run configure-cx7-direct-dgx-spark.sh)"))
    for tool in ["iperf3"] + (["ib_write_bw"] if a.rdma else []):
        if not a.demo and not shutil.which(tool):
            sys.exit(red(f"\n  {tool} not installed here:  sudo apt install iperf3 perftest"))
    if a.rdma and not a.ssh and not a.demo:
        sys.exit(red("\n  --rdma needs --ssh user@peer (it starts ib_write_bw on the peer for each test)"))

    section("Pre-flight")
    pings = [(1.0 * 30 + l["i"] * 3, True) if a.demo else (rtt_us(l), jumbo_ok(l)) for l in links]
    preflight_table(links, pings)
    for l in links:
        if l["peer"] and sh(["ip", "route", "get", l["peer"]]).stdout.startswith("local"):
            sys.exit(red(f"\n  {l['peer']} is a LOCAL address - you're testing loopback. Run on the other Spark."))
    if not a.demo and any(p[0] is None for p in pings):
        sys.exit(red("\n  a link doesn't answer ping - fix that first (is the peer configured with the other node number?)"))

    results = {}
    started_remote = False
    try:
        if a.ssh and not a.demo:
            rsh(a.ssh, "for f in /tmp/cx7-iperf-*.pid; do kill $(cat $f) 2>/dev/null; done; rm -f /tmp/cx7-iperf-*.pid")
            started_remote = True
            cmd = " ; ".join(f"iperf3 -s -B {l['peer']} -p {TCP_PORT + l['i']} -D --pidfile /tmp/cx7-iperf-{l['i']}.pid" for l in links)
            r = rsh(a.ssh, cmd)
            if r.returncode != 0:
                sys.exit(red(f"\n  couldn't start iperf3 servers on the peer: {r.stderr.strip()[:100]}"))
            time.sleep(1)
        if not a.demo:
            for l in links:
                try:
                    socket.create_connection((l["peer"], TCP_PORT + l["i"]), timeout=2, source_address=(l["ip"], 0)).close()
                except OSError:
                    sys.exit(red(f"\n  nothing listening on {l['peer']}:{TCP_PORT + l['i']} - run './cx7-speedtest.py server' on the peer, or use --ssh"))

        def tcp(label, reverse=False):
            if a.demo:
                import random
                random.seed(len(label))
                return [dict(gbps=random.uniform(82, 112), retr=random.randint(0, 40), cpu=random.uniform(20, 50)) for _ in links]
            return run_parallel([lambda l=l: iperf(l, a, reverse) for l in links], label, a.time + 1)

        def rdma(label, one_by_one):
            if a.demo:
                import random
                random.seed(len(label) + 7)
                return [dict(gbps=random.uniform(90, 99)) for _ in links]
            if one_by_one:
                return [run_parallel([lambda l=l: rdma_one(l, a)], f"{l['ifn']}", a.time + 3)[0] for l in links]
            return run_parallel([lambda l=l: rdma_one(l, a) for l in links], label, a.time + 3)

        phases = [("TCP", False)] + ([("TCP reverse", True)] if a.reverse else [])
        for pname, rev in phases:
            section(f"{pname} — one link at a time" + ("  (peer → here)" if rev else ""))
            res = []
            for l in links:
                if a.demo:
                    r = tcp(l["ifn"], rev)[l["i"] - 1]
                else:
                    r = run_parallel([lambda l=l: iperf(l, a, rev)], l["ifn"], a.time + 1)[0]
                row(l, r)
                res.append(r)
            results[f"{pname} (single)"] = res
            section(f"{pname} — all four links together" + ("  (peer → here)" if rev else ""))
            res = tcp(f"{pname} all", rev)
            for l, r in zip(links, res):
                row(l, r)
            results[f"{pname} (all links)"] = res
        if a.rdma:
            section("RDMA ib_write_bw — one link at a time")
            res = rdma("rdma single", True)
            for l, r in zip(links, res):
                row(l, r)
            results["RDMA (single)"] = res
            section("RDMA ib_write_bw — all four links together")
            res = rdma("rdma all", False)
            for l, r in zip(links, res):
                row(l, r)
            results["RDMA (all links)"] = res
    finally:
        if started_remote:
            rsh(a.ssh, "for f in /tmp/cx7-iperf-*.pid; do kill $(cat $f) 2>/dev/null; done; rm -f /tmp/cx7-iperf-*.pid")

    finish(results, a)
    print(dim(f"  finished in {time.time() - t_all:.0f}s\n"))


def main():
    ap = argparse.ArgumentParser(description="Speed test between two DGX Sparks over the direct CX-7 links")
    ap.add_argument("mode", nargs="?", choices=["client", "server"], default="client")
    ap.add_argument("--ssh", metavar="USER@PEER", help="start/stop the peer's servers over ssh (use its management IP/name)")
    ap.add_argument("--rdma", action="store_true", help="also run ib_write_bw (needs --ssh and the perftest package)")
    ap.add_argument("--reverse", action="store_true", help="also test peer → here")
    ap.add_argument("--time", type=int, default=10)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--base", default="10.200")
    ap.add_argument("--json", metavar="FILE")
    ap.add_argument("--demo", action="store_true", help="fake numbers, just to preview the output")
    a = ap.parse_args()
    if a.mode == "server":
        serve(a)
    else:
        client(a)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(red("\n  interrupted"))
        sys.exit(130)
