#!/usr/bin/env bash
# configure-cx7-direct.sh - Christopher Gray - 10/2026
#
# Configure the ConnectX-7 200G ports on a DGX Spark for a direct (no switch) link to a second DGX Spark.
#
# Layout (matches what LLDP showed on dgx-acer1 <-> aitopatom):
#   2 QSFP cables, and each physical port shows up as TWO netdevs (one per PCIe x4 half), so 4 netdevs total.
#   Every netdev gets its own point-to-point /30 so RoCE/NCCL can use all four halves for full bandwidth:
#
#       netdev           subnet (BASE=10.200)    node 1          node 2
#       enp1s0f0np0      BASE.1.0/30  (cable A)  BASE.1.1        BASE.1.2
#       enP2p1s0f0np0    BASE.2.0/30  (cable A)  BASE.2.1        BASE.2.2
#       enp1s0f1np1      BASE.3.0/30  (cable B)  BASE.3.1        BASE.3.2
#       enP2p1s0f1np1    BASE.4.0/30  (cable B)  BASE.4.1        BASE.4.2
#
#   - MTU 9000, no gateway, no DNS, no IPv6 (these links never touch the management/VLAN 200 network).
#   - NOT bonded: the halves are separate PCIe functions, a bond hides the RoCE devices NCCL needs.
#   - arp_ignore/arp_announce are set so the two halves on one cable don't answer ARP for each other's IPs.
#
# Usage (run on EACH Spark; node number must differ):
#     sudo ./configure-cx7-direct.sh 1 --dry-run      # validate only, changes nothing
#     sudo ./configure-cx7-direct.sh 1                # node 1
#     sudo ./configure-cx7-direct.sh 2                # node 2 (on the other Spark)
#     options: --mtu 9000   --base 10.200
#
# Safety: backs up /etc/netplan, validates the generated NetworkManager profiles offline first (NM silently
# drops invalid ones - that is what broke the VLAN config), applies under systemd-run so an SSH drop can't
# kill it, checks the management gateway still answers, and auto-rolls back after 4 minutes unless confirmed.
# Log if you get disconnected:  journalctl -u cx7-config --no-pager | tail -50

set -u

IFACES=(enp1s0f0np0 enP2p1s0f0np0 enp1s0f1np1 enP2p1s0f1np1)
NODE=""; DRY=0; MTU=9000; BASE=10.200

usage() { sed -n '2,/^set -u/p' "$0" | sed -n '/^# Usage/,/^# Safety/p' | sed 's/^# \{0,1\}//' | head -8; }

while [ $# -gt 0 ]; do
  case $1 in
    1|2)       NODE=$1 ;;
    --dry-run) DRY=1 ;;
    --mtu)     MTU=${2:?}; shift ;;
    --base)    BASE=${2:?}; shift ;;
    -h|--help) usage; exit 0 ;;
    *)         usage; exit 1 ;;
  esac
  shift
done
[ -n "$NODE" ] || { usage; exit 1; }
PEER=$((3 - NODE))
command -v netplan >/dev/null && command -v nmcli >/dev/null || { echo "netplan and nmcli are required"; exit 1; }

for n in "${IFACES[@]}"; do
  [ -d "/sys/class/net/$n" ] || { echo "ABORT: interface $n not found (check 'ip -br link')"; exit 1; }
done

# Re-run under systemd-run so the job survives an SSH disconnect
if [ "$DRY" = 0 ]; then
  [ "$EUID" -eq 0 ] || { echo "run with sudo"; exit 1; }
  if [ -z "${CX7_DETACHED:-}" ]; then
    exec systemd-run --unit=cx7-config --collect --pipe --wait --setenv=CX7_DETACHED=1 "$(readlink -f "$0")" "$NODE" --mtu "$MTU" --base "$BASE"
  fi
fi

gen() {
  echo "network:"
  echo "  version: 2"
  echo "  renderer: NetworkManager"
  echo "  ethernets:"
  local i=0 n
  for n in "${IFACES[@]}"; do
    i=$((i + 1))
    cat <<EOF
    $n:
      dhcp4: false
      dhcp6: false
      accept-ra: false
      link-local: []
      mtu: $MTU
      addresses: [$BASE.$i.$NODE/30]
EOF
  done
}

# NetworkManager silently refuses profiles that are method=manual without an address
chk() {
  local f=$1 m a
  [ -f "$f" ] || { echo "   missing $(basename "$f")"; return 1; }
  m=$(awk '/^\[ipv6\]/{s=1;next} /^\[/{s=0} s&&/^method=/{sub("method=","");print}' "$f")
  a=$(awk '/^\[ipv6\]/{s=1;next} /^\[/{s=0} s&&/^address/{print}' "$f")
  grep -q '^address1=' "$f" || { echo "   $(basename "$f"): no ipv4 address"; return 1; }
  [ "$m" != manual ] || [ -n "$a" ] || { echo "   $(basename "$f"): ipv6 manual with no address"; return 1; }
  return 0
}

# ---------- offline validation (touches nothing) ----------
T=$(mktemp -d /tmp/cx7-XXXX); mkdir -p "$T/etc/netplan"
gen > "$T/etc/netplan/70-cx7-direct.yaml"; chmod 600 "$T/etc/netplan/70-cx7-direct.yaml"
echo "== offline validation =="
netplan generate --root-dir "$T" 2>&1 | head -10
ok=1
for n in "${IFACES[@]}"; do
  chk "$T/run/NetworkManager/system-connections/netplan-$n.nmconnection" || ok=0
done
[ $ok = 1 ] || { echo "ABORT: generated profiles invalid. NOTHING CHANGED."; exit 1; }
echo "profiles OK"

if [ "$DRY" = 1 ]; then
  echo; echo "== would write /etc/netplan/70-cx7-direct.yaml =="; cat "$T/etc/netplan/70-cx7-direct.yaml"
  echo; echo "== existing netplan files that would be moved aside (they define these NICs) =="
  grep -lE "$(IFS='|'; echo "${IFACES[*]}")" /etc/netplan/*.yaml 2>/dev/null || echo "(none)"
  rm -rf "$T"; exit 0
fi
rm -rf "$T"

# ---------- backup + dead-man rollback ----------
GW=$(ip route show default | awk '/^default/{print $3; exit}')
BK=/root/netplan-backup-cx7-$(date +%F-%H%M%S); mkdir -p "$BK" && cp -a /etc/netplan/. "$BK/"
echo "backup: $BK   management gateway: ${GW:-none}"
cat > /root/cx7-rollback.sh <<RB
#!/bin/bash
[ -e /run/cx7-confirmed ] && exit 0
rm -f /etc/netplan/70-cx7-direct.yaml /etc/sysctl.d/90-cx7-arp.conf
cp -a $BK/*.yaml /etc/netplan/
netplan generate && netplan apply
echo "ROLLED BACK to $BK"
RB
chmod +x /root/cx7-rollback.sh; rm -f /run/cx7-confirmed
systemd-run --unit=cx7-rollback --on-active=240 /root/cx7-rollback.sh >/dev/null

# ---------- write config ----------
# move aside netplan files that already define these NICs (e.g. the qsfp-1/qsfp-2 DHCP profiles)
for f in /etc/netplan/*.yaml; do
  [ -f "$f" ] || continue
  if grep -qE "$(IFS='|'; echo "${IFACES[*]}")" "$f"; then echo "moving aside: $f"; mv "$f" "$BK/removed-$(basename "$f")"; fi
done
gen > /etc/netplan/70-cx7-direct.yaml; chmod 600 /etc/netplan/70-cx7-direct.yaml

{
  for n in "${IFACES[@]}"; do
    echo "net.ipv4.conf.$n.arp_ignore = 1"
    echo "net.ipv4.conf.$n.arp_announce = 2"
    echo "net.ipv4.conf.$n.rp_filter = 2"
  done
} > /etc/sysctl.d/90-cx7-arp.conf

netplan generate || { echo "generate failed"; /root/cx7-rollback.sh; exit 1; }

# drop auto-created DHCP profiles still bound to these NICs (never touches the management NIC)
nmcli -t -f NAME,DEVICE con show | while IFS=: read -r name dev; do
  for n in "${IFACES[@]}"; do
    [ "$dev" = "$n" ] && case $name in netplan-*) ;; *) nmcli con delete "$name" >/dev/null 2>&1 && echo "deleted stale profile: $name" ;; esac
  done
done

netplan apply
sysctl -q -p /etc/sysctl.d/90-cx7-arp.conf 2>/dev/null
sleep 8

# ---------- verify ----------
fail=0; warn=0; i=0
for n in "${IFACES[@]}"; do
  i=$((i + 1)); want="$BASE.$i.$NODE/30"
  for t in $(seq 1 20); do ip -4 -o addr show dev "$n" | grep -q " $want" && break; sleep 1; done
  ip -4 -o addr show dev "$n" | grep -q " $want" || { echo "FAIL: $n missing $want"; fail=1; continue; }
  [ "$(cat /sys/class/net/$n/mtu)" = "$MTU" ]    || { echo "FAIL: $n mtu $(cat /sys/class/net/$n/mtu) != $MTU"; fail=1; }
  [ "$(cat /sys/class/net/$n/carrier 2>/dev/null)" = 1 ] || echo "WARN: $n no carrier (cable?)"
  if ping -c2 -W1 -I "$n" "$BASE.$i.$PEER" >/dev/null 2>&1; then echo "ok:   $n $want  <->  $BASE.$i.$PEER"
  else echo "WARN: $n $want  peer $BASE.$i.$PEER not answering (expected until the other node is configured)"; warn=1; fi
done
if [ -n "$GW" ]; then ping -c3 -W2 "$GW" >/dev/null 2>&1 || { echo "FAIL: management gateway $GW unreachable"; fail=1; }; fi

if [ $fail = 0 ]; then
  touch /run/cx7-confirmed; systemctl stop cx7-rollback.timer
  echo; echo "SUCCESS - node $NODE configured. Backup: $BK"
  [ $warn = 0 ] || echo "Peer links not answering yet: run this script with the other node number on the other Spark, then re-test."
else
  echo "FAILED -> rolling back"; /root/cx7-rollback.sh
fi

cat <<EOF

Test (after BOTH nodes are done):
  node 1:  iperf3 -s -B $BASE.1.1          # repeat per link with the matching IP
  node 2:  iperf3 -c $BASE.1.1 -P 8 -t 10  # one link ~100G+ TCP; run all four links at once to see the aggregate
  RDMA:    rdma link show ; ibdev2netdev   # maps rocep*/roceP* devices to these netdevs
  jumbo:   ping -M do -s 8972 -c3 $BASE.1.$PEER
EOF
