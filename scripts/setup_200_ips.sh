#!/usr/bin/env bash
# setup_200_ips.sh
#
# Creates N network namespaces, each with a unique source IP in the
# 203.0.113.0/24 range (RFC 5737 documentation range → recognized by
# collector/enrich.py as ASN 9009 / M247).
#
# The namespaces can reach 192.168.122.0/24 through the br-attack bridge.
# Return traffic is routed by adding a static route on the victim VM.
#
# Usage:
#     sudo bash scripts/setup_200_ips.sh          # default N=200
#     sudo bash scripts/setup_200_ips.sh 100      # custom N
#
# Cleanup:
#     sudo bash scripts/cleanup_200_ips.sh
#
set -euo pipefail

N="${1:-200}"
BRIDGE="br-attack"
HOST_IFACE="virbr0"        # libvirt bridge to victim VM
ATTACK_NET="203.0.113.0/24"
BRIDGE_IP="203.0.113.1"

echo "===================================================="
echo "  Setting up $N attacker namespaces"
echo "===================================================="
echo "  bridge:     $BRIDGE"
echo "  host iface: $HOST_IFACE"
echo "  attack net: $ATTACK_NET"
echo "  bridge IP:  $BRIDGE_IP"
echo "===================================================="
echo

# 0. Preflight
if [ "$EUID" -ne 0 ]; then
    echo "✗ must run as root (use sudo)" >&2
    exit 1
fi

if ! ip link show "$HOST_IFACE" >/dev/null 2>&1; then
    echo "✗ host interface '$HOST_IFACE' not found" >&2
    echo "  available interfaces:" >&2
    ip -br link show >&2
    exit 1
fi

# 1. Create bridge
if ip link show "$BRIDGE" >/dev/null 2>&1; then
    echo "  bridge '$BRIDGE' already exists — reusing"
else
    ip link add name "$BRIDGE" type bridge
    echo "  created bridge '$BRIDGE'"
fi

ip link set "$BRIDGE" up

# Assign bridge IP (idempotent)
if ! ip addr show "$BRIDGE" | grep -q "$BRIDGE_IP"; then
    ip addr add "$BRIDGE_IP/24" dev "$BRIDGE"
    echo "  assigned $BRIDGE_IP/24 to $BRIDGE"
fi

# 2. Enable IP forwarding
sysctl -qw net.ipv4.ip_forward=1
echo "  ip_forward=1"

# 3. Route back to VM subnet via host iface
ip route replace 192.168.122.0/24 dev "$HOST_IFACE" 2>/dev/null || true
echo "  route 192.168.122.0/24 via $HOST_IFACE"

# 4. Allow forwarding br-attack <-> virbr0
iptables -C FORWARD -i "$BRIDGE" -o "$HOST_IFACE" -j ACCEPT 2>/dev/null || \
    iptables -I FORWARD -i "$BRIDGE" -o "$HOST_IFACE" -j ACCEPT
iptables -C FORWARD -i "$HOST_IFACE" -o "$BRIDGE" -j ACCEPT 2>/dev/null || \
    iptables -I FORWARD -i "$HOST_IFACE" -o "$BRIDGE" -j ACCEPT
echo "  FORWARD rules added"

# 5. Create namespaces
echo
echo "  creating $N namespaces..."
created=0
for i in $(seq 1 $N); do
    ns="attacker_$i"
    veth_host="veth_h_$i"
    veth_ns="veth_n_$i"
    # IP range: 203.0.113.2 through 203.0.113.254 (skip .1 which is bridge)
    ip_last=$(( (i - 1) % 253 + 2 ))
    ip_addr="203.0.113.$ip_last"

    # Clean up if exists
    ip netns del "$ns" 2>/dev/null || true
    ip link del "$veth_host" 2>/dev/null || true

    # Create namespace + veth pair
    ip netns add "$ns"
    ip link add "$veth_host" type veth peer name "$veth_ns"
    ip link set "$veth_ns" netns "$ns"

    # Attach host end to bridge
    ip link set "$veth_host" master "$BRIDGE"
    ip link set "$veth_host" up

    # Configure namespace
    ip netns exec "$ns" ip link set lo up
    ip netns exec "$ns" ip link set "$veth_ns" up
    ip netns exec "$ns" ip addr add "$ip_addr/24" dev "$veth_ns"
    ip netns exec "$ns" ip route add default via "$BRIDGE_IP"

    created=$((created + 1))

    if [ $((i % 25)) -eq 0 ]; then
        echo "    $i / $N"
    fi
done

echo
echo "  ✓ created $created namespaces"

# 6. Tell the VM how to route replies back
echo
echo "  adding return route on victim VM (192.168.122.30)..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes naveen@192.168.122.30 \
       "sudo -n ip route replace 203.0.113.0/24 via 192.168.122.1" 2>/dev/null; then
    echo "  ✓ VM return route added"
else
    echo "  ⚠ could not add VM route via SSH — add manually:"
    echo "      ssh naveen@192.168.122.30 \\"
    echo "        'sudo ip route add 203.0.113.0/24 via 192.168.122.1'"
fi

# 7. Test one namespace
echo
echo "  testing attacker_1 → victim..."
if ip netns exec attacker_1 curl -s -m 3 "http://192.168.122.30/" >/dev/null 2>&1; then
    echo "  ✓ attacker_1 can reach the victim"
else
    echo "  ✗ attacker_1 CANNOT reach the victim — check routes"
fi

echo
echo "===================================================="
echo "  Setup complete"
echo "===================================================="
echo "  Test one:  sudo ip netns exec attacker_1 curl http://192.168.122.30/"
echo "  Attack:    sudo bash scripts/attack_200_real_ips.sh 30"
echo "  Cleanup:   sudo bash scripts/cleanup_200_ips.sh"
echo "===================================================="