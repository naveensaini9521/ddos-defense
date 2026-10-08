"""Extract DDoS-relevant features from packet captures.

30 flow-level features (with renamed fields to avoid conflicts with
HTTP features: pkt_pkt_rate, pkt_byte_rate, pkt_syn_ratio).
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from core.logging import get_logger

log = get_logger("collector.packet_features")


# ---------------------------------------------------------------------------
# Feature names (renamed to avoid collisions with HTTP features)
# ---------------------------------------------------------------------------

PACKET_FEATURE_NAMES: list[str] = [
    "flow_duration",
    "total_pkts",
    "total_bytes",
    "pkt_pkt_rate",       # was: pkt_rate (conflicts with HTTP)
    "pkt_byte_rate",      # was: byte_rate (conflicts with HTTP)
    "fwd_pkt_count",
    "bwd_pkt_count",
    "fwd_byte_count",
    "bwd_byte_count",
    "fwd_bwd_ratio",
    "pkt_len_min",
    "pkt_len_max",
    "pkt_len_mean",
    "pkt_len_std",
    "iat_min",
    "iat_max",
    "iat_mean",
    "iat_std",
    "syn_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "ack_count",
    "pkt_syn_ratio",      # was: syn_ratio (conflicts with HTTP)
    "fin_ratio",
    "rst_ratio",
    "avg_header_len",
    "header_len_std",
    "avg_pkts_per_burst",
    "idle_time",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FlowKey:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: int = 6


@dataclass
class FlowStats:
    key: FlowKey
    start_ts: float
    end_ts: float
    fwd_pkts: list[tuple[float, int, int]] = field(default_factory=list)
    bwd_pkts: list[tuple[float, int, int]] = field(default_factory=list)
    flags: Counter = field(default_factory=Counter)

    def to_features(self) -> "PacketFeatures":
        duration = max(self.end_ts - self.start_ts, 0.001)
        fwd_count = len(self.fwd_pkts)
        bwd_count = len(self.bwd_pkts)
        total = fwd_count + bwd_count
        fwd_bytes = sum(p[1] for p in self.fwd_pkts)
        bwd_bytes = sum(p[1] for p in self.bwd_pkts)
        total_bytes = fwd_bytes + bwd_bytes

        all_lens = [p[1] for p in self.fwd_pkts + self.bwd_pkts]
        pkt_len_min = min(all_lens) if all_lens else 0
        pkt_len_max = max(all_lens) if all_lens else 0
        pkt_len_mean = statistics.mean(all_lens) if all_lens else 0
        pkt_len_std = statistics.stdev(all_lens) if len(all_lens) > 1 else 0

        all_ts = sorted([p[0] for p in self.fwd_pkts + self.bwd_pkts])
        iats = [all_ts[i+1] - all_ts[i] for i in range(len(all_ts) - 1)]
        iat_min = min(iats) if iats else 0
        iat_max = max(iats) if iats else 0
        iat_mean = statistics.mean(iats) if iats else 0
        iat_std = statistics.stdev(iats) if len(iats) > 1 else 0

        headers = [p[2] for p in self.fwd_pkts + self.bwd_pkts]
        avg_header_len = statistics.mean(headers) if headers else 0
        header_len_std = statistics.stdev(headers) if len(headers) > 1 else 0

        syn = self.flags.get("S", 0)
        fin = self.flags.get("F", 0)
        rst = self.flags.get("R", 0)
        psh = self.flags.get("P", 0)
        ack = self.flags.get("A", 0)

        return PacketFeatures(
            src_ip=self.key.src_ip,
            dst_ip=self.key.dst_ip,
            src_port=self.key.src_port,
            dst_port=self.key.dst_port,
            flow_duration=duration,
            total_pkts=total,
            total_bytes=total_bytes,
            pkt_pkt_rate=total / duration,
            pkt_byte_rate=total_bytes / duration,
            fwd_pkt_count=fwd_count,
            bwd_pkt_count=bwd_count,
            fwd_byte_count=fwd_bytes,
            bwd_byte_count=bwd_bytes,
            fwd_bwd_ratio=fwd_count / max(bwd_count, 1),
            pkt_len_min=float(pkt_len_min),
            pkt_len_max=float(pkt_len_max),
            pkt_len_mean=float(pkt_len_mean),
            pkt_len_std=float(pkt_len_std),
            iat_min=float(iat_min),
            iat_max=float(iat_max),
            iat_mean=float(iat_mean),
            iat_std=float(iat_std),
            syn_count=float(syn),
            fin_count=float(fin),
            rst_count=float(rst),
            psh_count=float(psh),
            ack_count=float(ack),
            pkt_syn_ratio=syn / max(total, 1),
            fin_ratio=fin / max(total, 1),
            rst_ratio=rst / max(total, 1),
            avg_header_len=float(avg_header_len),
            header_len_std=float(header_len_std),
            avg_pkts_per_burst=total / max(1, len(iats) // 10 + 1),
            idle_time=duration - sum(iats),
        )


@dataclass
class PacketFeatures:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int

    flow_duration: float
    total_pkts: float
    total_bytes: float
    pkt_pkt_rate: float
    pkt_byte_rate: float

    fwd_pkt_count: float
    bwd_pkt_count: float
    fwd_byte_count: float
    bwd_byte_count: float
    fwd_bwd_ratio: float

    pkt_len_min: float
    pkt_len_max: float
    pkt_len_mean: float
    pkt_len_std: float

    iat_min: float
    iat_max: float
    iat_mean: float
    iat_std: float

    syn_count: float
    fin_count: float
    rst_count: float
    psh_count: float
    ack_count: float
    pkt_syn_ratio: float
    fin_ratio: float
    rst_ratio: float

    avg_header_len: float
    header_len_std: float

    avg_pkts_per_burst: float
    idle_time: float

    def to_vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in PACKET_FEATURE_NAMES]


# ---------------------------------------------------------------------------
# pcap parser (scapy)
# ---------------------------------------------------------------------------

def parse_pcap(pcap_path: str | Path) -> dict[str, PacketFeatures]:
    """Parse a pcap file and return per-flow features (keyed by flow tuple)."""
    try:
        from scapy.all import IP, TCP, rdpcap
    except ImportError:
        log.error("scapy not installed: pip install scapy")
        return {}

    path = Path(pcap_path)
    if not path.exists():
        log.warning(f"pcap not found: {path}")
        return {}

    log.info(f"parsing pcap: {path}")

    try:
        packets = rdpcap(str(path))
    except Exception as e:
        log.error(f"failed to read pcap: {e}")
        return {}

    flows: dict[tuple, FlowStats] = {}

    for pkt in packets:
        if not pkt.haslayer(IP) or not pkt.haslayer(TCP):
            continue

        ip = pkt[IP]
        tcp = pkt[TCP]
        ts = float(pkt.time)

        endpoints = tuple(sorted([(ip.src, tcp.sport), (ip.dst, tcp.dport)]))
        (s_ip, s_port), (d_ip, d_port) = endpoints
        key = (s_ip, s_port, d_ip, d_port)

        if key not in flows:
            flows[key] = FlowStats(
                key=FlowKey(s_ip, d_ip, s_port, d_port),
                start_ts=ts,
                end_ts=ts,
            )
        f = flows[key]
        f.end_ts = max(f.end_ts, ts)

        is_fwd = (ip.src == f.key.src_ip)
        total_len = len(pkt)
        header_len = (ip.ihl or 0) * 4 + (tcp.dataofs or 0) * 4

        if is_fwd:
            f.fwd_pkts.append((ts, total_len, header_len))
        else:
            f.bwd_pkts.append((ts, total_len, header_len))

        flags = tcp.flags
        if flags & 0x02: f.flags["S"] += 1
        if flags & 0x01: f.flags["F"] += 1
        if flags & 0x04: f.flags["R"] += 1
        if flags & 0x08: f.flags["P"] += 1
        if flags & 0x10: f.flags["A"] += 1

    log.info(f"  parsed {len(flows)} flows from {len(packets)} packets")

    return {k: f.to_features() for k, f in flows.items()}


def features_from_pcap(pcap_path: str | Path) -> list[PacketFeatures]:
    return list(parse_pcap(pcap_path).values())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Extract packet features")
    p.add_argument("--pcap", required=True)
    p.add_argument("--top", type=int, default=3)
    args = p.parse_args()

    features = features_from_pcap(args.pcap)
    print(f"Extracted features from {len(features)} flows")
    print()

    for f in features[:args.top]:
        print(f"Flow: {f.src_ip}:{f.src_port} → {f.dst_ip}:{f.dst_port}")
        print(f"  duration: {f.flow_duration:.2f}s")
        print(f"  packets:  {int(f.total_pkts)} ({int(f.fwd_pkt_count)} fwd / {int(f.bwd_pkt_count)} bwd)")
        print(f"  bytes:    {int(f.total_bytes)}")
        print(f"  pkt_rate: {f.pkt_pkt_rate:.1f}/s")
        print(f"  flags:    SYN={int(f.syn_count)} ACK={int(f.ack_count)} FIN={int(f.fin_count)}")
        print()