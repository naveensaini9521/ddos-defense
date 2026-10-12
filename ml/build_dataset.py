"""Build ML training dataset from collected log records + pcaps.

Reads data/collected/*.jsonl (HTTP records with labels) and the sibling
*.pcap (packet capture), and produces a combined feature CSV.

Features:
    - 12 HTTP features (from aggregator)
    - 30 packet features (from packet_features)
    - Total: 42 features per row

Usage:
    python3 -m ml.build_dataset --source data/collected --out data/labeled/features_v7.csv --window 2
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from collector.aggregator import aggregate
from collector.nginx_parser import ParsedLog
from collector.packet_features import (
    PACKET_FEATURE_NAMES,
    PacketFeatures,
    parse_pcap,
)
from core.logging import get_logger
from core.schema import FEATURE_NAMES as HTTP_FEATURE_NAMES

log = get_logger("ml.build_dataset")

WINDOW_SECONDS = 10.0

# Combined feature order
COMBINED_FEATURE_NAMES = list(HTTP_FEATURE_NAMES) + list(PACKET_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Convert record → ParsedLog
# ---------------------------------------------------------------------------

def record_to_parsed(rec: dict) -> ParsedLog:
    return ParsedLog(
        ts=float(rec["ts"]),
        ip=rec["ip"],
        method=rec.get("method", "GET"),
        path=rec.get("path", "/"),
        proto="HTTP/1.1",
        status=int(rec.get("status", 200)),
        bytes=int(rec.get("bytes", 0)),
        referer=None,
        user_agent=rec.get("user_agent"),
        raw="",
    )


# ---------------------------------------------------------------------------
# Window splitting
# ---------------------------------------------------------------------------

def split_into_windows(records: list[ParsedLog],
                       window_seconds: float = WINDOW_SECONDS
                       ) -> list[list[ParsedLog]]:
    if not records:
        return []
    records = sorted(records, key=lambda r: r.ts)
    t_start = records[0].ts
    t_end = records[-1].ts
    windows: list[list[ParsedLog]] = []
    cur_start = t_start
    while cur_start < t_end:
        cur_end = cur_start + window_seconds
        window = [r for r in records if cur_start <= r.ts < cur_end]
        if window:
            windows.append(window)
        cur_start = cur_end
    return windows


# ---------------------------------------------------------------------------
# Packet flow aggregation by IP
# ---------------------------------------------------------------------------

def aggregate_packets_by_ip(
    flows: list[PacketFeatures],
) -> dict[str, PacketFeatures]:
    """Merge multiple flows per source IP into one feature vector.

    For each source IP, we combine its flows and compute one
    representative PacketFeatures vector.
    """
    by_ip: dict[str, list[PacketFeatures]] = {}
    for f in flows:
        by_ip.setdefault(f.src_ip, []).append(f)

    merged: dict[str, PacketFeatures] = {}

    for ip, ip_flows in by_ip.items():
        # Sum volume metrics
        total_pkts = sum(f.total_pkts for f in ip_flows)
        total_bytes = sum(f.total_bytes for f in ip_flows)
        fwd_pkts = sum(f.fwd_pkt_count for f in ip_flows)
        bwd_pkts = sum(f.bwd_pkt_count for f in ip_flows)
        fwd_bytes = sum(f.fwd_byte_count for f in ip_flows)
        bwd_bytes = sum(f.bwd_byte_count for f in ip_flows)

        # Aggregate duration
        flow_duration = max(f.flow_duration for f in ip_flows)

        # Average continuous metrics
        n = len(ip_flows)
        pkt_len_min = min(f.pkt_len_min for f in ip_flows)
        pkt_len_max = max(f.pkt_len_max for f in ip_flows)
        pkt_len_mean = sum(f.pkt_len_mean for f in ip_flows) / n
        pkt_len_std = sum(f.pkt_len_std for f in ip_flows) / n

        iat_min = min(f.iat_min for f in ip_flows if f.iat_min > 0) if any(f.iat_min > 0 for f in ip_flows) else 0.0
        iat_max = max(f.iat_max for f in ip_flows)
        iat_mean = sum(f.iat_mean for f in ip_flows) / n
        iat_std = sum(f.iat_std for f in ip_flows) / n

        # Sum flags
        syn = sum(f.syn_count for f in ip_flows)
        fin = sum(f.fin_count for f in ip_flows)
        rst = sum(f.rst_count for f in ip_flows)
        psh = sum(f.psh_count for f in ip_flows)
        ack = sum(f.ack_count for f in ip_flows)

        # Header stats
        avg_header_len = sum(f.avg_header_len for f in ip_flows) / n
        header_len_std = sum(f.header_len_std for f in ip_flows) / n

        # Burst and idle
        avg_pkts_per_burst = sum(f.avg_pkts_per_burst for f in ip_flows) / n
        idle_time = sum(f.idle_time for f in ip_flows)

        # Rates
        pkt_rate = total_pkts / max(flow_duration, 0.001)
        byte_rate = total_bytes / max(flow_duration, 0.001)

        merged[ip] = PacketFeatures(
            src_ip=ip,
            dst_ip=ip_flows[0].dst_ip,
            src_port=0,
            dst_port=0,
            flow_duration=flow_duration,
            total_pkts=total_pkts,
            total_bytes=total_bytes,
            pkt_rate=pkt_rate,
            byte_rate=byte_rate,
            fwd_pkt_count=fwd_pkts,
            bwd_pkt_count=bwd_pkts,
            fwd_byte_count=fwd_bytes,
            bwd_byte_count=bwd_bytes,
            fwd_bwd_ratio=fwd_pkts / max(bwd_pkts, 1),
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
            syn_ratio=syn / max(total_pkts, 1),
            fin_ratio=fin / max(total_pkts, 1),
            rst_ratio=rst / max(total_pkts, 1),
            avg_header_len=float(avg_header_len),
            header_len_std=float(header_len_std),
            avg_pkts_per_burst=float(avg_pkts_per_burst),
            idle_time=float(idle_time),
        )

    return merged


# ---------------------------------------------------------------------------
# Process one collected scenario (jsonl + pcap)
# ---------------------------------------------------------------------------

def process_scenario(jsonl_path: Path,
                     window_seconds: float = WINDOW_SECONDS) -> list[dict]:
    """Process one .jsonl + sibling .pcap → combined feature rows."""
    log.info(f"processing {jsonl_path.name}")

    # 1. Load HTTP records
    with jsonl_path.open() as f:
        raw = [json.loads(line) for line in f if line.strip()]

    if not raw:
        log.warning(f"  empty: {jsonl_path.name}")
        return []

    scenario = raw[0].get("scenario", "unknown")

    # 2. Load packet flows from sibling .pcap
    pcap_path = jsonl_path.with_suffix(".pcap")
    packet_by_ip: dict[str, PacketFeatures] = {}
    if pcap_path.exists():
        try:
            flows = list(parse_pcap(pcap_path).values())
            packet_by_ip = aggregate_packets_by_ip(flows)
            log.info(f"  loaded {len(flows)} packet flows "
                     f"({len(packet_by_ip)} unique src IPs)")
        except Exception as e:
            log.warning(f"  pcap failed: {e}")
    else:
        log.warning(f"  no pcap for {jsonl_path.name}")

    # 3. Label lookup by (ip, rounded_ts)
    label_lookup: dict[tuple[str, int], int] = {}
    for r in raw:
        key = (r["ip"], int(round(r["ts"])))
        label_lookup[key] = max(label_lookup.get(key, 0), int(r.get("label", 0)))

    # 4. Parse HTTP records
    parsed = [record_to_parsed(r) for r in raw]

    # 5. Split into windows
    windows = split_into_windows(parsed, window_seconds)
    log.info(f"  {len(parsed)} records → {len(windows)} windows")

    rows: list[dict] = []

    # 6. Per window, aggregate HTTP + lookup packet features
    for widx, window in enumerate(windows):
        http_feats = aggregate(window)  # dict[ip, IPFeatures]

        for ip, hf in http_feats.items():
            # Skip IPs that had no HTTP requests
            ip_labels: list[int] = []
            for r in window:
                if r.ip != ip:
                    continue
                key = (r.ip, int(round(r.ts)))
                if key in label_lookup:
                    ip_labels.append(label_lookup[key])

            if not ip_labels:
                continue

            label = 1 if (sum(ip_labels) / len(ip_labels)) > 0.5 else 0

            # Build row: metadata + HTTP features + packet features + label
            row = {
                "window_index": widx,
                "scenario": scenario,
                "ip": ip,
            }

            # HTTP features
            for name in HTTP_FEATURE_NAMES:
                row[name] = float(getattr(hf, name, 0.0))

            # Packet features (from pcap, keyed by src IP)
            pf = packet_by_ip.get(ip)
            for name in PACKET_FEATURE_NAMES:
                row[name] = float(getattr(pf, name, 0.0)) if pf else 0.0

            row["label"] = label
            rows.append(row)

    log.info(f"  produced {len(rows)} rows")
    return rows


# ---------------------------------------------------------------------------
# Save dataset
# ---------------------------------------------------------------------------

def save_dataset(rows: list[dict], out_path: Path) -> None:
    if not rows:
        log.warning("no rows")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "window_index", "scenario", "ip",
        *COMBINED_FEATURE_NAMES,
        "label",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    log.info(f"saved {len(rows)} rows → {out_path}")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_dataset(
    source_dir: str | Path = "data/collected",
    out_path: str | Path = "data/labeled/features.csv",
    window_seconds: float = WINDOW_SECONDS,
) -> Path:
    source = Path(source_dir)
    out = Path(out_path)

    jsonl_files = sorted(source.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"no .jsonl files in {source}")

    log.info(f"found {len(jsonl_files)} collected files")

    all_rows: list[dict] = []
    for jf in jsonl_files:
        try:
            all_rows.extend(process_scenario(jf, window_seconds))
        except Exception as e:
            log.error(f"failed: {jf.name}: {e}")

    if not all_rows:
        raise RuntimeError("no rows produced")

    n_attack = sum(1 for r in all_rows if r["label"] == 1)
    n_normal = len(all_rows) - n_attack
    log.info(f"total rows: {len(all_rows)}  "
             f"attack: {n_attack}  normal: {n_normal}")

    save_dataset(all_rows, out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build combined training dataset")
    p.add_argument("--source", default="data/collected")
    p.add_argument("--out", default="data/labeled/features.csv")
    p.add_argument("--window", type=float, default=WINDOW_SECONDS)
    args = p.parse_args()

    out = build_dataset(args.source, args.out, args.window)

    print()
    print(f"✅ Dataset built: {out}")

    with open(out) as f:
        rows = list(csv.DictReader(f))
    n_attack = sum(1 for r in rows if int(r["label"]) == 1)
    n_normal = len(rows) - n_attack
    n_features = len(rows[0]) - 4 if rows else 0    # minus meta+label

    print(f"   rows:     {len(rows)}")
    print(f"   attack:   {n_attack}")
    print(f"   normal:   {n_normal}")
    print(f"   features: {n_features}")