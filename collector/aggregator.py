"""Aggregate nginx log records into per-IP feature vectors.

Produces 12 features per IP:
    - 9 base features (rate, bytes, entropy, etc.)
    - ua_entropy          (botnets use one UA)
    - path_concentration  (botnets hit one path)
    - time_regularity     (bots fire at regular intervals)
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from collector.nginx_parser import ParsedLog
from core.logging import get_logger
from core.schema import FEATURE_NAMES

log = get_logger("collector.aggregator")


# ---------------------------------------------------------------------------
# IPFeatures
# ---------------------------------------------------------------------------

@dataclass
class IPFeatures:
    ip: str
    window_start: float
    window_end: float

    # 9 base features
    pkt_rate: float
    byte_rate: float
    syn_ratio: float
    avg_pkt_size: float
    unique_dports: float
    conn_duration: float
    http_reqs: float
    error_ratio: float
    src_ip_entropy: float

    # 3 new features
    ua_entropy: float
    path_concentration: float
    time_regularity: float

    def to_vector(self) -> list[float]:
        d = asdict(self)
        return [float(d[name]) for name in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def _normalized_entropy(counter: Counter) -> float:
    """Normalized entropy (0..1). 1 = uniform, 0 = one value only."""
    if len(counter) <= 1:
        return 0.0
    max_h = math.log2(len(counter))
    return _entropy(counter) / max_h if max_h > 0 else 0.0


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def aggregate(
    records: list[ParsedLog],
    window_start: float | None = None,
    window_end: float | None = None,
) -> dict[str, IPFeatures]:
    """Group records by source IP and compute one feature vector per IP."""
    if not records:
        return {}

    if window_start is None:
        window_start = min(r.ts for r in records)
    if window_end is None:
        window_end = max(r.ts for r in records)

    window = max(window_end - window_start, 0.001)

    # Group by IP
    by_ip: dict[str, list[ParsedLog]] = defaultdict(list)
    for r in records:
        by_ip[r.ip].append(r)

    out: dict[str, IPFeatures] = {}

    for ip, recs in by_ip.items():
        n = len(recs)
        total_bytes = sum(r.bytes for r in recs)
        errors = sum(1 for r in recs if r.is_error)

        # Path distribution
        path_counter: Counter[str] = Counter()
        for r in recs:
            path_counter[r.path] += 1

        # UA distribution
        ua_counter: Counter[str] = Counter()
        for r in recs:
            ua_counter[r.user_agent or "unknown"] += 1

        # Time deltas
        sorted_ts = sorted(r.ts for r in recs)
        if len(sorted_ts) >= 3:
            gaps = [sorted_ts[i + 1] - sorted_ts[i]
                    for i in range(len(sorted_ts) - 1)]
            mean_gap = sum(gaps) / len(gaps)
            if mean_gap > 0:
                var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
                std = math.sqrt(var)
                cv = std / mean_gap
                time_regularity = 1.0 / (1.0 + cv)     # 1.0 = perfectly periodic
            else:
                time_regularity = 0.0
        else:
            time_regularity = 0.0

        # Path concentration: 1 - normalized entropy
        # 1.0 = all requests to one path (botnet signature)
        # 0.0 = requests spread across many paths (normal user)
        path_entropy = _normalized_entropy(path_counter)
        path_concentration = 1.0 - path_entropy

        # UA entropy: 0.0 = one UA (botnet), 1.0 = many UAs
        ua_entropy = _normalized_entropy(ua_counter)

        # src_ip_entropy: same as path entropy (kept for backward compat)
        src_ip_entropy = path_entropy

        first_ts = min(r.ts for r in recs)
        last_ts = max(r.ts for r in recs)
        conn_duration = max(last_ts - first_ts, 0.0)

        feats = IPFeatures(
            ip=ip,
            window_start=window_start,
            window_end=window_end,
            pkt_rate=n / window,
            byte_rate=total_bytes / window,
            syn_ratio=0.0,
            avg_pkt_size=total_bytes / n if n else 0.0,
            unique_dports=float(len(path_counter)),
            conn_duration=conn_duration,
            http_reqs=float(n),
            error_ratio=errors / n if n else 0.0,
            src_ip_entropy=src_ip_entropy,
            ua_entropy=ua_entropy,
            path_concentration=path_concentration,
            time_regularity=time_regularity,
        )
        out[ip] = feats

    return out


def features_to_rows(features: dict[str, IPFeatures]) -> list[dict]:
    rows = []
    for ip, f in features.items():
        d = asdict(f)
        d["vector"] = f.to_vector()
        rows.append(d)
    return rows