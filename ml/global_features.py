"""Global (cross-IP) feature extraction.

Computes features describing the ENTIRE window, not individual IPs.
Used to detect distributed attacks that rotate source IPs.

Features:
    - total_requests
    - unique_ips
    - requests_per_ip           (low = rotation)
    - ip_churn_rate             (new IPs per second)
    - ip_subnet_entropy         (low = concentrated)
    - path_entropy              (low = coordinated)
    - ua_entropy                (low = botnet)
    - time_entropy              (low = synchronized bots)
    - error_rate                (high = fuzzing/scanning)
    - mean_inter_arrival_cv     (low = machine timing)

Usage:
    from ml.global_features import GlobalFeatureExtractor

    ex = GlobalFeatureExtractor()
    features = ex.extract(records)
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from collector.nginx_parser import ParsedLog
from core.logging import get_logger

log = get_logger("ml.global_features")


# ---------------------------------------------------------------------------
# Feature name registry (order matters for ML)
# ---------------------------------------------------------------------------

GLOBAL_FEATURE_NAMES: list[str] = [
    "total_requests",
    "unique_ips",
    "requests_per_ip",
    "ip_churn_rate",
    "ip_subnet_entropy",
    "path_entropy",
    "ua_entropy",
    "time_entropy",
    "error_rate",
    "mean_inter_arrival_cv",
]


# ---------------------------------------------------------------------------
# Feature vector dataclass
# ---------------------------------------------------------------------------

@dataclass
class GlobalFeatures:
    window_start: float
    window_end: float
    window_seconds: float
    total_requests: int
    unique_ips: int
    requests_per_ip: float
    ip_churn_rate: float
    ip_subnet_entropy: float
    path_entropy: float
    ua_entropy: float
    time_entropy: float
    error_rate: float
    mean_inter_arrival_cv: float

    def to_vector(self) -> list[float]:
        return [
            float(self.total_requests),
            float(self.unique_ips),
            float(self.requests_per_ip),
            float(self.ip_churn_rate),
            float(self.ip_subnet_entropy),
            float(self.path_entropy),
            float(self.ua_entropy),
            float(self.time_entropy),
            float(self.error_rate),
            float(self.mean_inter_arrival_cv),
        ]


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------

def _entropy(counter: Counter) -> float:
    """Shannon entropy of a counter (bits)."""
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
    """Normalized entropy in [0, 1]. 1 = perfectly uniform, 0 = all same."""
    if len(counter) <= 1:
        return 0.0
    max_h = math.log2(len(counter))
    return _entropy(counter) / max_h if max_h > 0 else 0.0


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class GlobalFeatureExtractor:
    """Extract global (cross-IP) features from a batch of records."""

    def __init__(self) -> None:
        # Track seen IPs to compute churn rate over time
        self._seen_ips: set[str] = set()
        self._last_reset: float = time.time()

        log.info("GlobalFeatureExtractor ready")

    def extract(self, records: list[ParsedLog]) -> GlobalFeatures | None:
        """Compute global features for one window of records."""
        if not records:
            return None

        now = time.time()
        t_min = min(r.ts for r in records)
        t_max = max(r.ts for r in records)
        window_seconds = max(t_max - t_min, 1.0)

        total = len(records)

        # Group by IP, path, UA
        by_ip: Counter[str] = Counter()
        by_path: Counter[str] = Counter()
        by_ua: Counter[str] = Counter()
        by_subnet: Counter[str] = Counter()
        by_second: Counter[int] = Counter()

        ips_in_order: list[str] = []

        for r in records:
            by_ip[r.ip] += 1
            by_path[r.path] += 1
            by_ua[r.user_agent or "unknown"] += 1
            by_subnet[_subnet_of(r.ip)] += 1
            by_second[int(r.ts)] += 1
            ips_in_order.append(r.ip)

        unique_ips = len(by_ip)
        requests_per_ip = total / max(unique_ips, 1)

        # IP churn rate: new IPs since last window
        new_ips = set(by_ip.keys()) - self._seen_ips
        elapsed = max(now - self._last_reset, 1.0)
        ip_churn_rate = len(new_ips) / elapsed
        self._seen_ips.update(by_ip.keys())

        # Entropies
        subnet_entropy = _normalized_entropy(by_subnet)
        path_entropy = _normalized_entropy(by_path)
        ua_entropy = _normalized_entropy(by_ua)
        time_entropy = _normalized_entropy(by_second)

        # Error rate
        errors = sum(1 for r in records if r.status >= 400)
        error_rate = errors / total

        # Inter-arrival CV
        # Sort records by time, compute gaps between consecutive requests
        sorted_ts = sorted(r.ts for r in records)
        if len(sorted_ts) >= 3:
            gaps = [sorted_ts[i+1] - sorted_ts[i]
                    for i in range(len(sorted_ts) - 1)]
            mean_gap = sum(gaps) / len(gaps)
            if mean_gap > 0:
                var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
                std = math.sqrt(var)
                cv = std / mean_gap
            else:
                cv = 0.0
        else:
            cv = 0.0

        return GlobalFeatures(
            window_start=t_min,
            window_end=t_max,
            window_seconds=window_seconds,
            total_requests=total,
            unique_ips=unique_ips,
            requests_per_ip=requests_per_ip,
            ip_churn_rate=ip_churn_rate,
            ip_subnet_entropy=subnet_entropy,
            path_entropy=path_entropy,
            ua_entropy=ua_entropy,
            time_entropy=time_entropy,
            error_rate=error_rate,
            mean_inter_arrival_cv=cv,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear seen IPs."""
        self._seen_ips.clear()
        self._last_reset = time.time()


# ---------------------------------------------------------------------------
# Subnet helper
# ---------------------------------------------------------------------------

def _subnet_of(ip: str, prefix: int = 24) -> str:
    """Return the /24 subnet for an IPv4 address."""
    try:
        parts = ip.split(".")
        if len(parts) != 4:
            return ip
        return ".".join(parts[:3]) + ".0/24"
    except Exception:
        return ip


# ---------------------------------------------------------------------------
# Heuristic signals from global features
# ---------------------------------------------------------------------------

def detect_anomalies(f: GlobalFeatures) -> list[str]:
    """Return list of suspicious signals in global features."""
    reasons: list[str] = []

    # Rotation: many IPs, few requests each
    if f.unique_ips >= 20 and f.requests_per_ip < 2.0:
        reasons.append(
            f"IP rotation: {f.unique_ips} ips, "
            f"{f.requests_per_ip:.2f} req/ip"
        )

    # Churn: too many new IPs per second
    if f.ip_churn_rate > 10.0:
        reasons.append(f"high churn: {f.ip_churn_rate:.1f} new ips/s")

    # Subnet concentration
    if f.ip_subnet_entropy < 0.3 and f.unique_ips >= 10:
        reasons.append(
            f"subnet concentration: entropy={f.ip_subnet_entropy:.2f}"
        )

    # Path concentration
    if f.path_entropy < 0.4 and f.total_requests >= 20:
        reasons.append(f"path concentration: entropy={f.path_entropy:.2f}")

    # UA concentration (bots often use same UA)
    if f.ua_entropy < 0.4 and f.unique_ips >= 10:
        reasons.append(f"UA concentration: entropy={f.ua_entropy:.2f}")

    # Timing regularity (machine-driven)
    if f.mean_inter_arrival_cv < 0.2 and f.total_requests >= 20:
        reasons.append(
            f"regular timing: cv={f.mean_inter_arrival_cv:.2f}"
        )

    # High error rate (scanning/fuzzing)
    if f.error_rate > 0.5:
        reasons.append(f"high error rate: {f.error_rate:.2f}")

    return reasons


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from collector.nginx_parser import parse_file

    p = argparse.ArgumentParser(description="Global features test")
    p.add_argument("--file", default="data/raw/training.log")
    args = p.parse_args()

    records = parse_file(args.file)
    print(f"loaded {len(records)} records from {args.file}")
    print()

    ex = GlobalFeatureExtractor()
    f = ex.extract(records)

    if f is None:
        print("no features extracted")
        raise SystemExit(0)

    print("=== Global Features ===")
    for name, val in zip(GLOBAL_FEATURE_NAMES, f.to_vector()):
        print(f"  {name:25s} = {val:.3f}")

    print()
    reasons = detect_anomalies(f)
    if reasons:
        print("=== Anomaly Signals ===")
        for r in reasons:
            print(f"  ⚠ {r}")
    else:
        print("no anomaly signals detected")