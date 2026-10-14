"""Advanced feature engineering.

Extends the base 9 features from core.schema with temporal, ratio,
and behavioral features that capture attacker dynamics over time.

Usage:
    from ml.feature_engineering import FeatureEngineer

    fe = FeatureEngineer()
    fe.update(ip_features)                      # feed new window
    enhanced = fe.compute(ip)                    # get enriched vector
"""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from collector.aggregator import IPFeatures
from core.logging import get_logger

log = get_logger("ml.feature_engineering")


# Names of the temporal features we add on top of core.schema.FEATURE_NAMES
TEMPORAL_FEATURE_NAMES: list[str] = [
    "pkt_rate_delta",         # change from previous window
    "pkt_rate_accel",         # 2nd derivative — is it accelerating?
    "pkt_rate_std",           # std-dev across recent windows
    "byte_per_request",       # bytes / requests
    "error_burst_score",      # recent error spikes
    "request_interval_cv",    # coefficient of variation of intervals
    "path_concentration",     # 1 - normalized entropy (0=spread, 1=concentrated)
    "new_ip_score",           # 1 if never seen before, decays with sightings
    "age_seconds",            # how long we've been tracking this IP
    "time_since_last_seen",   # seconds since previous window
]


@dataclass
class IPHistory:
    """Track per-IP features over a sliding window of recent samples."""
    ip: str
    samples: deque  # deque of (ts, IPFeatures)
    first_seen: float
    last_seen: float
    sightings: int

    def __post_init__(self):
        if self.samples is None:
            self.samples = deque(maxlen=10)
        if self.first_seen == 0:
            self.first_seen = time.time()
        if self.last_seen == 0:
            self.last_seen = time.time()
        if self.sightings == 0:
            self.sightings = 1


class FeatureEngineer:
    """Maintains per-IP history and computes temporal features."""

    def __init__(self, window: int = 10, decay: float = 0.9) -> None:
        self.window = window
        self.decay = decay
        self.history: dict[str, IPHistory] = {}
        self.sightings: defaultdict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Feed new observations
    # ------------------------------------------------------------------
    def update(self, features: dict[str, IPFeatures]) -> None:
        """Feed one aggregation window's worth of per-IP features."""
        now = time.time()
        for ip, f in features.items():
            if ip not in self.history:
                self.history[ip] = IPHistory(
                    ip=ip,
                    samples=deque(maxlen=self.window),
                    first_seen=now,
                    last_seen=now,
                    sightings=1,
                )
            h = self.history[ip]
            h.samples.append((now, f))
            h.last_seen = now
            h.sightings += 1
            self.sightings[ip] += 1

    # ------------------------------------------------------------------
    # Compute enriched feature vector
    # ------------------------------------------------------------------
    def compute(self, ip: str) -> np.ndarray | None:
        """Return the enriched feature vector for an IP, or None if unknown."""
        h = self.history.get(ip)
        if h is None or not h.samples:
            return None

        base = h.samples[-1][1]              # latest IPFeatures
        base_vector = np.asarray(base.to_vector(), dtype=float)

        # Temporal features
        pkt_rates = np.asarray([f.pkt_rate for _, f in h.samples], dtype=float)
        byte_rates = np.asarray([f.byte_rate for _, f in h.samples], dtype=float)
        errors = np.asarray([f.error_ratio for _, f in h.samples], dtype=float)
        timestamps = np.asarray([ts for ts, _ in h.samples], dtype=float)

        # 1. pkt_rate_delta — change from previous window
        if len(pkt_rates) >= 2:
            delta = pkt_rates[-1] - pkt_rates[-2]
        else:
            delta = 0.0

        # 2. pkt_rate_accel — 2nd derivative
        if len(pkt_rates) >= 3:
            accel = pkt_rates[-1] - 2 * pkt_rates[-2] + pkt_rates[-3]
        else:
            accel = 0.0

        # 3. pkt_rate_std — variance across window
        pkt_rate_std = float(pkt_rates.std()) if len(pkt_rates) > 1 else 0.0

        # 4. byte_per_request
        byte_per_request = (
            base.byte_rate / base.pkt_rate if base.pkt_rate > 0 else 0.0
        )

        # 5. error_burst_score — recent errors weighted higher
        weights = np.power(self.decay, np.arange(len(errors) - 1, -1, -1))
        error_burst = float(np.dot(errors, weights) / weights.sum()) if len(errors) else 0.0

        # 6. request_interval_cv — coefficient of variation
        if len(timestamps) >= 2:
            intervals = np.diff(timestamps)
            mean_int = intervals.mean()
            cv = float(intervals.std() / mean_int) if mean_int > 0 else 0.0
        else:
            cv = 0.0

        # 7. path_concentration — 1 - normalized entropy
        max_entropy = math.log2(max(base.unique_dports, 1))
        path_conc = (
            1.0 - base.src_ip_entropy / max_entropy if max_entropy > 0 else 0.0
        )

        # 8. new_ip_score — decays with sightings
        new_ip_score = 1.0 / (1.0 + math.log1p(self.sightings[ip]))

        # 9. age_seconds
        age = time.time() - h.first_seen

        # 10. time_since_last_seen
        since_last = time.time() - h.last_seen

        temporal = np.asarray([
            delta,
            accel,
            pkt_rate_std,
            byte_per_request,
            error_burst,
            cv,
            path_conc,
            new_ip_score,
            age,
            since_last,
        ], dtype=float)

        # Replace zeros/infs
        temporal = np.nan_to_num(temporal, nan=0.0, posinf=0.0, neginf=0.0)

        return np.concatenate([base_vector, temporal])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def all_feature_names(self) -> list[str]:
        from core.schema import FEATURE_NAMES
        return list(FEATURE_NAMES) + TEMPORAL_FEATURE_NAMES

    def forget(self, ip: str) -> None:
        """Stop tracking an IP (e.g., after it's been blocked)."""
        self.history.pop(ip, None)
        self.sightings.pop(ip, None)

    def known_ips(self) -> list[str]:
        return list(self.history.keys())