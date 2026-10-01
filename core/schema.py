"""
Single source of truth for feature names, constants, and shared types.
Every module imports from here to prevent schema drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --- Feature schema (order matters — used as ML input vector) ---
FEATURE_NAMES: list[str] = [
    "pkt_rate",
    "byte_rate",
    "syn_ratio",
    "avg_pkt_size",
    "unique_dports",
    "conn_duration",
    "http_reqs",
    "error_ratio",
    "src_ip_entropy",
]


@dataclass(frozen=True)
class Decision:
    ip: str
    action: Literal["allow", "block", "throttle"]
    confidence: float
    reason: str
    ttl_seconds: int = 60


@dataclass
class BlockRecord:
    ip: str
    since: float
    until: float
    reason: str
    strike: int = 1


WINDOW_SECONDS: int = 10
DEFAULT_TTL: int = 60
ATTACK_THRESHOLD: float = 0.7
