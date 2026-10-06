"""Subnet blocker — decides when to block entire subnets.

Combines global features + per-subnet IP counts to emit
Decision(ip="10.0.0.0/24", action="block", ...) when a subnet
attack is detected.

Works alongside EnsembleDetector: ensemble handles per-IP,
this handles per-subnet.

Usage:
    from blocker.subnet_blocker import SubnetBlocker, SubnetConfig

    sb = SubnetBlocker()
    decisions = sb.check(records)
    for d in decisions:
        block_manager.enforce(d)
"""
from __future__ import annotations

import ipaddress
from collections import Counter
from dataclasses import dataclass, field

from collector.nginx_parser import ParsedLog
from core.logging import get_logger
from core.schema import Decision
from ml.global_features import (
    GlobalFeatureExtractor,
    GlobalFeatures,
    detect_anomalies,
)

log = get_logger("blocker.subnet_blocker")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SubnetConfig:
    """Tunables for subnet blocking."""

    # Subnet prefix to block
    prefix: int = 24

    # Minimum IPs from a subnet before we consider blocking it
    min_ips_in_subnet: int = 8

    # Minimum fraction of unique IPs from the dominant subnet
    min_subnet_share: float = 0.5

    # Minimum global anomaly signals required
    min_global_signals: int = 2

    # Block parameters
    block_duration: float = 300.0    # 5 min
    confidence: float = 0.85

    # Safety: never block these ranges
    never_block: tuple[str, ...] = (
        "0.0.0.0/8",        # current network
        "10.0.0.0/8",       # private
        "127.0.0.0/8",      # loopback
        "169.254.0.0/16",   # link-local
        "172.16.0.0/12",    # private
        "192.0.0.0/24",     # IETF
        "192.168.0.0/16",   # private
        "224.0.0.0/4",      # multicast
        "240.0.0.0/4",      # reserved
        "255.255.255.255/32",
    )


# ---------------------------------------------------------------------------
# Subnet helper
# ---------------------------------------------------------------------------

def subnet_of(ip: str, prefix: int = 24) -> str:
    """Return the CIDR string for the IP at the given prefix."""
    try:
        net = ipaddress.ip_network(f"{ip}/{prefix}", strict=False)
        return str(net)
    except Exception:
        return ip


def is_in_any(ip: str, networks: tuple[str, ...]) -> bool:
    """True if ip is in any of the given networks."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for cidr in networks:
        try:
            if addr in ipaddress.ip_network(cidr, strict=False):
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Blocker
# ---------------------------------------------------------------------------

class SubnetBlocker:
    """Detects and blocks coordinated subnet attacks."""

    def __init__(self, cfg: SubnetConfig | None = None) -> None:
        self.cfg = cfg or SubnetConfig()
        self.extractor = GlobalFeatureExtractor()
        self._blocked_subnets: set[str] = set()

        log.info(f"SubnetBlocker ready: prefix=/{self.cfg.prefix} "
                 f"min_ips={self.cfg.min_ips_in_subnet} "
                 f"min_signals={self.cfg.min_global_signals}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def check(self, records: list[ParsedLog]) -> list[Decision]:
        """Return subnet-block decisions for the current window."""
        if not records:
            return []

        # 1. Global features
        features = self.extractor.extract(records)
        if features is None:
            return []

        signals = detect_anomalies(features)
        if len(signals) < self.cfg.min_global_signals:
            log.debug(f"subnet: only {len(signals)} signals, "
                      f"min={self.cfg.min_global_signals}")
            return []

        # 2. Count IPs per subnet
        subnet_counter: Counter[str] = Counter()
        for r in records:
            subnet_counter[subnet_of(r.ip, self.cfg.prefix)] += 1

        unique_ips = len(set(r.ip for r in records))
        if unique_ips == 0:
            return []

        # 3. Find dominant subnet
        decisions: list[Decision] = []
        for subnet, count in subnet_counter.most_common(3):
            share = count / unique_ips

            if count < self.cfg.min_ips_in_subnet:
                continue
            if share < self.cfg.min_subnet_share:
                continue
            if subnet in self._blocked_subnets:
                log.debug(f"subnet {subnet} already blocked")
                continue

            # Safety check — never block private/reserved ranges
            if self._is_private(subnet):
                log.warning(f"subnet {subnet} is private — skipping block")
                continue

            reason = (
                f"subnet_attack: {count} ips from {subnet} "
                f"({share:.0%}), signals: {', '.join(signals[:3])}"
            )

            decisions.append(Decision(
                ip=subnet,
                action="block",
                confidence=self.cfg.confidence,
                reason=reason,
                ttl_seconds=int(self.cfg.block_duration),
            ))

            log.warning(f"SUBNET BLOCK: {subnet} "
                        f"({count} ips, {share:.0%}, "
                        f"{len(signals)} signals)")

            # Only block one subnet per window to avoid collateral damage
            break

        return decisions

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------
    def _is_private(self, subnet: str) -> bool:
        """Check if subnet overlaps with never-block ranges."""
        try:
            net = ipaddress.ip_network(subnet, strict=False)
        except Exception:
            return True     # if unparseable, treat as private (safe)

        for cidr in self.cfg.never_block:
            try:
                other = ipaddress.ip_network(cidr, strict=False)
                if net.overlaps(other):
                    return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------
    def mark_blocked(self, subnet: str) -> None:
        """Record that we've blocked a subnet."""
        self._blocked_subnets.add(subnet)

    def mark_unblocked(self, subnet: str) -> None:
        self._blocked_subnets.discard(subnet)

    def list_blocked(self) -> list[str]:
        return sorted(self._blocked_subnets)

    def reset(self) -> None:
        self._blocked_subnets.clear()
        self.extractor.reset()


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarize(features: GlobalFeatures | None,
              signals: list[str],
              decisions: list[Decision]) -> dict:
    """Return a compact summary dict for logging."""
    if features is None:
        return {"status": "no_features"}

    return {
        "total_requests": features.total_requests,
        "unique_ips": features.unique_ips,
        "requests_per_ip": round(features.requests_per_ip, 2),
        "signals": len(signals),
        "signal_list": signals[:5],
        "decisions": len(decisions),
        "subnets": [d.ip for d in decisions],
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    from collector.nginx_parser import parse_file

    p = argparse.ArgumentParser(description="Subnet blocker test")
    p.add_argument("--file", default="data/raw/botnet_test.log")
    args = p.parse_args()

    records = parse_file(args.file)
    print(f"loaded {len(records)} records from {args.file}")
    print()

    sb = SubnetBlocker()
    decisions = sb.check(records)

    print("=== Decisions ===")
    if not decisions:
        print("  no subnet block decisions")
    else:
        for d in decisions:
            print(f"  block {d.ip} (conf={d.confidence}, "
                  f"ttl={d.ttl_seconds}s)")
            print(f"    reason: {d.reason}")