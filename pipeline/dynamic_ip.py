"""Dynamic / rotating IP detection with ASN awareness.

Solves the "each IP sends only 1 request" problem: no single IP looks bad,
but the aggregate pattern across many IPs is an attack.

Detects:
    - IP churn (many new source IPs per second)
    - Low requests-per-IP (botnet signature)
    - Path concentration (all IPs hitting same path)
    - UA concentration (all IPs using same User-Agent)
    - Subnet concentration (attack from one /24)
    - ASN concentration (attack from one autonomous system)
    - Datacenter concentration (attack spread across multiple cloud ASNs)
    - ASN type (datacenter = high suspicion; mobile = low suspicion)

Emits:
    - ASNBlockDecision   when a single ASN dominates OR all traffic is datacenter
    - SubnetBlockDecision fallback when ASN data missing but subnet dominates
    - nothing            when suspicious but not blockable

Usage:
    from pipeline.dynamic_ip import DynamicIPDetector

    detector = DynamicIPDetector()
    decisions = detector.check(records, features)
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from collector.aggregator import IPFeatures
from collector.nginx_parser import ParsedLog
from core.logging import get_logger
from core.schema import Decision

log = get_logger("pipeline.dynamic_ip")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DynamicConfig:
    """Thresholds for dynamic-IP detection."""

    # Base detection
    min_unique_ips: int = 10
    new_ip_rate_threshold: float = 5.0        # new IPs/sec of the window
    requests_per_ip_threshold: float = 2.0    # avg requests per IP
    path_concentration_threshold: float = 0.7
    ua_concentration_threshold: float = 0.8
    subnet_concentration_threshold: float = 0.5
    min_subnet_size: int = 8
    window_seconds: float = 10.0

    # ASN-specific
    asn_concentration_threshold: float = 0.5      # ≥50% of IPs from one ASN
    datacenter_share_threshold: float = 0.8       # ≥80% from datacenter ASNs
    min_asn_size: int = 8
    asn_block_enabled: bool = True
    datacenter_suspicion_weight: float = 3.0
    mobile_suspicion_weight: float = 0.3

    # Blocking
    subnet_size: int = 24
    block_duration: float = 300.0
    confidence: float = 0.85

    # Never block (safety) — these ASNs carry huge legitimate traffic
    never_block_asns: set[int] = field(default_factory=lambda: {
        15169,     # Google
        13335,     # Cloudflare
        20940,     # Akamai
    })


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DynamicSignal:
    """Per-window analysis of IP-level patterns."""
    window_start: float
    window_end: float
    unique_ips: int
    total_requests: int
    avg_requests_per_ip: float
    new_ip_rate: float
    path_entropy_ratio: float
    ua_entropy_ratio: float

    top_subnet: str | None
    top_subnet_share: float

    top_asn: int | None
    top_asn_share: float
    top_asn_type: str | None
    top_asn_name: str | None
    asn_distribution: dict[int, int] = field(default_factory=dict)
    datacenter_share: float = 0.0
    datacenter_asns: list[int] = field(default_factory=list)

    is_suspicious: bool = False
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Subnet helper
# ---------------------------------------------------------------------------

def subnet_of(ip: str, prefix: int = 24) -> str:
    """Return the /prefix subnet string for an IPv4 address."""
    try:
        parts = ip.split(".")
        if len(parts) != 4:
            return ip
        if prefix == 24:
            return ".".join(parts[:3]) + ".0/24"
        if prefix == 16:
            return ".".join(parts[:2]) + ".0.0/16"
        if prefix == 8:
            return parts[0] + ".0.0.0/8"
        return ip
    except Exception:
        return ip


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------

def _entropy_ratio(counter: Counter, total: int) -> float:
    """Normalized Shannon entropy (0..1). 1 = perfectly uniform."""
    if total <= 0 or len(counter) <= 1:
        return 0.0
    h = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    max_h = math.log2(len(counter))
    return h / max_h if max_h > 0 else 0.0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DynamicIPDetector:
    """Analyzes cross-IP patterns to detect rotating/botnet attacks."""

    def __init__(self, config: DynamicConfig | None = None) -> None:
        self.cfg = config or DynamicConfig()
        self._seen_ips: set[str] = set()
        self._last_reset: float = time.time()

        log.info(
            f"DynamicIPDetector ready: subnet=/{self.cfg.subnet_size} "
            f"asn_block={self.cfg.asn_block_enabled} "
            f"block_duration={self.cfg.block_duration}s"
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def analyze(
        self,
        records: list[ParsedLog],
        features: dict[str, IPFeatures] | None = None,
    ) -> DynamicSignal | None:
        """Analyze a batch of records for dynamic-IP patterns."""
        if not records:
            return None

        window_start = min(r.ts for r in records)
        window_end = max(r.ts for r in records)
        window_secs = max(window_end - window_start, 0.001)

        # Aggregate per-IP, per-path, per-UA
        by_ip: Counter[str] = Counter()
        by_path: Counter[str] = Counter()
        by_ua: Counter[str] = Counter()

        for r in records:
            by_ip[r.ip] += 1
            by_path[r.path] += 1
            by_ua[r.user_agent or "unknown"] += 1

        total = len(records)
        unique_ips = len(by_ip)
        avg_req_per_ip = total / max(unique_ips, 1)

        # New IP rate — measured against the WINDOW DURATION, not wall-clock
        # since construction (which gave absurd values right after init).
        new_ips = by_ip.keys() - self._seen_ips
        new_ip_rate = len(new_ips) / window_secs

        # Entropy metrics
        path_ratio = _entropy_ratio(by_path, total)
        ua_ratio = _entropy_ratio(by_ua, total)

        # Subnet concentration
        subnet_counter: Counter[str] = Counter()
        for ip in by_ip:
            subnet_counter[subnet_of(ip, self.cfg.subnet_size)] += 1
        top_subnet, top_share = (None, 0.0)
        if subnet_counter:
            top_subnet, top_count = subnet_counter.most_common(1)[0]
            top_share = top_count / unique_ips

        # ASN analysis — requires enrichment fields on records
        asn_counter: Counter[int] = Counter()
        asn_type_map: dict[int, str] = {}
        for r in records:
            asn = getattr(r, "asn", None)
            if asn is None:
                continue
            asn_counter[asn] += 1
            if asn not in asn_type_map:
                atype = getattr(r, "asn_type", None)
                if atype:
                    asn_type_map[asn] = atype

        asn_total = sum(asn_counter.values())
        top_asn, top_asn_share = (None, 0.0)
        top_asn_type = None
        top_asn_name = None
        if asn_counter:
            top_asn, top_count = asn_counter.most_common(1)[0]
            top_asn_share = top_count / max(asn_total, 1)
            top_asn_type = asn_type_map.get(top_asn, "unknown")
            top_asn_name = f"ASN{top_asn}"

        # Combined datacenter share
        datacenter_ips = 0
        datacenter_asns: list[int] = []
        for asn, count in asn_counter.items():
            if asn_type_map.get(asn) == "datacenter":
                datacenter_ips += count
                datacenter_asns.append(asn)
        datacenter_share = datacenter_ips / max(asn_total, 1)

        # --- Detection signals ---
        reasons: list[str] = []

        # Too few IPs to talk about "rotation"
        if unique_ips < self.cfg.min_unique_ips:
            signal = DynamicSignal(
                window_start=window_start,
                window_end=window_end,
                unique_ips=unique_ips,
                total_requests=total,
                avg_requests_per_ip=avg_req_per_ip,
                new_ip_rate=new_ip_rate,
                path_entropy_ratio=path_ratio,
                ua_entropy_ratio=ua_ratio,
                top_subnet=top_subnet,
                top_subnet_share=top_share,
                top_asn=top_asn,
                top_asn_share=top_asn_share,
                top_asn_type=top_asn_type,
                top_asn_name=top_asn_name,
                asn_distribution=dict(asn_counter),
                datacenter_share=datacenter_share,
                datacenter_asns=datacenter_asns,
                is_suspicious=False,
                reasons=["insufficient unique IPs"],
            )
            self._update_seen(by_ip)
            return signal

        # Standard dynamic-IP signals
        if new_ip_rate > self.cfg.new_ip_rate_threshold:
            reasons.append(f"high new-IP rate ({new_ip_rate:.1f}/s)")

        if avg_req_per_ip < self.cfg.requests_per_ip_threshold:
            reasons.append(f"low req/IP ({avg_req_per_ip:.1f})")

        if path_ratio < self.cfg.path_concentration_threshold:
            reasons.append(f"path concentration (H={path_ratio:.2f})")

        if ua_ratio < self.cfg.ua_concentration_threshold:
            reasons.append(f"UA concentration (H={ua_ratio:.2f})")

        if top_share >= self.cfg.subnet_concentration_threshold:
            reasons.append(
                f"subnet concentration ({top_subnet} = {top_share:.0%})"
            )

        # Single ASN concentration
        if top_asn is not None and top_asn_share >= self.cfg.asn_concentration_threshold:
            weight = 1.0
            if top_asn_type == "datacenter":
                weight = self.cfg.datacenter_suspicion_weight
            elif top_asn_type == "mobile":
                weight = self.cfg.mobile_suspicion_weight
            reasons.append(
                f"ASN concentration (AS{top_asn}/{top_asn_type} = "
                f"{top_asn_share:.0%} w={weight:.1f})"
            )

        # Datacenter aggregate — catches multi-cloud botnets where no
        # single ASN dominates but all traffic is from cloud providers.
        if (datacenter_share >= self.cfg.datacenter_share_threshold
                and len(datacenter_asns) >= 2):
            reasons.append(
                f"datacenter concentration ({datacenter_share:.0%} from "
                f"{len(datacenter_asns)} datacenter ASN(s))"
            )

        is_suspicious = len(reasons) >= 2

        signal = DynamicSignal(
            window_start=window_start,
            window_end=window_end,
            unique_ips=unique_ips,
            total_requests=total,
            avg_requests_per_ip=avg_req_per_ip,
            new_ip_rate=new_ip_rate,
            path_entropy_ratio=path_ratio,
            ua_entropy_ratio=ua_ratio,
            top_subnet=top_subnet,
            top_subnet_share=top_share,
            top_asn=top_asn,
            top_asn_share=top_asn_share,
            top_asn_type=top_asn_type,
            top_asn_name=top_asn_name,
            asn_distribution=dict(asn_counter),
            datacenter_share=datacenter_share,
            datacenter_asns=datacenter_asns,
            is_suspicious=is_suspicious,
            reasons=reasons,
        )

        self._update_seen(by_ip)

        if is_suspicious:
            log.warning(
                f"dynamic IP signal: {signal.unique_ips} IPs, "
                f"top ASN {top_asn} ({top_asn_share:.0%}), "
                f"dc_share={datacenter_share:.0%}, "
                f"reasons: {', '.join(reasons)}"
            )
        else:
            log.debug(
                f"dynamic IP ok: {unique_ips} IPs, "
                f"top ASN {top_asn} ({top_asn_share:.0%}), "
                f"no strong signal"
            )

        return signal

    # ------------------------------------------------------------------
    # Decision generation
    # ------------------------------------------------------------------
    def decisions_from_signal(
        self,
        signal: DynamicSignal | None,
    ) -> list[Decision]:
        """Emit ASN or subnet block decisions."""
        if signal is None or not signal.is_suspicious:
            return []

        # Prefer ASN block when we have ASN data
        if self.cfg.asn_block_enabled and signal.top_asn is not None:
            asn = signal.top_asn
            share = signal.top_asn_share
            atype = signal.top_asn_type or "unknown"

            if asn in self.cfg.never_block_asns:
                log.warning(
                    f"ASN {asn} is on the never-block list; "
                    f"falling back to subnet scoping"
                )
            elif share >= self.cfg.asn_concentration_threshold:
                # Single-ASN attack (e.g. NordVPN fleet from ASN 9009)
                return [self._asn_decision(signal, asn, share, atype, reason_kind="single")]

            elif (signal.datacenter_share >= self.cfg.datacenter_share_threshold
                  and len(signal.datacenter_asns) >= 2):
                # Multi-cloud attack: block the largest datacenter ASN.
                # We could emit multiple decisions (one per ASN), but for
                # safety we start with just the top one.
                return [self._asn_decision(signal, asn, share, atype, reason_kind="multi")]

        # Fallback: subnet block
        if (signal.top_subnet is None
                or signal.top_subnet_share < self.cfg.subnet_concentration_threshold
                or signal.unique_ips < self.cfg.min_subnet_size):
            return []

        reason = "dynamic_ip: " + ", ".join(signal.reasons[:3])
        d = Decision(
            ip=signal.top_subnet,
            action="block",
            confidence=self.cfg.confidence,
            reason=reason,
            ttl_seconds=int(self.cfg.block_duration),
        )
        object.__setattr__(d, "subnet", signal.top_subnet)
        object.__setattr__(d, "scope", "subnet")

        log.warning(
            f"SUBNET BLOCK DECISION: {signal.top_subnet} "
            f"ips={signal.unique_ips} ttl={self.cfg.block_duration:.0f}s"
        )
        return [d]

    def _asn_decision(
        self,
        signal: DynamicSignal,
        asn: int,
        share: float,
        atype: str,
        reason_kind: str = "single",
    ) -> Decision:
        """Build an ASN-scoped Decision."""
        if reason_kind == "multi":
            reason_prefix = (
                f"datacenter_attack: {signal.unique_ips} ips, "
                f"{signal.datacenter_share:.0%} datacenter; top AS{asn}"
            )
        else:
            reason_prefix = (
                f"asn_attack: {signal.unique_ips} ips "
                f"({share:.0%} from AS{asn}/{atype})"
            )
        reason = reason_prefix + f"; signals: {', '.join(signal.reasons[:3])}"

        d = Decision(
            ip=f"AS{asn}",
            action="block",
            confidence=self.cfg.confidence,
            reason=reason,
            ttl_seconds=int(self.cfg.block_duration),
        )
        object.__setattr__(d, "asn", asn)
        object.__setattr__(d, "scope", "asn")

        log.warning(
            f"ASN BLOCK DECISION: AS{asn} ({atype}) "
            f"share={share:.0%} dc_share={signal.datacenter_share:.0%} "
            f"ttl={self.cfg.block_duration:.0f}s"
        )
        return d

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def check(
        self,
        records: list[ParsedLog],
        features: dict[str, IPFeatures] | None = None,
    ) -> list[Decision]:
        """Analyze records and return dynamic-block decisions."""
        signal = self.analyze(records, features)
        return self.decisions_from_signal(signal)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _update_seen(self, by_ip: Counter) -> None:
        self._seen_ips.update(by_ip.keys())

    def reset(self) -> None:
        self._seen_ips.clear()
        self._last_reset = time.time()


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from dataclasses import replace

    from collector.enrich import enrich
    from collector.nginx_parser import parse_file

    p = argparse.ArgumentParser(description="Dynamic IP detector test")
    p.add_argument("--file", default="data/raw/training.log")
    args = p.parse_args()

    records = parse_file(args.file)
    print(f"loaded {len(records)} records from {args.file}")

    # Attach enrichment if not already present
    if records and getattr(records[0], "asn", None) is None:
        records = [
            replace(
                r,
                subnet=enrich(r.ip)["subnet"],
                asn=enrich(r.ip)["asn"],
                asn_type=enrich(r.ip)["asn_type"],
            )
            for r in records
        ]

    det = DynamicIPDetector()
    signal = det.analyze(records)

    if signal is None:
        print("no signal")
    else:
        print()
        print("=== Dynamic IP Signal ===")
        print(f"  unique IPs:       {signal.unique_ips}")
        print(f"  total requests:   {signal.total_requests}")
        print(f"  avg req/IP:       {signal.avg_requests_per_ip:.2f}")
        print(f"  new IP rate:      {signal.new_ip_rate:.2f}/s")
        print(f"  path entropy:     {signal.path_entropy_ratio:.2f}")
        print(f"  UA entropy:       {signal.ua_entropy_ratio:.2f}")
        print(f"  top subnet:       {signal.top_subnet} ({signal.top_subnet_share:.0%})")
        print(f"  top ASN:          AS{signal.top_asn} "
              f"({signal.top_asn_type}, {signal.top_asn_share:.0%})")
        print(f"  ASN distribution: {signal.asn_distribution}")
        print(f"  datacenter share: {signal.datacenter_share:.0%} "
              f"({signal.datacenter_asns})")
        print(f"  suspicious:       {signal.is_suspicious}")
        print(f"  reasons:          {signal.reasons}")

        decisions = det.decisions_from_signal(signal)
        if decisions:
            print()
            print("=== Block Decisions ===")
            for d in decisions:
                scope = getattr(d, "scope", "ip")
                print(f"  block {d.ip} (scope={scope}, conf={d.confidence}, "
                      f"ttl={d.ttl_seconds}s)")
                print(f"    reason: {d.reason}")
        else:
            print()
            print("  no block decision produced")