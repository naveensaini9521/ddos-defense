"""Enrich observed IPs with subnet, ASN, and TLS-fingerprint context.

This is what makes dynamic-IP defense possible. Every log record gets:

    subnet      203.0.113.0/24     ← what to block when we see rotation
    asn         9009               ← who owns the IP (datacenter? mobile?)
    asn_type    datacenter         ← classification that drives decisions
    ja3         abc123...          ← TLS fingerprint (optional)

Used by:
    - pipeline/live_collector.py    (attach to every parsed record)
    - pipeline/dynamic_ip.py        (aggregate concentration per subnet/ASN)
    - blocker/block_manager.py      (store context in Redis)

Design notes:
    - Uses pyasn if a local database exists (fast, offline, accurate).
    - Falls back to a curated ASN table for well-known datacenter ranges
      and for our simulated test ranges (203.0.113.x, 198.51.100.x,
      192.0.2.x — the RFC 5737 documentation networks).
    - Results are LRU-cached; each unique IP pays the lookup cost once.

Usage:
    from collector.enrich import enrich

    ctx = enrich("203.0.113.42")
    # {'ip': '203.0.113.42', 'subnet': '203.0.113.0/24',
    #  'asn': 9009, 'asn_type': 'datacenter', 'ja3': None}
"""
from __future__ import annotations

import ipaddress
from functools import lru_cache
from pathlib import Path

from core.logging import get_logger

log = get_logger("collector.enrich")


# ---------------------------------------------------------------------------
# ASN classification tables
# ---------------------------------------------------------------------------

# Well-known datacenter / hosting ASNs. Dynamic-IP attacks usually originate
# from these because spinning up cheap VPS instances is easy.
DATACENTER_ASNS: set[int] = {
    16509,    # Amazon AWS
    14618,    # Amazon AWS
    15169,    # Google Cloud
    396982,   # Google Cloud
    14061,    # DigitalOcean
    20473,    # Vultr
    63949,    # Linode / Akamai
    24940,    # Hetzner
    16276,    # OVH
    9009,     # M247 (NordVPN infrastructure)
    212238,   # Datacamp (NordVPN infrastructure)
    60068,    # Datacamp
    8075,     # Microsoft Azure
    20940,    # Akamai
    19551,    # Incapsula
    13335,    # Cloudflare
    54113,    # Fastly
}

# Mobile carrier ASNs — dynamic IPs by nature, but usually legitimate.
MOBILE_ASNS: set[int] = {
    21928,    # T-Mobile USA
    6167,     # Verizon Wireless
    22394,    # Cellco Verizon
    10507,    # Sprint
    3320,     # Deutsche Telekom
    5607,     # Sky UK
    5089,     # Virgin Media
}

# Curated IP → ASN mapping for the RFC 5737 test ranges. This lets the
# simulator produce realistic-looking ASN context without needing pyasn.
# (Real ASNs would never be in 203.0.113.0/24, but that's what makes this
# useful for testing: our dynamic_ip scenario generates IPs from here.)
TEST_RANGE_ASN: dict[str, tuple[int, str]] = {
    "203.0.113.0/24": (9009, "M247"),          # simulate NordVPN
    "198.51.100.0/24": (16509, "Amazon AWS"),
    "192.0.2.0/24": (14061, "DigitalOcean"),
}


# ---------------------------------------------------------------------------
# Optional pyasn (real lookups when a DB is present)
# ---------------------------------------------------------------------------

_ASN_DB = None
_ASN_DB_PATH = Path("data/ipasn.dat")

try:
    import pyasn  # type: ignore
    if _ASN_DB_PATH.exists():
        _ASN_DB = pyasn.pyasn(str(_ASN_DB_PATH))
        log.info(f"pyasn loaded: {_ASN_DB_PATH}")
    else:
        log.info(f"pyasn importable but {_ASN_DB_PATH} missing; using curated table only")
except ImportError:
    log.info("pyasn not installed; using curated ASN table only")


# ---------------------------------------------------------------------------
# Core helpers (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=100_000)
def subnet_of(ip: str, prefix: int = 24) -> str:
    """Return the /prefix subnet string for an IPv4 or IPv6 address."""
    try:
        net = ipaddress.ip_network(ip, strict=False)
        # For IPv6, use /64 — that's the closest analog to a /24 in IPv4
        effective_prefix = prefix if net.version == 4 else max(prefix, 64)
        return str(net.supernet(new_prefix=effective_prefix))
    except Exception:
        return ip


@lru_cache(maxsize=100_000)
def asn_of(ip: str) -> int | None:
    """Real ASN lookup via pyasn if available, else curated table."""
    # 1. Curated test-range table first (handles the simulator's fake IPs)
    try:
        addr = ipaddress.ip_address(ip)
        for cidr, (asn_num, _name) in TEST_RANGE_ASN.items():
            if addr in ipaddress.ip_network(cidr, strict=False):
                return asn_num
    except Exception:
        pass

    # 2. Real pyasn lookup
    if _ASN_DB is not None:
        try:
            asn_num, _prefix = _ASN_DB.lookup(ip)
            return int(asn_num) if asn_num else None
        except Exception:
            return None

    return None


@lru_cache(maxsize=1024)
def asn_type(asn_num: int | None) -> str:
    """Classify an ASN as datacenter / mobile / residential / unknown."""
    if asn_num is None:
        return "unknown"
    if asn_num in DATACENTER_ASNS:
        return "datacenter"
    if asn_num in MOBILE_ASNS:
        return "mobile"
    return "residential"


@lru_cache(maxsize=1024)
def asn_name(asn_num: int | None) -> str:
    """Human-readable name for an ASN. Falls back to 'ASN<n>'."""
    if asn_num is None:
        return "unknown"
    for _cidr, (num, name) in TEST_RANGE_ASN.items():
        if num == asn_num:
            return name
    return f"ASN{asn_num}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich(ip: str, ja3: str | None = None) -> dict:
    """Return full enrichment context for one IP.

    Args:
        ip: source IP address
        ja3: optional TLS fingerprint (from nginx with the right module)

    Returns dict with keys: ip, subnet, asn, asn_type, asn_name, ja3
    """
    sn = subnet_of(ip)
    a = asn_of(ip)
    return {
        "ip": ip,
        "subnet": sn,
        "asn": a,
        "asn_type": asn_type(a),
        "asn_name": asn_name(a),
        "ja3": ja3,
    }


def enrich_batch(ips: list[str]) -> dict[str, dict]:
    """Enrich a batch of IPs. Returns {ip: ctx}."""
    return {ip: enrich(ip) for ip in set(ips)}


def is_datacenter_ip(ip: str) -> bool:
    """Convenience: True if this IP's ASN is a known datacenter."""
    return asn_type(asn_of(ip)) == "datacenter"


def stats() -> dict:
    """Diagnostics: cache info + which DB is loaded."""
    return {
        "pyasn_loaded": _ASN_DB is not None,
        "pyasn_path": str(_ASN_DB_PATH) if _ASN_DB is not None else None,
        "subnet_cache": subnet_of.cache_info()._asdict(),
        "asn_cache": asn_of.cache_info()._asdict(),
    }


# ---------------------------------------------------------------------------
# CLI — quick checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="IP enrichment lookup")
    p.add_argument("--ip", help="single IP to enrich")
    p.add_argument("--batch", nargs="+", help="multiple IPs")
    p.add_argument("--stats", action="store_true", help="show cache stats")
    args = p.parse_args()

    if args.stats:
        print(json.dumps(stats(), indent=2, default=str))
        raise SystemExit(0)

    if args.ip:
        print(json.dumps(enrich(args.ip), indent=2))
    elif args.batch:
        for ip in args.batch:
            print(json.dumps(enrich(ip)))
    else:
        # Default demo: show the three test ranges
        demo = ["203.0.113.42", "198.51.100.7", "192.0.2.99", "8.8.8.8"]
        print("=== Enrichment Demo ===")
        for ip in demo:
            ctx = enrich(ip)
            print(
                f"{ip:16s}  subnet={ctx['subnet']:20s}  "
                f"asn={str(ctx['asn']):>8s}  type={ctx['asn_type']:12s}  "
                f"name={ctx['asn_name']}"
            )