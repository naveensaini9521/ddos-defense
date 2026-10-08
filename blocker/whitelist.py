"""Whitelist — never-block list for critical infrastructure IPs.

Protects:
    - Loopback (127.0.0.0/8)
    - Private gateways
    - DNS servers
    - Admin IPs (from config)
    - Docker/Kubernetes service ranges

Usage:
    from blocker.whitelist import Whitelist

    wl = Whitelist()
    # wl.add_ip("192.168.122.1")
    wl.add_network("10.0.0.0/24")

    if wl.is_whitelisted("192.168.122.1"):
        ...  # skip blocking
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from pathlib import Path

from core.logging import get_logger

log = get_logger("blocker.whitelist")


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class Whitelist:
    """Holds exact IPs and CIDR ranges that must never be blocked."""

    ips: set[str] = field(default_factory=set)
    networks: list = field(default_factory=list)

    def __post_init__(self) -> None:
        # Always include safe defaults
        self.ips.update({
            "127.0.0.1",
            "::1",
            "0.0.0.0",
        })
        # RFC1918 private networks are not auto-whitelisted by default —
        # we do add loopback and link-local because they can never be
        # remote attackers.
        self.networks.extend([
            ipaddress.ip_network("127.0.0.0/8"),
            ipaddress.ip_network("169.254.0.0/16"),
        ])

    # ------------------------------------------------------------------
    # Adding entries
    # ------------------------------------------------------------------
    def add_ip(self, ip: str) -> None:
        """Add an exact IP to the whitelist."""
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            log.warning(f"not a valid IP: {ip}")
            return
        self.ips.add(ip)

    def add_network(self, cidr: str) -> None:
        """Add a CIDR range to the whitelist."""
        try:
            net = ipaddress.ip_network(cidr, strict=False)
        except ValueError:
            log.warning(f"not a valid CIDR: {cidr}")
            return
        self.networks.append(net)

    def add_many(self, entries: list[str]) -> None:
        """Add a list of IPs and/or CIDRs — auto-detects type."""
        for entry in entries:
            if "/" in entry:
                self.add_network(entry)
            else:
                self.add_ip(entry)

    # ------------------------------------------------------------------
    # Checking
    # ------------------------------------------------------------------
    def is_whitelisted(self, ip: str) -> bool:
        """True if IP is exact-matched or falls inside any whitelisted network."""
        if ip in self.ips:
            return True

        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False

        for net in self.networks:
            try:
                if addr in net:
                    return True
            except TypeError:
                # IPv4 vs IPv6 mismatch — skip
                continue
        return False

    def filter_allowed(self, ips: list[str]) -> list[str]:
        """Return the subset of IPs that are NOT whitelisted (safe to block)."""
        return [ip for ip in ips if not self.is_whitelisted(ip)]

    # ------------------------------------------------------------------
    # Bulk loading
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict) -> "Whitelist":
        """Build whitelist from a config dict (blocker.whitelist list)."""
        wl = cls()
        entries = config.get("blocker", {}).get("whitelist", [])
        wl.add_many(entries)
        log.info(f"whitelist loaded: {len(wl.ips)} exact IPs, "
                 f"{len(wl.networks)} networks")
        return wl

    @classmethod
    def from_file(cls, path: str | Path) -> "Whitelist":
        """Load from a text file (one IP or CIDR per line, # for comments)."""
        wl = cls()
        p = Path(path)
        if not p.exists():
            log.warning(f"whitelist file missing: {p}")
            return wl

        entries = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                entries.append(line)
        wl.add_many(entries)
        log.info(f"loaded {len(entries)} entries from {p}")
        return wl

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.ips) + len(self.networks)

    def dump(self) -> dict:
        return {
            "exact_ips": sorted(self.ips),
            "networks": [str(n) for n in self.networks],
        }

    def __repr__(self) -> str:
        return f"Whitelist(ips={len(self.ips)}, networks={len(self.networks)})"