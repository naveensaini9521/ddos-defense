"""TTL and exponential backoff for block management.

Responsibilities:
    - Compute TTL based on strike count (exponential backoff)
    - Background thread that unblocks expired IPs
    - Called by block_manager to decide how long a block lasts

Backoff schedule (default, configurable):
    strike 1  → 60s     (1 minute)
    strike 2  → 300s    (5 minutes)
    strike 3  → 1800s   (30 minutes)
    strike 4  → 86400s  (24 hours)
    strike 5+ → -1      (permanent — requires manual unblock)

Usage:
    from blocker.expiry import ExpiryPolicy, ExpiryDaemon

    policy = ExpiryPolicy()                      # default schedule
    ttl = policy.ttl_for(strike=2)               # → 300

    daemon = ExpiryDaemon(state, on_unblock=callback)
    daemon.start()                               # background thread
    daemon.stop()
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from core.logging import get_logger

log = get_logger("blocker.expiry")


# ---------------------------------------------------------------------------
# Policy — TTL computation
# ---------------------------------------------------------------------------

# Special value meaning "never expire"
PERMANENT = -1.0


@dataclass
class ExpiryPolicy:
    """Computes block TTL from strike count.

    Attributes:
        backoff: TTL (seconds) for strike 1, 2, 3, ...
                 After the list is exhausted, use `permanent_threshold`.
        permanent_threshold: if strike >= this, block is permanent.
        max_ttl: cap TTL at this value (safety limit).
    """

    backoff: list[float] = field(default_factory=lambda: [
        60,       # strike 1 → 1 minute
        300,      # strike 2 → 5 minutes
        1800,     # strike 3 → 30 minutes
        86400,    # strike 4 → 24 hours
    ])
    permanent_threshold: int = 5
    max_ttl: float = 86400.0    # 24 hours cap by default

    def ttl_for(self, strike: int) -> float:
        """Return TTL in seconds for a given strike count.

        Returns:
            > 0  → normal TTL
            -1   → permanent (never expires)
        """
        if strike <= 0:
            strike = 1

        if strike >= self.permanent_threshold:
            return PERMANENT

        idx = min(strike - 1, len(self.backoff) - 1)
        ttl = self.backoff[idx]

        if self.max_ttl > 0:
            ttl = min(ttl, self.max_ttl)

        return float(ttl)

    def describe(self, strike: int) -> str:
        """Human-readable schedule description for a given strike."""
        ttl = self.ttl_for(strike)
        if ttl == PERMANENT:
            return "permanent (manual review)"
        if ttl < 60:
            return f"{ttl:.0f}s"
        if ttl < 3600:
            return f"{ttl / 60:.0f}min"
        if ttl < 86400:
            return f"{ttl / 3600:.1f}hr"
        return f"{ttl / 86400:.1f}day"


# ---------------------------------------------------------------------------
# Daemon — background unblock thread
# ---------------------------------------------------------------------------

UnblockCallback = Callable[[str], None]


class ExpiryDaemon:
    """Background thread that periodically unblocks expired IPs."""

    def __init__(
        self,
        state,                       # BlockState instance
        on_unblock: UnblockCallback | None = None,
        check_interval: float = 5.0,
        auto_purge: bool = True,
    ) -> None:
        self.state = state
        self.on_unblock = on_unblock or (lambda ip: None)
        self.check_interval = check_interval
        self.auto_purge = auto_purge

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Stats
        self.total_unblocked = 0
        self.total_purged = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            log.warning("ExpiryDaemon already running")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="expiry-daemon")
        self._thread.start()
        log.info(f"ExpiryDaemon started (interval={self.check_interval}s)")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        log.info(f"ExpiryDaemon stopped "
                 f"(unblocked={self.total_unblocked}, purged={self.total_purged})")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                log.warning(f"expiry tick failed: {e}")
            self._stop.wait(self.check_interval)

    def _tick(self) -> None:
        """One iteration: check expired, call callback, optionally purge."""
        now = time.time()
        expired = self.state.expired_blocks()

        for entry in expired:
            # Notify callback (block_manager uses this to remove iptables rule)
            try:
                self.on_unblock(entry.ip)
            except Exception as e:
                log.warning(f"on_unblock failed for {entry.ip}: {e}")

            # Remove from current blocks (but keep in history)
            self.state.remove_block(entry.ip)
            self.total_unblocked += 1

            log.info(f"auto-unblocked: {entry.ip} "
                     f"(was blocked {now - entry.since:.0f}s)")

        if self.auto_purge:
            n = self.state.purge_expired()
            if n:
                self.total_purged += n

    # ------------------------------------------------------------------
    # Manual trigger (for tests)
    # ------------------------------------------------------------------
    def tick_once(self) -> int:
        """Run a single tick synchronously. Returns number unblocked."""
        before = self.total_unblocked
        self._tick()
        return self.total_unblocked - before


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_ttl(ttl: float) -> str:
    """Format a TTL for display."""
    if ttl == PERMANENT:
        return "permanent"
    if ttl < 60:
        return f"{ttl:.0f}s"
    if ttl < 3600:
        return f"{ttl / 60:.1f}min"
    if ttl < 86400:
        return f"{ttl / 3600:.1f}hr"
    return f"{ttl / 86400:.1f}day"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Expiry policy viewer")
    p.add_argument("--strikes", type=int, default=6,
                   help="max strike count to display")
    args = p.parse_args()

    policy = ExpiryPolicy()

    print("=== Expiry Policy ===")
    print(f"backoff schedule: {policy.backoff}")
    print(f"permanent after:  strike {policy.permanent_threshold}")
    print(f"max TTL:          {format_ttl(policy.max_ttl)}")
    print()
    print(f"{'strike':>6s}  {'ttl':>12s}  description")
    print("-" * 40)
    for strike in range(1, args.strikes + 1):
        ttl = policy.ttl_for(strike)
        print(f"{strike:>6d}  {format_ttl(ttl):>12s}  {policy.describe(strike)}")