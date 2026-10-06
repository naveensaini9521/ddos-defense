"""Orchestrator — turns ML Decisions into actual blocks.

Coordinates:
    - whitelist.py  (safety check)
    - state.py      (persistence)
    - expiry.py     (TTL + backoff + daemon)
    - firewall.py   (iptables)

Usage:
    from blocker.block_manager import BlockManager

    manager = BlockManager()                    # loads config automatically
    manager.enforce(decision)                   # one call does everything
    manager.enforce_many(decisions)
    manager.shutdown()
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from core.config_loader import load as load_config
from core.logging import get_logger
from core.schema import Decision
from blocker.expiry import ExpiryDaemon, ExpiryPolicy, PERMANENT, format_ttl
from blocker.firewall import Firewall, FirewallConfig
from blocker.state import BlockState
from blocker.whitelist import Whitelist

log = get_logger("blocker.block_manager")


# ---------------------------------------------------------------------------
# Stats (in-memory, reset on restart)
# ---------------------------------------------------------------------------

@dataclass
class ManagerStats:
    decisions_seen: int = 0
    blocks_enforced: int = 0
    whitelist_skips: int = 0
    allow_skips: int = 0
    throttle_skips: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def summary(self) -> dict:
        return {
            "decisions_seen": self.decisions_seen,
            "blocks_enforced": self.blocks_enforced,
            "whitelist_skips": self.whitelist_skips,
            "allow_skips": self.allow_skips,
            "throttle_skips": self.throttle_skips,
            "errors": self.errors,
            "uptime_seconds": int(time.time() - self.started_at),
        }


# ---------------------------------------------------------------------------
# BlockManager
# ---------------------------------------------------------------------------

class BlockManager:
    """Orchestrates the enforcement of ML Decisions."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        state: BlockState | None = None,
        firewall: Firewall | None = None,
        whitelist: Whitelist | None = None,
        policy: ExpiryPolicy | None = None,
        start_daemon: bool = True,
    ) -> None:
        # Load config (or accept explicit overrides)
        self.cfg = load_config(config_path) if config_path else load_config()

        # Build components
        self.whitelist = whitelist or Whitelist.from_config(self.cfg)
        self.state = state or BlockState()
        self.firewall = firewall or self._firewall_from_config()
        self.policy = policy or self._policy_from_config()

        # Stats
        self.stats = ManagerStats()

        # Expiry daemon (auto-unblock when TTL expires)
        self.daemon: ExpiryDaemon | None = None
        if start_daemon:
            self.daemon = ExpiryDaemon(
                state=self.state,
                on_unblock=self._on_unblock,
                check_interval=self._daemon_interval(),
                auto_purge=self.cfg.get("blocker", {}).get("auto_purge", True),
            )
            self.daemon.start()

        log.info(
            f"BlockManager ready: "
            f"backend={self.firewall.cfg.backend} "
            f"dry_run={self.firewall.cfg.dry_run} "
            f"whitelist={len(self.whitelist)} entries"
        )

    # ------------------------------------------------------------------
    # Config-driven construction
    # ------------------------------------------------------------------
    def _firewall_from_config(self) -> Firewall:
        b = self.cfg.get("blocker", {})
        return Firewall(FirewallConfig(
            backend=b.get("backend", "iptables"),
            dry_run=b.get("dry_run", False),
            remote_host=b.get("remote_host"),
            remote_sudo=b.get("remote_sudo", True),
            chain=b.get("chain", "INPUT"),
            rule_target=b.get("rule_target", "DROP"),
            use_sudo=b.get("use_sudo", True),
        ))

    def _policy_from_config(self) -> ExpiryPolicy:
        b = self.cfg.get("blocker", {})
        backoff = b.get("backoff_seconds", [60, 300, 1800, 86400])
        return ExpiryPolicy(
            backoff=[float(x) for x in backoff],
            permanent_threshold=int(b.get("permanent_after_strike", 5)),
            max_ttl=float(b.get("max_ttl", 86400)),
        )

    def _daemon_interval(self) -> float:
        return float(self.cfg.get("blocker", {}).get(
            "expiry_check_interval", 5.0
        ))

    # ------------------------------------------------------------------
    # Core: enforce decisions
    # ------------------------------------------------------------------
    def enforce(self, decision: Decision) -> dict:
        """Take action on a single Decision. Returns a result dict."""
        self.stats.decisions_seen += 1

        ip = decision.ip
        action = decision.action

        # 1. Only "block" actions trigger enforcement.
        #    "throttle" and "allow" are no-ops for now.
        if action == "allow":
            self.stats.allow_skips += 1
            return {"ip": ip, "action": "skipped", "reason": "allow"}

        if action == "throttle":
            self.stats.throttle_skips += 1
            log.info(f"throttle not implemented; skipping {ip}")
            return {"ip": ip, "action": "skipped", "reason": "throttle-unimplemented"}

        if action != "block":
            log.warning(f"unknown action '{action}' for {ip}; skipping")
            return {"ip": ip, "action": "skipped", "reason": "unknown-action"}

        # 2. Whitelist check — critical safety
        if self.whitelist.is_whitelisted(ip):
            self.stats.whitelist_skips += 1
            log.info(f"whitelist prevents blocking {ip}")
            return {"ip": ip, "action": "skipped", "reason": "whitelisted"}

        # 3. Compute strike count and TTL
        strike = self._next_strike(ip)
        ttl = self.policy.ttl_for(strike)

        # 4. Save to state DB (creates/extends record)
        entry = self.state.add_block(
            ip=ip,
            ttl=ttl if ttl != PERMANENT else 10 * 365 * 24 * 3600,  # ~10 years
            reason=decision.reason,
            confidence=decision.confidence,
            model=decision.reason.split(" ")[0].replace("model=", "") or "unknown",
        )
        # If PERMANENT was requested, override `until`
        if ttl == PERMANENT:
            with self.state._conn:  # pylint: disable=protected-access
                self.state._conn.execute(
                    "UPDATE blocks SET until = ? WHERE ip = ?",
                    (-1.0, ip),
                )

        # 5. Apply iptables
        fw_result = self.firewall.block(ip)

        if fw_result.ok:
            self.stats.blocks_enforced += 1
            log.info(
                f"ENFORCED block on {ip}  "
                f"ttl={format_ttl(ttl)}  "
                f"strike={strike}  "
                f"conf={decision.confidence:.2f}"
            )
        else:
            self.stats.errors += 1
            log.error(f"firewall.block({ip}) failed: {fw_result.stderr}")

        return {
            "ip": ip,
            "action": "blocked",
            "strike": strike,
            "ttl": ttl,
            "ttl_human": format_ttl(ttl),
            "confidence": decision.confidence,
            "fw_ok": fw_result.ok,
            "fw_command": fw_result.command,
        }

    def enforce_many(self, decisions: list[Decision]) -> list[dict]:
        """Process a batch of decisions."""
        return [self.enforce(d) for d in decisions]

    # ------------------------------------------------------------------
    # Strike calculation
    # ------------------------------------------------------------------
    def _next_strike(self, ip: str) -> int:
        existing = self.state.get(ip)
        if existing is None:
            # Check history for past offenses
            history = self.state.history(ip, limit=10)
            block_events = [h for h in history if h["event"] == "block"]
            return len(block_events) + 1

        # Currently blocked → escalate
        if existing.is_active:
            return existing.strike + 1

        # Block expired but record still there → count it
        return existing.strike + 1

    # Unblock hooks
    def _on_unblock(self, ip: str) -> None:
        """Called by expiry daemon when a TTL expires."""
        result = self.firewall.unblock(ip)
        if result.ok:
            log.info(f"removed iptables rule for {ip}")
        else:
            log.warning(f"firewall.unblock({ip}) failed: {result.stderr}")

    def manual_unblock(self, ip: str) -> dict:
        """Remove a block immediately (for admin use)."""
        entry = self.state.get(ip)
        if entry is None:
            return {"ip": ip, "action": "not-found"}

        self.state.remove_block(ip)
        self.firewall.unblock(ip)
        log.info(f"manual unblock: {ip}")
        return {"ip": ip, "action": "unblocked"}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def active_blocks(self) -> list[dict]:
        return [e.to_dict() for e in self.state.active_blocks()]

    def is_blocked(self, ip: str) -> bool:
        return self.state.is_blocked(ip)

    def stats_summary(self) -> dict:
        s = self.stats.summary()
        s["active_blocks"] = len(self.state.active_blocks())
        s["total_records"] = self.state.count()["total_records"]
        return s

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Stop daemon and clean up."""
        if self.daemon:
            self.daemon.stop()
        log.info("BlockManager shutdown")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    from core.schema import Decision

    p = argparse.ArgumentParser(description="Block manager CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    block = sub.add_parser("block", help="block an IP")
    block.add_argument("ip")
    block.add_argument("--confidence", type=float, default=0.9)
    block.add_argument("--reason", default="manual")

    unblock = sub.add_parser("unblock")
    unblock.add_argument("ip")

    sub.add_parser("list")
    sub.add_parser("stats")

    args = p.parse_args()

    # Do not start the daemon for CLI usage (would keep the process alive)
    manager = BlockManager(start_daemon=False)

    if args.cmd == "block":
        d = Decision(
            ip=args.ip,
            action="block",
            confidence=args.confidence,
            reason=args.reason,
            ttl_seconds=60,
        )
        result = manager.enforce(d)
        print(json.dumps(result, indent=2, default=str))
    elif args.cmd == "unblock":
        result = manager.manual_unblock(args.ip)
        print(json.dumps(result, indent=2))
    elif args.cmd == "list":
        for b in manager.active_blocks():
            print(json.dumps(b, default=str))
    elif args.cmd == "stats":
        print(json.dumps(manager.stats_summary(), indent=2))

    manager.shutdown()