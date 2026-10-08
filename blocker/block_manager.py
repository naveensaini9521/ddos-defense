"""Orchestrator — turns ML Decisions into actual blocks.

Coordinates:
    - whitelist.py  (safety check)
    - state.py or redis_state.py (persistence)
    - expiry.py     (TTL + backoff + daemon)
    - firewall.py   (iptables, local or remote; supports ASN expansion)

Usage:
    from blocker.block_manager import BlockManager

    manager = BlockManager()
    manager.enforce(decision)
    manager.enforce_many(decisions)
    manager.shutdown()
"""
from __future__ import annotations
import os

import ipaddress
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.config_loader import load as load_config
from core.logging import get_logger
from core.schema import Decision
from blocker.expiry import ExpiryDaemon, ExpiryPolicy, PERMANENT, format_ttl
from blocker.firewall import Firewall, FirewallConfig, FirewallResult
from blocker.state import BlockState
from blocker.whitelist import Whitelist

log = get_logger("blocker.block_manager")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class ManagerStats:
    decisions_seen: int = 0
    blocks_enforced: int = 0
    asn_blocks_enforced: int = 0
    whitelist_skips: int = 0
    allow_skips: int = 0
    throttle_skips: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def summary(self) -> dict:
        return {
            "decisions_seen": self.decisions_seen,
            "blocks_enforced": self.blocks_enforced,
            "asn_blocks_enforced": self.asn_blocks_enforced,
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
        state=None,
        firewall: Firewall | None = None,
        whitelist: Whitelist | None = None,
        policy: ExpiryPolicy | None = None,
        start_daemon: bool = True,
    ) -> None:
        self.cfg = load_config(config_path) if config_path else load_config()

        self.whitelist = whitelist or Whitelist.from_config(self.cfg)
        self.state = state or self._build_state()
        self.firewall = firewall or self._firewall_from_config()
        self.policy = policy or self._policy_from_config()

        self.stats = ManagerStats()

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
            f"state={type(self.state).__name__} "
            f"backend={self.firewall.cfg.backend} "
            f"dry_run={self.firewall.cfg.dry_run} "
            f"whitelist={len(self.whitelist)} entries"
        )

    # ------------------------------------------------------------------
    # Config-driven construction
    # ------------------------------------------------------------------
    def _build_state(self):
        """Prefer Redis when config.redis.host is set; fall back to SQLite."""
        redis_cfg = dict(self.cfg.get("redis", {}))
        if os.environ.get("REDIS_HOST"):
            redis_cfg["host"] = os.environ["REDIS_HOST"]
        if os.environ.get("REDIS_PORT"):
            redis_cfg["port"] = os.environ["REDIS_PORT"]
        if redis_cfg.get("host"):
            try:
                from blocker.redis_state import RedisBlockState
                log.info(
                    f"using RedisBlockState "
                    f"({redis_cfg['host']}:{redis_cfg.get('port', 6379)})"
                )
                return RedisBlockState(
                    host=redis_cfg["host"],
                    port=int(redis_cfg.get("port", 6379)),
                    db=int(redis_cfg.get("db", 0)),
                    prefix=redis_cfg.get("prefix", "ddos:"),
                )
            except Exception as e:
                log.warning(f"Redis unavailable ({e}); falling back to SQLite")
        return BlockState()

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
            use_ipset=bool(b.get("use_ipset", False)),
            max_asn_prefixes=int(b.get("max_asn_prefixes", 500)),
            asn_db_path=b.get("asn_db_path", "data/ipasn.dat"),
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
    # Enforce
    # ------------------------------------------------------------------
    def enforce(self, decision: Decision) -> dict:
        """Take action on a single Decision. Returns a result dict."""
        self.stats.decisions_seen += 1

        ip = decision.ip
        action = decision.action
        scope = getattr(decision, "scope", "ip")

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

        # Whitelist — only meaningful for IP/CIDR scopes
        if scope in ("ip", "subnet") and self.whitelist.is_whitelisted(ip):
            self.stats.whitelist_skips += 1
            log.info(f"whitelist prevents blocking {ip}")
            return {"ip": ip, "action": "skipped", "reason": "whitelisted"}

        # Compute strike + TTL. Explicit TTL wins.
        strike = self._next_strike(ip)
        explicit_ttl = getattr(decision, "ttl_seconds", None)
        if explicit_ttl is not None and explicit_ttl > 0:
            ttl = float(explicit_ttl)
        else:
            ttl = self.policy.ttl_for(strike)

        # Save to state
        entry = self.state.add_block(
            ip=ip,
            ttl=ttl,
            reason=decision.reason,
            confidence=decision.confidence,
            model=self._model_from_reason(decision.reason),
            **self._state_context(ip, decision),
        )

        # Dispatch to firewall based on scope
        if scope == "asn":
            fw_result = self._enforce_asn(decision)
        else:
            fw_result = self.firewall.block(ip)

        if fw_result.ok:
            if scope == "asn":
                self.stats.asn_blocks_enforced += 1
            self.stats.blocks_enforced += 1
            log.info(
                f"ENFORCED block on {ip}  scope={scope}  "
                f"ttl={format_ttl(ttl)}  strike={strike}  "
                f"conf={decision.confidence:.2f}"
            )
        else:
            self.stats.errors += 1
            log.error(f"firewall block failed for {ip}: {fw_result.stderr}")

        return {
            "ip": ip,
            "action": "blocked",
            "scope": scope,
            "strike": strike,
            "ttl": ttl,
            "ttl_human": format_ttl(ttl),
            "confidence": decision.confidence,
            "fw_ok": fw_result.ok,
            "fw_command": fw_result.command,
            "prefixes": fw_result.prefixes,
        }

    def _enforce_asn(self, decision: Decision) -> FirewallResult:
        """ASN-scoped enforcement: expand to prefixes and block each."""
        asn = getattr(decision, "asn", None)
        if asn is None:
            # Try to parse from the target string (e.g. "AS9009")
            target = decision.ip
            if target.upper().startswith("AS"):
                try:
                    asn = int(target[2:])
                except ValueError:
                    pass

        if asn is None:
            log.error(f"ASN-scoped decision but no ASN found: {decision.ip}")
            return FirewallResult(
                ok=False, ip=decision.ip, action="block_asn",
                command="(no ASN in decision)",
                stderr="scope=asn but asn attribute missing",
            )

        return self.firewall.block_asn(int(asn))

    def enforce_many(self, decisions: list[Decision]) -> list[dict]:
        return [self.enforce(d) for d in decisions]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _model_from_reason(reason: str) -> str:
        if not reason:
            return "unknown"
        first = reason.split(" ")[0]
        if first.startswith("model="):
            return first.replace("model=", "") or "unknown"
        return "unknown"

    def _state_context(self, ip: str, decision) -> dict:
        """Subnet/ASN/JA3/scope kwargs for state storage (Redis-only)."""
        from blocker.redis_state import RedisBlockState
        if not isinstance(self.state, RedisBlockState):
            return {}

        ctx: dict = {}
        scope = getattr(decision, "scope", None)
        if scope:
            ctx["scope"] = scope

        subnet = getattr(decision, "subnet", None)
        asn = getattr(decision, "asn", None)
        ja3 = getattr(decision, "ja3", None)

        # Fallback: /24 derived from a plain IPv4
        if not subnet and "/" not in ip and ":" not in ip:
            try:
                subnet = str(ipaddress.ip_network(f"{ip}/24", strict=False))
            except Exception:
                subnet = None

        if subnet:
            ctx["subnet"] = subnet
        if asn:
            ctx["asn"] = asn
        if ja3:
            ctx["ja3"] = ja3
        return ctx

    # ------------------------------------------------------------------
    # Strike
    # ------------------------------------------------------------------
    def _next_strike(self, ip: str) -> int:
        existing = self.state.get(ip)
        if existing is None:
            history = self.state.history(ip, limit=10)
            block_events = [h for h in history if h.get("event") == "block"]
            return len(block_events) + 1
        return existing.strike + 1

    # ------------------------------------------------------------------
    # Unblock hooks
    # ------------------------------------------------------------------
    def _on_unblock(self, ip: str) -> None:
        """Called by expiry daemon. Handles both IP and ASN scope."""
        # Redis stores the target string. If it's "ASnnnn", unblock the ASN.
        if ip.upper().startswith("AS"):
            try:
                asn = int(ip[2:])
            except ValueError:
                asn = None
            if asn is not None:
                result = self.firewall.unblock_asn(asn)
                if result.ok:
                    log.info(f"removed iptables rules for AS{asn}")
                else:
                    log.warning(f"unblock_asn(AS{asn}) failed: {result.stderr}")
                return

        result = self.firewall.unblock(ip)
        if result.ok:
            log.info(f"removed iptables rule for {ip}")
        else:
            log.warning(f"firewall.unblock({ip}) failed: {result.stderr}")

    def manual_unblock(self, ip: str) -> dict:
        entry = self.state.get(ip)
        if entry is None:
            return {"ip": ip, "action": "not-found"}

        self.state.remove_block(ip)

        if ip.upper().startswith("AS"):
            try:
                asn = int(ip[2:])
                self.firewall.unblock_asn(asn)
            except ValueError:
                self.firewall.unblock(ip)
        else:
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
        s["state_backend"] = type(self.state).__name__
        return s

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if self.daemon:
            self.daemon.stop()
        try:
            close = getattr(self.state, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
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

    block = sub.add_parser("block")
    block.add_argument("ip")
    block.add_argument("--confidence", type=float, default=0.9)
    block.add_argument("--reason", default="manual")
    block.add_argument("--ttl", type=int, default=60)

    block_asn = sub.add_parser("block-asn")
    block_asn.add_argument("asn", type=int)
    block_asn.add_argument("--ttl", type=int, default=300)

    unblock = sub.add_parser("unblock")
    unblock.add_argument("ip")

    sub.add_parser("list")
    sub.add_parser("stats")
    sub.add_parser("clear")

    args = p.parse_args()

    manager = BlockManager(start_daemon=False)

    if args.cmd == "block":
        d = Decision(
            ip=args.ip, action="block", confidence=args.confidence,
            reason=args.reason, ttl_seconds=args.ttl,
        )
        print(json.dumps(manager.enforce(d), indent=2, default=str))
    elif args.cmd == "block-asn":
        d = Decision(
            ip=f"AS{args.asn}", action="block", confidence=0.95,
            reason=f"manual_asn_block", ttl_seconds=args.ttl,
        )
        object.__setattr__(d, "asn", args.asn)
        object.__setattr__(d, "scope", "asn")
        print(json.dumps(manager.enforce(d), indent=2, default=str))
    elif args.cmd == "unblock":
        print(json.dumps(manager.manual_unblock(args.ip), indent=2))
    elif args.cmd == "list":
        for b in manager.active_blocks():
            print(json.dumps(b, default=str))
    elif args.cmd == "stats":
        print(json.dumps(manager.stats_summary(), indent=2))
    elif args.cmd == "clear":
        manager.state.clear()
        print("state cleared")

    manager.shutdown()