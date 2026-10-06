"""iptables wrapper for DDoS blocking.

Supports three backends:
    - iptables  (default, requires root/sudo)
    - nftables  (modern alternative)
    - null      (dry-run — logs commands without running them)

Modes:
    - Local:  runs iptables on the current machine
    - Remote: runs iptables on a remote host via SSH

Usage:
    fw = Firewall(backend="iptables", dry_run=True)
    fw.block("1.2.3.4")
    fw.unblock("1.2.3.4")
    fw.list_blocked()
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.logging import get_logger

log = get_logger("blocker.firewall")

# Where we track blocked IPs locally (to remove them later)
BLOCKED_FILE = Path("data/firewall_blocked.txt")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FirewallResult:
    ok: bool
    ip: str
    action: str           # "block" | "unblock"
    command: str
    stderr: str = ""
    elapsed: float = 0.0


@dataclass
class FirewallConfig:
    backend: str = "iptables"       # "iptables" | "nftables" | "null"
    dry_run: bool = False
    remote_host: str | None = None  # e.g., "naveen@192.168.122.30"
    remote_sudo: bool = True
    chain: str = "INPUT"
    rule_target: str = "DROP"       # "DROP" | "REJECT"
    use_sudo: bool = True


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------

class Firewall:
    """iptables/nftables wrapper with dry-run and remote support."""

    def __init__(self, config: FirewallConfig | None = None) -> None:
        self.cfg = config or FirewallConfig()

        # Track blocked IPs in memory (persistent file also)
        self._blocked: set[str] = set()
        self._load_blocked()

        # Verify backend binary exists
        if not self.cfg.dry_run:
            self._check_backend()

        log.info(f"Firewall ready: backend={self.cfg.backend} "
                 f"dry_run={self.cfg.dry_run} "
                 f"remote={self.cfg.remote_host or 'local'}")

    # ------------------------------------------------------------------
    # Setup checks
    # ------------------------------------------------------------------
    def _check_backend(self) -> None:
        if self.cfg.backend == "iptables":
            binary = shutil.which("iptables")
            if not binary:
                raise RuntimeError("iptables not found on PATH")
        elif self.cfg.backend == "nftables":
            binary = shutil.which("nft")
            if not binary:
                raise RuntimeError("nft not found on PATH")
        elif self.cfg.backend == "null":
            pass
        else:
            raise ValueError(f"unknown backend: {self.cfg.backend}")

    # ------------------------------------------------------------------
    # Persistence of blocked set
    # ------------------------------------------------------------------
    def _load_blocked(self) -> None:
        if BLOCKED_FILE.exists():
            with BLOCKED_FILE.open() as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self._blocked.add(line)
            log.info(f"loaded {len(self._blocked)} previously blocked IPs")

    def _save_blocked(self) -> None:
        BLOCKED_FILE.parent.mkdir(parents=True, exist_ok=True)
        with BLOCKED_FILE.open("w") as f:
            f.write(f"# firewall_blocked.txt — auto-generated\n")
            f.write(f"# last update: {time.ctime()}\n")
            for ip in sorted(self._blocked):
                f.write(f"{ip}\n")

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------
    def _build_command(self, args: list[str]) -> str:
        """Build the full command string (sudo + remote + binary)."""
        if self.cfg.backend == "iptables":
            base = ["iptables"]
        elif self.cfg.backend == "nftables":
            base = ["nft"]
        else:
            base = ["echo"]

        cmd_parts = base + args

        if self.cfg.use_sudo and self.cfg.backend != "null":
            cmd_parts = ["sudo", "-n"] + cmd_parts    # -n = non-interactive

        if self.cfg.remote_host:
            # Wrap in SSH
            ssh = ["ssh", self.cfg.remote_host]
            if self.cfg.remote_sudo:
                ssh += ["sudo", "-n"]
                cmd_parts = cmd_parts[2:]  # drop local sudo, remote sudo already added
            ssh += cmd_parts
            return " ".join(ssh)

        return " ".join(cmd_parts)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _run(self, args: list[str]) -> FirewallResult:
        """Run a firewall command. Returns result."""
        cmd = self._build_command(args)
        t0 = time.time()

        if self.cfg.dry_run or self.cfg.backend == "null":
            log.info(f"[DRY-RUN] {cmd}")
            return FirewallResult(
                ok=True, ip="", action="dry-run",
                command=cmd, elapsed=time.time() - t0,
            )

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10,
            )
            elapsed = time.time() - t0
            ok = result.returncode == 0

            if not ok:
                log.warning(f"command failed (rc={result.returncode}): {cmd}")
                log.warning(f"stderr: {result.stderr.strip()}")

            return FirewallResult(
                ok=ok, ip="", action="exec",
                command=cmd, stderr=result.stderr.strip(),
                elapsed=elapsed,
            )
        except subprocess.TimeoutExpired:
            log.error(f"command timed out: {cmd}")
            return FirewallResult(
                ok=False, ip="", action="exec",
                command=cmd, stderr="timeout",
                elapsed=time.time() - t0,
            )
        except Exception as e:
            log.error(f"command error: {e}")
            return FirewallResult(
                ok=False, ip="", action="exec",
                command=cmd, stderr=str(e),
                elapsed=time.time() - t0,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def block(self, ip: str) -> FirewallResult:
        """Add an iptables rule to drop traffic from `ip`."""
        if ip in self._blocked:
            log.info(f"already blocked: {ip}")
            return FirewallResult(
                ok=True, ip=ip, action="block",
                command="(already blocked)",
            )

        if self.cfg.backend == "iptables":
            args = [
                "-I", self.cfg.chain,
                "-s", ip,
                "-j", self.cfg.rule_target,
            ]
        elif self.cfg.backend == "nftables":
            args = [
                "add", "rule", "ip", "filter", "input",
                "ip", "saddr", ip, "drop",
            ]
        else:
            args = ["[null]", ip]

        result = self._run(args)

        if result.ok:
            self._blocked.add(ip)
            self._save_blocked()
            log.info(f"BLOCKED {ip}  ({result.elapsed*1000:.0f}ms)")

        result.ip = ip
        result.action = "block"
        return result

    def unblock(self, ip: str) -> FirewallResult:
        """Remove the iptables rule for `ip`."""
        if ip not in self._blocked:
            log.info(f"not currently blocked: {ip}")
            return FirewallResult(
                ok=True, ip=ip, action="unblock",
                command="(not blocked)",
            )

        if self.cfg.backend == "iptables":
            args = [
                "-D", self.cfg.chain,
                "-s", ip,
                "-j", self.cfg.rule_target,
            ]
        elif self.cfg.backend == "nftables":
            args = ["delete", "rule", "ip", "filter", "input",
                    "ip", "saddr", ip, "drop"]
        else:
            args = ["[null]", ip]

        result = self._run(args)

        if result.ok:
            self._blocked.discard(ip)
            self._save_blocked()
            log.info(f"UNBLOCKED {ip}")

        result.ip = ip
        result.action = "unblock"
        return result

    def is_blocked(self, ip: str) -> bool:
        return ip in self._blocked

    def list_blocked(self) -> list[str]:
        """Return the list of IPs we've blocked (from our local tracking)."""
        return sorted(self._blocked)

    def list_iptables(self) -> list[str]:
        """Ask iptables for the actual rules (ground truth)."""
        if self.cfg.backend != "iptables":
            return self.list_blocked()

        args = ["-L", self.cfg.chain, "-n", "--line-numbers"]
        result = self._run(args)
        if not result.ok:
            return []
        return result.command.splitlines() if result.command else []

    def flush_all(self) -> FirewallResult:
        """Remove ALL rules we've added. Dangerous — for tests only."""
        log.warning(f"flushing {len(self._blocked)} blocks")
        removed = 0
        for ip in list(self._blocked):
            r = self.unblock(ip)
            if r.ok:
                removed += 1
        return FirewallResult(
            ok=True, ip="*", action="flush",
            command=f"removed {removed} rules",
        )

    def count(self) -> int:
        return len(self._blocked)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def firewall_from_config(cfg: dict) -> Firewall:
    """Build a Firewall from the loaded config.yaml."""
    blocker_cfg = cfg.get("blocker", {})
    fw_cfg = FirewallConfig(
        backend=blocker_cfg.get("backend", "iptables"),
        dry_run=blocker_cfg.get("dry_run", False),
        remote_host=blocker_cfg.get("remote_host"),
        chain=blocker_cfg.get("chain", "INPUT"),
        rule_target=blocker_cfg.get("rule_target", "DROP"),
    )
    return Firewall(fw_cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Firewall CLI")
    p.add_argument("--dry-run", action="store_true",
                   help="log commands without executing")
    p.add_argument("--remote", default=None,
                   help="remote host (user@ip) — empty for local")
    p.add_argument("--backend", default="iptables",
                   choices=["iptables", "nftables", "null"])

    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("block")
    b.add_argument("ip")

    u = sub.add_parser("unblock")
    u.add_argument("ip")

    sub.add_parser("list")
    sub.add_parser("count")
    sub.add_parser("flush")

    args = p.parse_args()

    fw_cfg = FirewallConfig(
        backend=args.backend,
        dry_run=args.dry_run,
        remote_host=args.remote,
    )
    fw = Firewall(fw_cfg)

    if args.cmd == "block":
        r = fw.block(args.ip)
        print(f"block {args.ip}: ok={r.ok} cmd={r.command}")
    elif args.cmd == "unblock":
        r = fw.unblock(args.ip)
        print(f"unblock {args.ip}: ok={r.ok} cmd={r.command}")
    elif args.cmd == "list":
        for ip in fw.list_blocked():
            print(ip)
    elif args.cmd == "count":
        print(fw.count())
    elif args.cmd == "flush":
        r = fw.flush_all()
        print(r.command)