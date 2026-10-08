"""iptables wrapper for DDoS blocking, with ASN expansion via ipset.

Backends:
    - iptables  (default, requires root/sudo)
    - nftables  (modern alternative)
    - null      (dry-run)

Modes:
    - Local:  runs on the current machine
    - Remote: runs on a remote host via SSH

Block granularity:
    - block(ip)       → single IP or CIDR
    - block_asn(asn)  → ASN → prefixes (via ipset if enabled)
    - unblock(ip)
    - unblock_asn(asn)

Safety:
    - max_asn_prefixes  refuse ASNs announcing more than N prefixes
    - never_block_asns  hardcoded protection for critical ASNs
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from core.logging import get_logger

log = get_logger("blocker.firewall")

BLOCKED_FILE = Path("data/firewall_blocked.txt")
ASN_INDEX_FILE = Path("data/firewall_asn_index.json")


# ASNs we will never block — protects critical infrastructure
NEVER_BLOCK_ASNS: set[int] = {
    15169,     # Google
    13335,     # Cloudflare
    20940,     # Akamai
    16509,     # Amazon AWS
    14618,     # Amazon AWS
    14061,     # DigitalOcean
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FirewallResult:
    ok: bool
    ip: str
    action: str
    command: str
    stderr: str = ""
    elapsed: float = 0.0
    prefixes: list[str] | None = None


@dataclass
class FirewallConfig:
    backend: str = "iptables"
    dry_run: bool = False
    remote_host: str | None = None
    remote_sudo: bool = True
    chain: str = "INPUT"
    rule_target: str = "DROP"
    use_sudo: bool = True

    # ASN
    max_asn_prefixes: int = 5000
    asn_db_path: str = "data/ipasn.dat"
    use_ipset: bool = False


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------

class Firewall:
    """iptables/nftables wrapper with dry-run, remote, ASN, and ipset support."""

    def __init__(self, config: FirewallConfig | None = None) -> None:
        self.cfg = config or FirewallConfig()

        self._blocked: set[str] = set()
        self._asn_index: dict[int, list[str]] = {}
        self._load_blocked()

        self._pyasn_db = None

        if not self.cfg.dry_run:
            self._check_backend()

        log.info(
            f"Firewall ready: backend={self.cfg.backend} "
            f"dry_run={self.cfg.dry_run} "
            f"remote={self.cfg.remote_host or 'local'} "
            f"use_ipset={self.cfg.use_ipset} "
            f"max_asn_prefixes={self.cfg.max_asn_prefixes}"
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _check_backend(self) -> None:
        if self.cfg.backend == "iptables":
            if not shutil.which("iptables"):
                raise RuntimeError("iptables not found on PATH")
        elif self.cfg.backend == "nftables":
            if not shutil.which("nft"):
                raise RuntimeError("nft not found on PATH")
        elif self.cfg.backend == "null":
            pass
        else:
            raise ValueError(f"unknown backend: {self.cfg.backend}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load_blocked(self) -> None:
        if BLOCKED_FILE.exists():
            with BLOCKED_FILE.open() as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self._blocked.add(line)
            log.info(f"loaded {len(self._blocked)} previously blocked IPs")

        if ASN_INDEX_FILE.exists():
            try:
                import json
                with ASN_INDEX_FILE.open() as f:
                    raw = json.load(f)
                self._asn_index = {int(k): v for k, v in raw.items()}
                log.info(f"loaded ASN index for {len(self._asn_index)} ASNs")
            except Exception as e:
                log.warning(f"failed to load ASN index: {e}")

    def _save_blocked(self) -> None:
        BLOCKED_FILE.parent.mkdir(parents=True, exist_ok=True)
        with BLOCKED_FILE.open("w") as f:
            f.write("# firewall_blocked.txt — auto-generated\n")
            f.write(f"# last update: {time.ctime()}\n")
            for ip in sorted(self._blocked):
                f.write(f"{ip}\n")

        if self._asn_index:
            import json
            with ASN_INDEX_FILE.open("w") as f:
                json.dump({str(k): v for k, v in self._asn_index.items()}, f)

    # ------------------------------------------------------------------
    # pyasn
    # ------------------------------------------------------------------
    def _load_pyasn(self) -> bool:
        if self._pyasn_db is not None:
            return True
        db_path = Path(self.cfg.asn_db_path)
        if not db_path.exists():
            log.warning(f"pyasn db not found at {db_path}")
            return False
        try:
            import pyasn
            self._pyasn_db = pyasn.pyasn(str(db_path))
            log.info(f"pyasn loaded: {db_path}")
            return True
        except ImportError:
            log.warning("pyasn not installed")
            return False
        except Exception as e:
            log.warning(f"pyasn failed: {e}")
            return False

    def asn_to_prefixes(self, asn: int) -> list[str]:
        if not self._load_pyasn():
            return []
        try:
            gen = self._pyasn_db.get_as_prefixes(asn)
            if gen is None:
                return []
            prefixes = [p for p in gen if ":" not in p]
            log.info(f"AS{asn} → {len(prefixes)} IPv4 prefixes")
            return prefixes
        except Exception as e:
            log.warning(f"get_as_prefixes({asn}) failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------
    def _build_command(self, args: list[str]) -> str:
        if self.cfg.backend == "iptables":
            base = ["iptables"]
        elif self.cfg.backend == "nftables":
            base = ["nft"]
        else:
            base = ["echo"]

        cmd_parts = base + args
        if self.cfg.use_sudo and self.cfg.backend != "null":
            cmd_parts = ["sudo", "-n"] + cmd_parts

        if self.cfg.remote_host:
            ssh = ["ssh", self.cfg.remote_host]
            if self.cfg.remote_sudo:
                ssh += ["sudo", "-n"]
                cmd_parts = cmd_parts[2:]
            ssh += cmd_parts
            return " ".join(ssh)
        return " ".join(cmd_parts)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _run(self, args: list[str]) -> FirewallResult:
        cmd = self._build_command(args)
        t0 = time.time()

        if self.cfg.dry_run or self.cfg.backend == "null":
            log.info(f"[DRY-RUN] {cmd}")
            return FirewallResult(ok=True, ip="", action="dry-run",
                                  command=cmd, elapsed=time.time() - t0)

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                    text=True, timeout=10)
            elapsed = time.time() - t0
            ok = result.returncode == 0
            if not ok:
                log.warning(f"command failed (rc={result.returncode}): {cmd}")
                log.warning(f"stderr: {result.stderr.strip()}")
            return FirewallResult(ok=ok, ip="", action="exec", command=cmd,
                                  stderr=result.stderr.strip(), elapsed=elapsed)
        except subprocess.TimeoutExpired:
            log.error(f"command timed out: {cmd}")
            return FirewallResult(ok=False, ip="", action="exec",
                                  command=cmd, stderr="timeout",
                                  elapsed=time.time() - t0)
        except Exception as e:
            log.error(f"command error: {e}")
            return FirewallResult(ok=False, ip="", action="exec",
                                  command=cmd, stderr=str(e),
                                  elapsed=time.time() - t0)

    def _run_ipset(self, args: list[str],
                   stdin: str | None = None) -> FirewallResult:
        """Run ipset locally or remotely. Supports stdin (for `restore`)."""
        t0 = time.time()

        if self.cfg.remote_host:
            remote = "sudo -n ipset " + " ".join(args)
            cmd = ["ssh", self.cfg.remote_host, remote]
            display = " ".join(cmd)
        else:
            cmd = ["sudo", "-n", "ipset"] + args
            display = " ".join(cmd)

        if self.cfg.dry_run:
            log.info(f"[DRY-RUN] {display}")
            return FirewallResult(ok=True, ip="", action="ipset",
                                  command=display, elapsed=0)

        try:
            r = subprocess.run(cmd, input=stdin, capture_output=True,
                               text=True, timeout=60)
            return FirewallResult(
                ok=r.returncode == 0, ip="", action="ipset",
                command=display, stderr=r.stderr.strip(),
                elapsed=time.time() - t0,
            )
        except subprocess.TimeoutExpired:
            return FirewallResult(ok=False, ip="", action="ipset",
                                  command=display, stderr="timeout",
                                  elapsed=time.time() - t0)
        except Exception as e:
            return FirewallResult(ok=False, ip="", action="ipset",
                                  command=display, stderr=str(e),
                                  elapsed=time.time() - t0)

    # ------------------------------------------------------------------
    # IP / CIDR
    # ------------------------------------------------------------------
    def block(self, ip: str) -> FirewallResult:
        if ip in self._blocked:
            log.info(f"already blocked: {ip}")
            return FirewallResult(ok=True, ip=ip, action="block",
                                  command="(already blocked)")

        if self.cfg.backend == "iptables":
            args = ["-I", self.cfg.chain, "-s", ip, "-j", self.cfg.rule_target]
        elif self.cfg.backend == "nftables":
            args = ["add", "rule", "ip", "filter", "input",
                    "ip", "saddr", ip, "drop"]
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
        if ip not in self._blocked:
            log.info(f"not currently blocked: {ip}")
            return FirewallResult(ok=True, ip=ip, action="unblock",
                                  command="(not blocked)")

        if self.cfg.backend == "iptables":
            args = ["-D", self.cfg.chain, "-s", ip, "-j", self.cfg.rule_target]
        elif self.cfg.backend == "nftables":
            args = ["delete", "rule", "ip", "filter", "input",
                    "ip", "saddr", ip, "drop"]
        else:
            args = ["[null]", ip]

        result = self._run(args)

        # Treat "rule doesn't exist" as success — the goal is met
        missing = (
            not result.ok
            and ("Bad rule" in result.stderr
                 or "does a matching rule exist" in result.stderr)
        )
        if result.ok or missing:
            if missing:
                log.info(f"UNBLOCK {ip}: rule already absent")
            self._blocked.discard(ip)
            self._save_blocked()
            if result.ok:
                log.info(f"UNBLOCKED {ip}")

        result.ip = ip
        result.action = "unblock"
        return result

    # ------------------------------------------------------------------
    # ASN — dispatcher
    # ------------------------------------------------------------------
    def block_asn(self, asn: int) -> FirewallResult:
        if self.cfg.use_ipset:
            return self._block_asn_ipset(asn)
        return self._block_asn_iptables(asn)

    def unblock_asn(self, asn: int) -> FirewallResult:
        if self.cfg.use_ipset:
            return self._unblock_asn_ipset(asn)
        return self._unblock_asn_iptables(asn)

    def _asn_safety_check(self, asn: int) -> FirewallResult | None:
        """Return a FirewallResult on refusal, or None if safe to proceed."""
        if asn in NEVER_BLOCK_ASNS:
            log.error(f"AS{asn} is on the never-block list; refusing")
            return FirewallResult(
                ok=False, ip=f"AS{asn}", action="block_asn",
                command="(refused: never-block list)",
                stderr=f"AS{asn} in NEVER_BLOCK_ASNS",
            )
        return None

    # ------------------------------------------------------------------
    # ASN — ipset path
    # ------------------------------------------------------------------
    def _block_asn_ipset(self, asn: int) -> FirewallResult:
        t0 = time.time()
        refusal = self._asn_safety_check(asn)
        if refusal:
            return refusal

        prefixes = self.asn_to_prefixes(asn)
        if not prefixes:
            return FirewallResult(ok=False, ip=f"AS{asn}", action="block_asn",
                                  command=f"AS{asn}",
                                  stderr="no IPv4 prefixes found",
                                  elapsed=time.time() - t0)

        if len(prefixes) > self.cfg.max_asn_prefixes:
            log.error(f"AS{asn}: {len(prefixes)} prefixes > "
                      f"cap {self.cfg.max_asn_prefixes}")
            return FirewallResult(
                ok=False, ip=f"AS{asn}", action="block_asn",
                command=f"AS{asn}: {len(prefixes)} prefixes",
                stderr=f"too many prefixes ({len(prefixes)} > "
                       f"{self.cfg.max_asn_prefixes})",
                elapsed=time.time() - t0,
            )

        set_name = f"ddos_as{asn}"

        # 1. create (idempotent)
        self._run_ipset(["create", set_name, "hash:net", "-exist"])

        # 2. bulk load
        payload = "\n".join(f"add {set_name} {p} -exist" for p in prefixes)
        load = self._run_ipset(["restore", "-exist"], stdin=payload)
        if not load.ok:
            log.error(f"ipset restore failed for AS{asn}: {load.stderr}")
            return FirewallResult(ok=False, ip=f"AS{asn}",
                                  action="block_asn",
                                  command=load.command,
                                  stderr=load.stderr,
                                  elapsed=time.time() - t0)

        # 3. single iptables rule
        if self.cfg.backend == "iptables":
            args = ["-I", self.cfg.chain,
                    "-m", "set", "--match-set", set_name, "src",
                    "-j", self.cfg.rule_target]
        else:
            args = ["[null]", set_name]
        self._run(args)

        self._asn_index[asn] = list(prefixes)
        self._save_blocked()

        elapsed = time.time() - t0
        log.info(f"BLOCKED AS{asn} via ipset "
                 f"({len(prefixes)} prefixes, {elapsed:.2f}s)")
        return FirewallResult(
            ok=True, ip=f"AS{asn}", action="block_asn",
            command=f"ipset {set_name}: {len(prefixes)} prefixes",
            prefixes=prefixes, elapsed=elapsed,
        )

    def _unblock_asn_ipset(self, asn: int) -> FirewallResult:
        t0 = time.time()
        set_name = f"ddos_as{asn}"

        if self.cfg.backend == "iptables":
            self._run([
                "-D", self.cfg.chain,
                "-m", "set", "--match-set", set_name, "src",
                "-j", self.cfg.rule_target,
            ])
        self._run_ipset(["destroy", set_name])

        prefixes = self._asn_index.pop(asn, [])
        self._save_blocked()

        elapsed = time.time() - t0
        log.info(f"UNBLOCKED AS{asn} via ipset ({elapsed:.2f}s)")
        return FirewallResult(
            ok=True, ip=f"AS{asn}", action="unblock_asn",
            command=f"destroy {set_name}",
            prefixes=prefixes, elapsed=elapsed,
        )

    # ------------------------------------------------------------------
    # ASN — legacy iptables path
    # ------------------------------------------------------------------
    def _block_asn_iptables(self, asn: int) -> FirewallResult:
        t0 = time.time()
        refusal = self._asn_safety_check(asn)
        if refusal:
            return refusal

        prefixes = self.asn_to_prefixes(asn)
        if not prefixes:
            return FirewallResult(ok=False, ip=f"AS{asn}", action="block_asn",
                                  command=f"AS{asn}",
                                  stderr="no IPv4 prefixes found",
                                  elapsed=time.time() - t0)

        if len(prefixes) > self.cfg.max_asn_prefixes:
            log.error(f"AS{asn}: {len(prefixes)} prefixes > "
                      f"cap {self.cfg.max_asn_prefixes}")
            return FirewallResult(
                ok=False, ip=f"AS{asn}", action="block_asn",
                command=f"AS{asn}: {len(prefixes)} prefixes",
                stderr=f"too many prefixes ({len(prefixes)} > "
                       f"{self.cfg.max_asn_prefixes})",
                elapsed=time.time() - t0,
            )

        to_add = [p for p in prefixes if p not in self._blocked]
        already = len(prefixes) - len(to_add)

        log.info(f"block_asn(AS{asn}): {len(prefixes)} prefixes "
                 f"({len(to_add)} new, {already} existing)")

        results = [self.block(p) for p in to_add]
        added_ok = sum(1 for r in results if r.ok)

        self._asn_index[asn] = list(prefixes)
        self._save_blocked()

        elapsed = time.time() - t0
        log.info(f"BLOCKED AS{asn}: {added_ok}/{len(to_add)} new "
                 f"({elapsed:.2f}s)")
        return FirewallResult(
            ok=added_ok == len(to_add),
            ip=f"AS{asn}", action="block_asn",
            command=f"blocked {added_ok} new + {already} existing prefixes",
            prefixes=prefixes, elapsed=elapsed,
        )

    def _unblock_asn_iptables(self, asn: int) -> FirewallResult:
        t0 = time.time()
        prefixes = self._asn_index.get(asn)
        if not prefixes:
            prefixes = self.asn_to_prefixes(asn)
            if not prefixes:
                return FirewallResult(ok=False, ip=f"AS{asn}",
                                      action="unblock_asn",
                                      command=f"AS{asn}",
                                      stderr="not in ASN index",
                                      elapsed=time.time() - t0)

        log.info(f"unblock_asn(AS{asn}): removing {len(prefixes)} prefixes")
        removed = 0
        for p in prefixes:
            if self.unblock(p).ok:
                removed += 1

        self._asn_index.pop(asn, None)
        self._save_blocked()

        elapsed = time.time() - t0
        log.info(f"UNBLOCKED AS{asn}: {removed}/{len(prefixes)} ({elapsed:.2f}s)")
        return FirewallResult(
            ok=True, ip=f"AS{asn}", action="unblock_asn",
            command=f"removed {removed} prefixes",
            prefixes=prefixes, elapsed=elapsed,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def is_blocked(self, ip: str) -> bool:
        return ip in self._blocked

    def list_blocked(self) -> list[str]:
        return sorted(self._blocked)

    def list_blocked_asns(self) -> dict[int, list[str]]:
        return dict(self._asn_index)

    def list_iptables(self) -> list[str]:
        if self.cfg.backend != "iptables":
            return self.list_blocked()
        args = ["-L", self.cfg.chain, "-n", "--line-numbers"]
        result = self._run(args)
        if not result.ok:
            return []
        return result.command.splitlines() if result.command else []

    def flush_all(self) -> FirewallResult:
        log.warning(f"flushing {len(self._blocked)} blocks")
        removed = 0
        for ip in list(self._blocked):
            if self.unblock(ip).ok:
                removed += 1
        # Also destroy any ipsets we created
        for asn in list(self._asn_index.keys()):
            if self.cfg.use_ipset:
                self._run_ipset(["destroy", f"ddos_as{asn}"])
        self._asn_index.clear()
        self._save_blocked()
        return FirewallResult(ok=True, ip="*", action="flush",
                              command=f"removed {removed} rules")

    def count(self) -> int:
        return len(self._blocked)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def firewall_from_config(cfg: dict) -> Firewall:
    b = cfg.get("blocker", {})
    fw_cfg = FirewallConfig(
        backend=b.get("backend", "iptables"),
        dry_run=b.get("dry_run", False),
        remote_host=b.get("remote_host"),
        chain=b.get("chain", "INPUT"),
        rule_target=b.get("rule_target", "DROP"),
        max_asn_prefixes=int(b.get("max_asn_prefixes", 5000)),
        asn_db_path=b.get("asn_db_path", "data/ipasn.dat"),
        use_ipset=bool(b.get("use_ipset", False)),
    )
    return Firewall(fw_cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Firewall CLI")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--remote", default=None)
    p.add_argument("--backend", default="iptables",
                   choices=["iptables", "nftables", "null"])
    p.add_argument("--ipset", action="store_true",
                   help="enable ipset path for ASN blocks")

    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("block"); b.add_argument("ip")
    u = sub.add_parser("unblock"); u.add_argument("ip")
    ba = sub.add_parser("block-asn"); ba.add_argument("asn", type=int)
    ua = sub.add_parser("unblock-asn"); ua.add_argument("asn", type=int)
    ex = sub.add_parser("expand-asn"); ex.add_argument("asn", type=int)

    sub.add_parser("list")
    sub.add_parser("list-asns")
    sub.add_parser("count")
    sub.add_parser("flush")

    args = p.parse_args()

    fw_cfg = FirewallConfig(
        backend=args.backend, dry_run=args.dry_run,
        remote_host=args.remote, use_ipset=args.ipset,
    )
    fw = Firewall(fw_cfg)

    if args.cmd == "block":
        r = fw.block(args.ip)
        print(f"block {args.ip}: ok={r.ok} cmd={r.command}")
    elif args.cmd == "unblock":
        r = fw.unblock(args.ip)
        print(f"unblock {args.ip}: ok={r.ok} cmd={r.command}")
    elif args.cmd == "block-asn":
        r = fw.block_asn(args.asn)
        print(f"block-asn AS{args.asn}: ok={r.ok} cmd={r.command}")
        if r.stderr:
            print(f"  stderr: {r.stderr}")
    elif args.cmd == "unblock-asn":
        r = fw.unblock_asn(args.asn)
        print(f"unblock-asn AS{args.asn}: ok={r.ok} cmd={r.command}")
    elif args.cmd == "expand-asn":
        prefixes = fw.asn_to_prefixes(args.asn)
        print(f"AS{args.asn}: {len(prefixes)} IPv4 prefixes")
        for p in prefixes[:20]:
            print(f"  {p}")
        if len(prefixes) > 20:
            print(f"  ... +{len(prefixes) - 20} more")
    elif args.cmd == "list":
        for ip in fw.list_blocked():
            print(ip)
    elif args.cmd == "list-asns":
        for asn, prefixes in fw.list_blocked_asns().items():
            print(f"AS{asn}: {len(prefixes)} prefixes")
    elif args.cmd == "count":
        print(f"ips={fw.count()}  asns={len(fw.list_blocked_asns())}")
    elif args.cmd == "flush":
        print(fw.flush_all().command)