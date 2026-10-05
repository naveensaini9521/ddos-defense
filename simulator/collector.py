"""Data collection pipeline.

Runs an attack scenario against the VM while simultaneously capturing
the VM's logs, then labels every log line based on the scenario running.

Labels are assigned by INDEX position in the collected stream (not by
timestamp) to avoid clock-skew issues between host and VM.

Output: data/collected/ directory with:
    - <timestamp>_<scenario>.log        raw logs
    - <timestamp>_<scenario>.meta.json  metadata
    - <timestamp>_<scenario>.jsonl      labeled records

Usage:
    python3 -m simulator.collector --scenario http_flood \
        --target http://192.168.122.30/ \
        --vm-host naveen@192.168.122.30 --duration 30
"""
from __future__ import annotations

import json
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from collector.nginx_parser import ParsedLog, parse_line
from core.logging import get_logger
from simulator.attack_scenarios import SCENARIOS

log = get_logger("simulator.collector")

COLLECTED_DIR = Path("data/collected")


# ---------------------------------------------------------------------------
# Labeled record
# ---------------------------------------------------------------------------

@dataclass
class CollectedRecord:
    ts: float
    ip: str
    method: str
    path: str
    status: int
    bytes: int
    user_agent: str | None
    label: int
    phase: str
    scenario: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Remote log tailer (SSH)
# ---------------------------------------------------------------------------

class RemoteLogCollector:
    def __init__(self, host: str, log_path: str = "/var/log/nginx/access.log",
                 use_sudo: bool = False) -> None:
        self.host = host
        self.log_path = log_path
        self.use_sudo = use_sudo
        self.records: list[ParsedLog] = []
        self.raw_lines: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(f"log collector started → {self.host}:{self.log_path}")

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._thread:
            self._thread.join(timeout=5)
        log.info(f"log collector stopped: {len(self.records)} records")

    def count(self) -> int:
        with self._lock:
            return len(self.records)

    def _loop(self) -> None:
        path = self.log_path if self.log_path.startswith("/") else "/" + self.log_path
        tail_cmd = f"tail -F -n 0 {shlex.quote(path)}"
        if self.use_sudo:
            tail_cmd = f"sudo -n {tail_cmd}"
        remote_cmd = f"stdbuf -oL -eL {tail_cmd}"

        cmd = [
            "ssh", "-T",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            self.host, remote_cmd,
        ]

        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, universal_newlines=False,
        )

        def drain():
            p = self._proc
            if p and p.stderr:
                for _ in iter(p.stderr.readline, b""):
                    pass
        threading.Thread(target=drain, daemon=True).start()

        try:
            p = self._proc
            if p is None or p.stdout is None:
                return
            while not self._stop.is_set():
                raw = p.stdout.readline()
                if not raw:
                    break
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                except Exception:
                    continue
                if not line.strip():
                    continue
                with self._lock:
                    self.raw_lines.append(line)
                    rec = parse_line(line)
                    if rec:
                        self.records.append(rec)
        finally:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
            self._proc = None


# ---------------------------------------------------------------------------
# Normal traffic generator (for warmup/cooldown labeling)
# ---------------------------------------------------------------------------

class NormalTrafficGenerator:
    def __init__(self, target: str, rps: float = 3.0, timeout: float = 3.0) -> None:
        self.target = target.rstrip("/") + "/"
        self.rps = rps
        self.timeout = timeout
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(f"normal traffic generator started ({self.rps} rps)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        import random
        import requests
        interval = 1.0 / max(self.rps, 0.001)
        paths = ["/", "/api/data", "/health"]
        while not self._stop.is_set():
            try:
                path = random.choice(paths).lstrip("/")
                requests.get(self.target + path, timeout=self.timeout)
            except Exception:
                pass
            self._stop.wait(interval)


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

def drain_wait(
    log_col: RemoteLogCollector,
    quiet_seconds: float = 2.0,
    max_wait: float = 30.0,
) -> int:
    """Wait until the SSH tail stops producing new records.

    Uses a "quiet period" heuristic: if no new records arrive for
    `quiet_seconds`, assume we've caught up. Capped by `max_wait`.

    Returns the final record count.
    """
    start = time.time()
    last_count = log_col.count()
    last_change = time.time()

    while (time.time() - start) < max_wait:
        time.sleep(0.2)
        current = log_col.count()
        if current != last_count:
            last_count = current
            last_change = time.time()
        elif (time.time() - last_change) >= quiet_seconds:
            # quiet for long enough — we're drained
            break

    return log_col.count()

def remote_tcpdump(host: str, action: str, pcap_path: str,
                   timeout: float = 15.0) -> tuple[bool, str]:
    """Control tcpdump on the remote via the controller script.

    Args:
        host: ssh target (user@ip)
        action: "start" | "stop" | "status"
        pcap_path: remote path for the pcap file
        timeout: SSH command timeout

    Returns:
        (ok, message)
    """
    cmd = (
        f"sudo -n /usr/local/bin/tcpdump_control {action} "
        f"{shlex.quote(pcap_path)}"
    )
    try:
        r = subprocess.run(
            ["ssh", "-T",
             "-o", "StrictHostKeyChecking=no",
             "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=10",
             host, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        ok = r.returncode == 0
        msg = (r.stdout.strip() or r.stderr.strip() or "").strip()
        return ok, msg
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def remote_copy_pcap(host: str, remote_path: str,
                     local_path: Path) -> bool:
    """Copy a remote pcap to the local machine."""
    try:
        r = subprocess.run(
            ["scp", "-q",
             "-o", "StrictHostKeyChecking=no",
             "-o", "BatchMode=yes",
             f"{host}:{remote_path}", str(local_path)],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0 and local_path.exists()
    except Exception:
        return False

def collect_scenario(
    scenario: str,
    target: str,
    vm_host: str,
    duration: int = 30,
    warmup: int = 5,
    cooldown: int = 5,
    vm_log_path: str = "/var/log/nginx/access.log",
    scenario_kwargs: dict | None = None,
    normal_rps: float = 3.0,
    capture_packets: bool = True,
) -> Path:
    """Run one scenario and collect labeled records.

    Labels are assigned by INDEX position in the collected stream:
        - Records 0..warmup_end_idx            → normal  (warmup)
        - Records warmup_end_idx..attack_end_idx → attack
        - Records attack_end_idx..end          → normal  (cooldown)

    This avoids clock skew between host and VM.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    
    COLLECTED_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = COLLECTED_DIR / f"{timestamp}_{scenario}"
    
    log.info(f"=== collecting: {scenario} ===")
    log.info(f"  target:   {target}")
    log.info(f"  vm_host:  {vm_host}")
    log.info(f"  duration: warmup={warmup}s attack={duration}s cooldown={cooldown}s")
    log.info(f"  packet capture: {capture_packets}")

    # 0. Start tcpdump on the VM (if enabled)
    remote_pcap = f"/tmp/{timestamp}_{scenario}.pcap"
    
    tcpdump_started = False
    if capture_packets:
        ok, msg = remote_tcpdump(vm_host, "start", remote_pcap)
        tcpdump_started = ok
        if ok:
            log.info(f"tcpdump started: {msg}")
        else:
            log.warning(f"tcpdump failed to start: {msg}")

    # 1. Start SSH log tailer
    log_col = RemoteLogCollector(vm_host, vm_log_path)
    log_col.start()

    # 2. Start normal traffic generator
    normal_gen = NormalTrafficGenerator(target, rps=normal_rps)
    normal_gen.start()

    # 3. Warmup phase
    log.info(f"phase 1: warmup ({warmup}s)")
    time.sleep(warmup)
    # Wait for tail to catch up (no new records for 1s)
    drain_wait(log_col, quiet_seconds=1.0, max_wait=10.0)
    warmup_end_idx = log_col.count()
    log.info(f"  warmup ended: {warmup_end_idx} records collected")

    # 4. Attack phase
    log.info(f"phase 2: attack ({duration}s of {scenario})")
    sim = SCENARIOS[scenario](target)
    kwargs = dict(scenario_kwargs or {})
    kwargs.setdefault("duration", duration)
    if scenario == "slowloris":
        pass
    elif scenario == "mixed":
        pass
    elif scenario != "ramping_flood":
        kwargs.setdefault("threads", 30)

    sim.run(**kwargs)

    # CRITICAL: wait for SSH tail to drain all attack traffic
    log.info("  waiting for collector to drain attack traffic...")
    drain_wait(log_col, quiet_seconds=2.0, max_wait=30.0)
    attack_end_idx = log_col.count()
    log.info(f"  attack ended: {attack_end_idx} records collected "
             f"({attack_end_idx - warmup_end_idx} attack)")

    # 5. Cooldown phase
    log.info(f"phase 3: cooldown ({cooldown}s)")
    time.sleep(cooldown)
    drain_wait(log_col, quiet_seconds=1.0, max_wait=10.0)
    cooldown_end_idx = log_col.count()
    log.info(f"  cooldown ended: {cooldown_end_idx} records collected "
             f"({cooldown_end_idx - attack_end_idx} normal)")
    # 5. Cooldown phase
    log.info(f"phase 3: cooldown ({cooldown}s)")
    time.sleep(cooldown)
    cooldown_end_idx = log_col.count()
    log.info(f"  cooldown ended: {cooldown_end_idx} records collected "
             f"({cooldown_end_idx - attack_end_idx} normal)")

    # 6. Stop normal traffic
    normal_gen.stop()
    time.sleep(0.5)

    # 7. Stop tcpdump and copy pcap
    pcap_local: Path | None = None
    if tcpdump_started:
        ok, msg = remote_tcpdump(vm_host, "stop", remote_pcap)
        if ok:
            log.info(f"tcpdump stopped: {msg}")
            pcap_local = base.with_suffix(".pcap")
            if remote_copy_pcap(vm_host, remote_pcap, pcap_local):
                log.info(f"pcap copied: {pcap_local}")
            else:
                log.warning("pcap copy failed")
                pcap_local = None
        else:
            log.warning(f"tcpdump stop failed: {msg}")

    # 8. Stop log tailer
    log_col.stop()

    # 7. Label by INDEX position
    log.info("labeling records by index position...")
    records = log_col.records
    total = len(records)

    # Clamp boundaries to actual record count
    warmup_end_idx = min(warmup_end_idx, total)
    attack_end_idx = min(attack_end_idx, total)

    labeled: list[CollectedRecord] = []
    for i, rec in enumerate(records):
        if i < warmup_end_idx:
            phase, label = "warmup", 0
        elif i < attack_end_idx:
            phase, label = "attack", 1
        else:
            phase, label = "cooldown", 0

        labeled.append(CollectedRecord(
            ts=rec.ts,
            ip=rec.ip,
            method=rec.method,
            path=rec.path,
            status=rec.status,
            bytes=rec.bytes,
            user_agent=rec.user_agent,
            label=label,
            phase=phase,
            scenario=scenario,
        ))

    n_attack = sum(1 for r in labeled if r.label == 1)
    n_normal = len(labeled) - n_attack

    # 8. Write outputs
    raw_path = base.with_suffix(".log")
    raw_path.write_text("\n".join(log_col.raw_lines))

    meta = {
        "scenario": scenario,
        "target": target,
        "vm_host": vm_host,
        "warmup_seconds": warmup,
        "attack_seconds": duration,
        "cooldown_seconds": cooldown,
        "total_records": total,
        "attack_records": n_attack,
        "normal_records": n_normal,
        "warmup_end_idx": warmup_end_idx,
        "attack_end_idx": attack_end_idx,
        "scenario_kwargs": scenario_kwargs or {},
        "timestamp": timestamp,
        "pcap_captured": pcap_local is not None,
        "pcap_path": str(pcap_local) if pcap_local else None,
    }
    
    meta_path = base.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    jsonl_path = base.with_suffix(".jsonl")
    with jsonl_path.open("w") as f:
        for rec in labeled:
            f.write(json.dumps(rec.to_dict()) + "\n")

    log.info("✅ collection complete:")
    log.info(f"   raw:    {raw_path}")
    log.info(f"   meta:   {meta_path}")
    log.info(f"   jsonl:  {jsonl_path}")
    if pcap_local:
        log.info(f"   pcap:   {pcap_local}")
    log.info(f"   total={total}  attack={n_attack}  normal={n_normal}")
    
    return jsonl_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Collect labeled attack data")
    p.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    p.add_argument("--target", required=True)
    p.add_argument("--vm-host", required=True)
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--cooldown", type=int, default=5)
    p.add_argument("--normal-rps", type=float, default=3.0)
    p.add_argument("--threads", type=int, default=30)
    p.add_argument("--connections", type=int, default=50)
    args = p.parse_args()

    kwargs: dict = {}
    if args.scenario == "slowloris":
        kwargs["connections"] = args.connections
    elif args.scenario == "mixed":
        kwargs["attack_threads"] = args.threads
    elif args.scenario != "ramping_flood":
        kwargs["threads"] = args.threads

    out = collect_scenario(
        scenario=args.scenario,
        target=args.target,
        vm_host=args.vm_host,
        duration=args.duration,
        warmup=args.warmup,
        cooldown=args.cooldown,
        normal_rps=args.normal_rps,
        scenario_kwargs=kwargs,
    )

    print()
    print(f"✅ Data collected: {out}")