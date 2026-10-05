"""Data collection pipeline.

Runs an attack scenario against the VM while simultaneously capturing
the VM's logs, then labels every log line based on the scenario running.

Output: data/collected/ directory with:
    - <timestamp>_<scenario>.log        raw logs captured during attack
    - <timestamp>_<scenario>.meta.json  metadata (scenario, times, params)
    - <timestamp>_<scenario>.jsonl      labeled records for ML training

Usage:
    from simulator.collector import collect_scenario

    collect_scenario(
        scenario="http_flood",
        target="http://192.168.122.30/",
        vm_host="naveen@192.168.122.30",
        duration=30,
    )
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
from simulator.attack_scenarios import SCENARIOS, ScenarioBase

log = get_logger("simulator.collector")

COLLECTED_DIR = Path("data/collected")


# ---------------------------------------------------------------------------
# Label record
# ---------------------------------------------------------------------------

@dataclass
class CollectedRecord:
    """One log line + its label."""
    ts: float
    ip: str
    method: str
    path: str
    status: int
    bytes: int
    user_agent: str | None
    label: int              # 0 = normal, 1 = attack
    phase: str              # "warmup", "attack", "cooldown"
    scenario: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Log capture thread
# ---------------------------------------------------------------------------

class RemoteLogCollector:
    """Tails the VM's log during the attack window."""

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

        # Drain stderr in background
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
# Warmup traffic generator (normal traffic before/after attack)
# ---------------------------------------------------------------------------

class NormalTrafficGenerator:
    """Generates low-rate normal traffic to label as normal (label=0)."""

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
        import requests
        interval = 1.0 / max(self.rps, 0.001)
        paths = ["/", "/api/data", "/health"]
        import random
        while not self._stop.is_set():
            try:
                path = random.choice(paths).lstrip("/")
                requests.get(self.target + path, timeout=self.timeout)
            except Exception:
                pass
            self._stop.wait(interval)


# ---------------------------------------------------------------------------
# Main collection driver
# ---------------------------------------------------------------------------

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
) -> Path:
    """Run a scenario and collect labeled data.

    Args:
        scenario: one of SCENARIOS keys
        target: HTTP target (VM URL)
        vm_host: SSH host for log collection (user@host)
        duration: attack duration in seconds
        warmup: seconds of normal traffic BEFORE attack (labeled 0)
        cooldown: seconds of normal traffic AFTER attack (labeled 0)
        scenario_kwargs: extra args for the scenario (threads, etc.)

    Returns:
        Path to the labeled JSONL file.
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

    # 1. Start log collector
    log_col = RemoteLogCollector(vm_host, vm_log_path)
    log_col.start()

    # 2. Start normal traffic generator
    normal_gen = NormalTrafficGenerator(target, rps=normal_rps)
    normal_gen.start()

    # 3. Phase 1: warmup — collect normal traffic only
    log.info(f"phase 1: warmup ({warmup}s of normal traffic)")
    warmup_start = time.time()
    time.sleep(warmup)
    warmup_end = time.time()

    # Snapshot the record count so we know where attack-phase starts
    warmup_record_count = len(log_col.records)

    # 4. Phase 2: attack — run the scenario
    log.info(f"phase 2: attack ({duration}s of {scenario})")
    attack_start = time.time()
    attack_record_start = len(log_col.records)

    sim = SCENARIOS[scenario](target)
    kwargs = scenario_kwargs or {}

    # Dispatch per scenario
    if scenario == "slowloris":
        kwargs.setdefault("duration", duration)
        sim.run(**kwargs)
    elif scenario == "mixed":
        kwargs.setdefault("duration", duration)
        sim.run(**kwargs)
    else:
        kwargs.setdefault("duration", duration)
        kwargs.setdefault("threads", 30)
        sim.run(**kwargs)

    attack_end = time.time()
    attack_record_end = len(log_col.records)

    # 5. Phase 3: cooldown — normal traffic only
    log.info(f"phase 3: cooldown ({cooldown}s)")
    cooldown_start = time.time()
    time.sleep(cooldown)
    cooldown_end = time.time()

    # 6. Stop everything
    normal_gen.stop()
    time.sleep(0.5)
    log_col.stop()

    # 7. Label every record
    log.info("labeling records...")
    labeled: list[CollectedRecord] = []
    for i, rec in enumerate(log_col.records):
        # Determine phase and label by index position within the stream
        if rec.ts < attack_start:
            phase, label = "warmup", 0
        elif rec.ts <= attack_end:
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

    # 8. Write outputs
    # Raw log
    raw_path = base.with_suffix(".log")
    raw_path.write_text("\n".join(log_col.raw_lines))

    # Metadata
    meta = {
        "scenario": scenario,
        "target": target,
        "vm_host": vm_host,
        "warmup_seconds": warmup,
        "attack_seconds": duration,
        "cooldown_seconds": cooldown,
        "warmup_start": warmup_start,
        "attack_start": attack_start,
        "attack_end": attack_end,
        "cooldown_end": cooldown_end,
        "total_records": len(log_col.records),
        "attack_records": attack_record_end - attack_record_start,
        "normal_records": len(log_col.records) - (attack_record_end - attack_record_start),
        "scenario_kwargs": scenario_kwargs or {},
        "timestamp": timestamp,
    }
    meta_path = base.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    # Labeled JSONL (for ML training)
    jsonl_path = base.with_suffix(".jsonl")
    with jsonl_path.open("w") as f:
        for rec in labeled:
            f.write(json.dumps(rec.to_dict()) + "\n")

    log.info(f"✅ collection complete:")
    log.info(f"   raw:    {raw_path}")
    log.info(f"   meta:   {meta_path}")
    log.info(f"   jsonl:  {jsonl_path}  ({len(labeled)} records)")
    log.info(f"   attack records: {meta['attack_records']}, "
             f"normal records: {meta['normal_records']}")

    return jsonl_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Collect labeled attack data")
    p.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    p.add_argument("--target", required=True)
    p.add_argument("--vm-host", required=True,
                   help="e.g. naveen@192.168.122.30")
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--cooldown", type=int, default=5)
    p.add_argument("--normal-rps", type=float, default=3.0)
    p.add_argument("--threads", type=int, default=30)
    p.add_argument("--connections", type=int, default=50)
    args = p.parse_args()

    kwargs = {}
    if args.scenario == "slowloris":
        kwargs["connections"] = args.connections
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