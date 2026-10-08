"""Live log collector — tails remote or local logs and provides sliding windows.

Runs a background thread that:
    - Reads lines from a source (local file or SSH `tail -F`)
    - Parses each line via collector/nginx_parser
    - Enriches each parsed record with subnet / ASN context
    - Stores parsed records in a bounded deque
    - Discards records older than `max_age` seconds

For SSH sources, uses pty.openpty() + `ssh -tt` so the remote `tail` sees a
TTY and line-buffers its output. Without a PTY, remote tail block-buffers,
and the collector reads nothing until ~4KB accumulates.

Usage:
    from pipeline.live_collector import LiveCollector

    lc = LiveCollector(source="ssh://naveen@192.168.122.30/var/log/nginx/access.log")
    lc.start()
    records = lc.window(seconds=10)
    lc.stop()
"""
from __future__ import annotations

import os
import pty
import re
import select
import shlex
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, replace

from collector.enrich import enrich
from collector.nginx_parser import ParsedLog, parse_line
from core.logging import get_logger

log = get_logger("pipeline.live_collector")


# ---------------------------------------------------------------------------
# Source parsing
# ---------------------------------------------------------------------------

@dataclass
class SourceSpec:
    kind: str              # "file" | "ssh" | "command"
    path: str = ""
    host: str = ""
    command: str = ""


SSH_RE = re.compile(r"^ssh://(?P<user>[^@]+)@(?P<host>[^/]+)/(?P<path>.+)$")


def parse_source(source: str) -> SourceSpec:
    """Parse a source string.

        /path/to/file.log                    — local file
        ssh://user@host/path/to/file.log     — remote file via SSH
        cmd:tail -f /path                    — arbitrary command
    """
    if source.startswith("ssh://"):
        m = SSH_RE.match(source)
        if not m:
            raise ValueError(f"bad ssh source: {source}")
        path = "/" + m.group("path")     # restore leading slash
        return SourceSpec(
            kind="ssh",
            host=f"{m.group('user')}@{m.group('host')}",
            path=path,
        )
    if source.startswith("cmd:"):
        return SourceSpec(kind="command", command=source[4:])
    return SourceSpec(kind="file", path=source)


def build_command(spec: SourceSpec, use_sudo: bool = False) -> list[str]:
    if spec.kind == "file":
        return ["tail", "-F", "-n", "0", spec.path]

    if spec.kind == "ssh":
        remote_path = spec.path if spec.path.startswith("/") else "/" + spec.path
        tail_cmd = f"tail -F -n 0 {shlex.quote(remote_path)}"
        if use_sudo:
            tail_cmd = f"sudo -n {tail_cmd}"
        # -tt forces PTY allocation; no stdbuf needed when we have a PTY
        return [
            "ssh", "-T",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "BatchMode=yes",
            spec.host,
            tail_cmd,
        ]

    if spec.kind == "command":
        return ["bash", "-c", spec.command]

    raise ValueError(f"unknown source kind: {spec.kind}")


# ---------------------------------------------------------------------------
# LiveCollector
# ---------------------------------------------------------------------------

class LiveCollector:
    """Tails a log source and provides sliding windows of parsed records."""

    def __init__(
        self,
        source: str | None = None,
        max_records: int = 100_000,
        max_age: float = 3600.0,
        reconnect_delay: float = 3.0,
        host: str | None = None,
        path: str | None = None,
        use_sudo: bool = False,
    ) -> None:
        if source is None:
            if host and path:
                source = f"ssh://{host}{path}"
            elif path:
                source = path
            else:
                raise ValueError("must provide source, or host+path")

        self.source_str = source
        self.spec = parse_source(source)
        self.use_sudo = use_sudo
        self.command = build_command(self.spec, use_sudo=use_sudo)

        self.max_records = max_records
        self.max_age = max_age
        self.reconnect_delay = reconnect_delay

        self._buf: deque[ParsedLog] = deque(maxlen=max_records)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None

        self.total_lines_read = 0
        self.total_parsed = 0
        self.total_parse_errors = 0
        self.reconnects = 0
        self.started_at: float = 0.0

        log.info(f"LiveCollector source={self.source_str} kind={self.spec.kind}")
        log.info(f"  command: {' '.join(self.command)}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            log.warning("LiveCollector already running")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="live-collector",
        )
        self._thread.start()
        self.started_at = time.time()
        log.info("LiveCollector started")

    def stop(self, timeout: float = 5.0) -> None:
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
            self._thread.join(timeout=timeout)
        log.info(
            f"LiveCollector stopped: "
            f"lines={self.total_lines_read} "
            f"parsed={self.total_parsed} "
            f"errors={self.total_parse_errors} "
            f"reconnects={self.reconnects}"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tail_once()
            except Exception as e:
                log.warning(f"tail loop error: {e}")
            if not self._stop.is_set():
                self.reconnects += 1
                log.info(
                    f"reconnecting in {self.reconnect_delay}s "
                    f"(attempt {self.reconnects})"
                )
                self._stop.wait(self.reconnect_delay)

    # ------------------------------------------------------------------
    # Tail: dispatcher
    # ------------------------------------------------------------------
    def _tail_once(self) -> None:
        """Tail the source. Uses a PTY for SSH; plain pipes otherwise."""
        if self.spec.kind == "ssh":
            self._tail_once_pty()
        else:
            self._tail_once_plain()

    # ------------------------------------------------------------------
    # SSH tail with PTY — the fix for remote line-buffering
    # ------------------------------------------------------------------
    def _tail_once_pty(self) -> None:
        """Spawn SSH with a local PTY so remote `tail` line-buffers.

        Without this, `tail -F` block-buffers over an SSH pipe and we see
        nothing until ~4KB accumulates.
        """
        cmd = list(self.command)
        log.info(f"spawning (pty): {' '.join(cmd)}")

        master_fd, slave_fd = pty.openpty()
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
                preexec_fn=os.setsid,
            )
        finally:
            os.close(slave_fd)      # only need the master side

        buf = b""
        try:
            while not self._stop.is_set():
                # 0.5s select timeout so we honor stop() promptly
                r, _, _ = select.select([master_fd], [], [], 0.5)
                if not r:
                    if self._proc.poll() is not None:
                        log.info("tail subprocess exited")
                        break
                    continue

                try:
                    chunk = os.read(master_fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break

                buf += chunk
                # PTY gives \r\n — split on \n, strip \r
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.rstrip(b"\r").decode("utf-8", errors="replace")
                    self._handle_line(line)

        except Exception as e:
            log.warning(f"pty reader error: {e}")
        finally:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=2)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
            self._proc = None
            try:
                os.close(master_fd)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Plain tail (local file / cmd:) — no PTY needed
    # ------------------------------------------------------------------
    def _tail_once_plain(self) -> None:
        """Spawn tail command and read lines with readline().

        Used for local files and `cmd:` sources where buffering isn't a
        problem.
        """
        cmd = list(self.command)
        log.info(f"spawning (plain): {' '.join(cmd)}")

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            universal_newlines=False,
        )

        # Drain stderr in a background thread
        def read_stderr():
            proc = self._proc
            if proc is None or proc.stderr is None:
                return
            try:
                while True:
                    raw = proc.stderr.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if line:
                        log.warning(f"tail stderr: {line}")
            except Exception as e:
                log.debug(f"stderr reader error: {e}")

        threading.Thread(target=read_stderr, daemon=True).start()

        try:
            proc = self._proc
            if proc is not None and proc.stdout is not None:
                while not self._stop.is_set():
                    raw = proc.stdout.readline()
                    if not raw:
                        log.info("tail subprocess EOF")
                        break
                    try:
                        line = raw.decode("utf-8", errors="replace")
                    except Exception:
                        continue
                    self._handle_line(line)
        except Exception as e:
            log.warning(f"stdout reader error: {e}")
        finally:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=2)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
            self._proc = None

    # ------------------------------------------------------------------
    # Line handling
    # ------------------------------------------------------------------
    def _handle_line(self, line: str) -> None:
        """Parse one log line and enrich with subnet/ASN context."""
        self.total_lines_read += 1
        line = line.rstrip("\r\n")

        if not line.strip():
            return

        parsed = parse_line(line)
        if parsed is None:
            self.total_parse_errors += 1
            if self.total_parse_errors <= 5:
                log.debug(f"unparseable line: {line[:120]}")
            return

        # Attach enrichment (LRU-cached; near-free for repeat IPs)
        try:
            ctx = enrich(parsed.ip)
            enriched = replace(
                parsed,
                subnet=ctx.get("subnet"),
                asn=ctx.get("asn"),
                asn_type=ctx.get("asn_type"),
            )
        except Exception as e:
            # Never drop a record just because enrichment failed
            log.debug(f"enrich failed for {parsed.ip}: {e}")
            enriched = parsed

        with self._lock:
            self._buf.append(enriched)
            self.total_parsed += 1

    # ------------------------------------------------------------------
    # Window queries
    # ------------------------------------------------------------------
    def window(self, seconds: float = 10.0,
               evict: bool = True) -> list[ParsedLog]:
        cutoff = time.time() - seconds
        with self._lock:
            records = [r for r in self._buf if r.ts >= cutoff]
            if evict:
                hard_cutoff = time.time() - self.max_age
                while self._buf and self._buf[0].ts < hard_cutoff:
                    self._buf.popleft()
        return records

    def since(self, ts: float) -> list[ParsedLog]:
        with self._lock:
            return [r for r in self._buf if r.ts > ts]

    def latest(self, n: int = 10) -> list[ParsedLog]:
        with self._lock:
            return list(self._buf)[-n:]

    def all_records(self) -> list[ParsedLog]:
        with self._lock:
            return list(self._buf)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            buf_size = len(self._buf)
        return {
            "source": self.source_str,
            "buffer_size": buf_size,
            "total_lines_read": self.total_lines_read,
            "total_parsed": self.total_parsed,
            "total_parse_errors": self.total_parse_errors,
            "reconnects": self.reconnects,
            "uptime_seconds": int(time.time() - self.started_at)
                              if self.started_at else 0,
        }

    def wait_for_records(self, count: int = 1, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if len(self._buf) >= count:
                    return True
            time.sleep(0.1)
        return False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def collector_from_config(cfg: dict) -> LiveCollector:
    c = cfg.get("collector", {})
    path = c.get("log_path", "/var/log/nginx/access.log")

    if c.get("remote_log"):
        host = c.get("remote_host")
        if not host:
            raise ValueError("remote_log=true but remote_host not set")
        source = f"ssh://{host}{path}"
    else:
        source = path

    return LiveCollector(
        source=source,
        max_age=float(c.get("max_age", 3600)),
        use_sudo=bool(c.get("use_sudo", False)),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Live log collector test")
    p.add_argument("--source", default=None,
                   help="e.g. ssh://naveen@192.168.122.30/var/log/nginx/access.log")
    p.add_argument("--duration", type=float, default=30)
    p.add_argument("--window", type=float, default=10)
    p.add_argument("--sudo", action="store_true")
    args = p.parse_args()

    if args.source is None:
        from core.config_loader import load
        load(reload=True)
        lc = collector_from_config(load())
    else:
        lc = LiveCollector(args.source, use_sudo=args.sudo)

    lc.start()
    print(f"collecting for {args.duration}s...")

    try:
        end = time.time() + args.duration
        while time.time() < end:
            time.sleep(2)
            s = lc.stats()
            w = lc.window(seconds=args.window)
            print(
                f"[{int(time.time())}] buffer={s['buffer_size']} "
                f"lines={s['total_lines_read']} "
                f"parsed={s['total_parsed']} "
                f"errors={s['total_parse_errors']} "
                f"window({args.window}s)={len(w)}"
            )
    finally:
        lc.stop()