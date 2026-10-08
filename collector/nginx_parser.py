"""collector/nginx_parser.py

Parse nginx-style access log lines into structured records.

Supports multiple time formats:
    - 01/Oct/2026:11:13:10 +0000     (nginx default with TZ)
    - 01/Oct/2026:11:13:10           (nginx no TZ)
    - 01/Oct/2026 11:13:10           (victim_server.py)

Log format expected:
    192.168.122.1 - - [01/Oct/2026:11:13:10 +0000] "GET / HTTP/1.1" 200 615 "-" "curl/8.5.0"
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from core.logging import get_logger

log = get_logger("collector.nginx_parser")


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# Main log line regex — matches both nginx default and victim_server format
LOG_RE = re.compile(
    r'^(?P<ip>\S+) '                    # client IP
    r'\S+ \S+ '                         # ident, user (usually "- -")
    r'\[(?P<time>[^\]]+)\] '            # [01/Oct/2026:11:13:10 +0000]
    r'"(?P<method>\S+) '                # HTTP method
    r'(?P<path>\S+) '                   # path
    r'(?P<proto>[^"]+)" '               # protocol (HTTP/1.1)
    r'(?P<status>\d{3})'                # status code
    r'(?: (?P<bytes>\S+))?'             # bytes (optional)
    r'(?: "(?P<referer>[^"]*)" "(?P<ua>[^"]*)")?'  # optional referer + UA
)

# Time formats (tried in order)
TIME_FORMATS = [
    "%d/%b/%Y:%H:%M:%S %z",     # nginx default:  01/Oct/2026:11:13:10 +0000
    "%d/%b/%Y:%H:%M:%S",        # nginx no TZ:    01/Oct/2026:11:13:10
    "%d/%b/%Y %H:%M:%S",        # victim_server:  01/Oct/2026 11:13:10
]


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedLog:
    ts: float
    ip: str
    method: str
    path: str
    proto: str
    status: int
    bytes: int
    referer: str | None
    user_agent: str | None
    raw: str
    subnet: str | None = None
    asn: int | None = None
    asn_type: str | None = None
    ja3: str | None = None

    @property
    def is_error(self) -> bool:
        return self.status >= 400

    @property
    def is_server_error(self) -> bool:
        return self.status >= 500

# ---------------------------------------------------------------------------
# Time parsing — tries multiple formats
# ---------------------------------------------------------------------------

def _parse_time(s: str) -> float | None:
    """Try multiple nginx time formats. Returns unix timestamp or None."""
    for fmt in TIME_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            # If no timezone info, assume UTC
            if dt.tzinfo is None:
                ts = dt.replace(tzinfo=timezone.utc).timestamp()
            else:
                ts = dt.timestamp()
            return ts
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_line(line: str) -> ParsedLog | None:
    """Parse one log line. Returns None if it doesn't match."""
    line = line.strip()
    if not line:
        return None

    m = LOG_RE.match(line)
    if not m:
        return None

    ts = _parse_time(m.group("time"))
    if ts is None:
        return None

    try:
        status = int(m.group("status"))
    except (TypeError, ValueError):
        return None

    bytes_str = m.group("bytes")
    try:
        nbytes = int(bytes_str) if bytes_str not in (None, "-") else 0
    except (TypeError, ValueError):
        nbytes = 0

    return ParsedLog(
        ts=ts,
        ip=m.group("ip"),
        method=m.group("method"),
        path=m.group("path"),
        proto=m.group("proto"),
        status=status,
        bytes=nbytes,
        referer=m.group("referer"),
        user_agent=m.group("ua"),
        raw=line,
    )


def parse_file(path: str | Path) -> list[ParsedLog]:
    """Parse every line in a file. Skips unparseable lines."""
    p = Path(path)
    if not p.exists():
        log.warning(f"log file does not exist: {p}")
        return []

    out: list[ParsedLog] = []
    with p.open("r", errors="ignore") as f:
        for line in f:
            rec = parse_line(line)
            if rec:
                out.append(rec)
    return out


def parse_stream(lines: Iterable[str]) -> Iterator[ParsedLog]:
    """Parse an iterable of lines lazily — useful for tailing live logs."""
    for line in lines:
        rec = parse_line(line)
        if rec:
            yield rec


# ---------------------------------------------------------------------------
# CLI (test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Nginx log parser")
    p.add_argument("--file", default=None, help="parse a file")
    p.add_argument("--line", default=None, help="parse a single line")
    p.add_argument("--stdin", action="store_true", help="parse stdin lines")
    args = p.parse_args()

    if args.line:
        r = parse_line(args.line)
        if r:
            print(f"OK: ip={r.ip} status={r.status} ts={r.ts:.0f} path={r.path}")
        else:
            print("PARSE FAILED")

    elif args.file:
        records = parse_file(args.file)
        print(f"parsed {len(records)} records from {args.file}")
        for r in records[:5]:
            print(f"  {r.ip:15s} {r.method:5s} {r.path:20s} {r.status}")

    elif args.stdin:
        count = 0
        for line in sys.stdin:
            r = parse_line(line)
            if r:
                count += 1
                print(f"{r.ip:15s} {r.status} {r.path}")
        print(f"\nparsed {count} records", file=sys.stderr)

    else:
        # default: run a built-in sanity test
        tests = [
            ("nginx with TZ", '192.168.122.1 - - [01/Oct/2026:11:13:10 +0000] "GET / HTTP/1.1" 200 615 "-" "curl/8.5.0"'),
            ("nginx no TZ",   '192.168.122.1 - - [01/Oct/2026:11:13:10] "GET / HTTP/1.1" 200 615'),
            ("victim_server", '192.168.122.1 - - [01/Oct/2026 11:13:10] "GET / HTTP/1.1" 200'),
            ("garbage",       'this is not a log line'),
        ]
        for name, line in tests:
            r = parse_line(line)
            status = "OK" if r else "FAIL"
            print(f"[{status}] {name}")
            if r:
                print(f"        ip={r.ip} status={r.status} ts={r.ts:.0f}")