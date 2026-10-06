"""Persistent block state — SQLite-backed storage of blocked IPs.

Survives restarts of the blocker. Supports:
    - Add / remove blocks
    - Query active blocks
    - Query block history
    - Automatic expiry of old records (permanent blocks are never purged)
    - Strike counts (repeat offenders)

Permanent blocks are represented by until == -1.

Usage:
    from blocker.state import BlockState

    state = BlockState()
    state.add_block("1.2.3.4", ttl=60, reason="http_flood", confidence=0.94)
    state.is_blocked("1.2.3.4")     # True
    state.active_blocks()           # list of active records
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from core.logging import get_logger

log = get_logger("blocker.state")

DEFAULT_DB = Path("data/blocker_state.db")

# Sentinel value meaning "never expires"
PERMANENT_UNTIL = -1.0


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class BlockEntry:
    ip: str
    since: float
    until: float
    reason: str
    confidence: float
    strike: int
    model: str

    @property
    def is_permanent(self) -> bool:
        return self.until == PERMANENT_UNTIL

    @property
    def is_active(self) -> bool:
        if self.is_permanent:
            return True
        return time.time() < self.until

    @property
    def ttl_remaining(self) -> float:
        if self.is_permanent:
            return float("inf")
        return max(self.until - time.time(), 0.0)

    def to_dict(self) -> dict:
        return {
            "ip": self.ip,
            "since": self.since,
            "until": self.until,
            "reason": self.reason,
            "confidence": self.confidence,
            "strike": self.strike,
            "model": self.model,
            "active": self.is_active,
            "permanent": self.is_permanent,
            "ttl_remaining": self.ttl_remaining,
        }


# ---------------------------------------------------------------------------
# SQLite-backed store
# ---------------------------------------------------------------------------

class BlockState:
    """SQLite-backed persistent block state."""

    def __init__(self, db_path: str | Path = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        log.info(f"BlockState ready: {self.db_path}")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    ip          TEXT PRIMARY KEY,
                    since       REAL NOT NULL,
                    until       REAL NOT NULL,
                    reason      TEXT NOT NULL,
                    confidence  REAL NOT NULL DEFAULT 0.0,
                    strike      INTEGER NOT NULL DEFAULT 1,
                    model       TEXT NOT NULL DEFAULT 'ensemble'
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_blocks_until
                ON blocks(until)
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip          TEXT NOT NULL,
                    since       REAL NOT NULL,
                    until       REAL NOT NULL,
                    reason      TEXT NOT NULL,
                    confidence  REAL NOT NULL DEFAULT 0.0,
                    strike      INTEGER NOT NULL DEFAULT 1,
                    model       TEXT NOT NULL DEFAULT 'ensemble',
                    event       TEXT NOT NULL
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_ip
                ON history(ip)
            """)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def add_block(
        self,
        ip: str,
        ttl: float,
        reason: str = "",
        confidence: float = 0.0,
        model: str = "ensemble",
        strike: int | None = None,
    ) -> BlockEntry:
        """Add or extend a block.

        Args:
            ip: source IP
            ttl: seconds until expiry. -1 = permanent.
            reason: human-readable reason
            confidence: model confidence (0..1)
            model: which model produced this decision
            strike: explicit strike count (overrides auto-increment if provided)
        """
        now = time.time()
        until = PERMANENT_UNTIL if ttl == -1 else now + ttl

        # Determine strike count
        existing = self.get(ip)
        if strike is None:
            if existing is not None:
                strike = existing.strike + 1
            else:
                strike = 1

        entry = BlockEntry(
            ip=ip,
            since=now,
            until=until,
            reason=reason,
            confidence=confidence,
            strike=strike,
            model=model,
        )

        with self._conn:
            self._conn.execute("""
                INSERT INTO blocks (ip, since, until, reason, confidence, strike, model)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ip) DO UPDATE SET
                    since=excluded.since,
                    until=excluded.until,
                    reason=excluded.reason,
                    confidence=excluded.confidence,
                    strike=excluded.strike,
                    model=excluded.model
            """, (entry.ip, entry.since, entry.until, entry.reason,
                  entry.confidence, entry.strike, entry.model))

            self._conn.execute("""
                INSERT INTO history (ip, since, until, reason, confidence,
                                     strike, model, event)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'block')
            """, (entry.ip, entry.since, entry.until, entry.reason,
                  entry.confidence, entry.strike, entry.model))

        log.info(f"block added: ip={ip} ttl={ttl:.0f}s reason={reason} "
                 f"conf={confidence:.2f} strike={strike}")
        return entry

    def remove_block(self, ip: str) -> bool:
        """Remove a block. Returns True if something was removed."""
        entry = self.get(ip)
        if entry is None:
            return False

        with self._conn:
            self._conn.execute("DELETE FROM blocks WHERE ip = ?", (ip,))
            self._conn.execute("""
                INSERT INTO history (ip, since, until, reason, confidence,
                                     strike, model, event)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'unblock')
            """, (entry.ip, entry.since, entry.until, entry.reason,
                  entry.confidence, entry.strike, entry.model))

        log.info(f"block removed: ip={ip}")
        return True

    def get(self, ip: str) -> BlockEntry | None:
        """Fetch a single block entry."""
        cur = self._conn.execute(
            "SELECT * FROM blocks WHERE ip = ?", (ip,)
        )
        row = cur.fetchone()
        return self._row_to_entry(row) if row else None

    def is_blocked(self, ip: str) -> bool:
        """True if the IP has an active block."""
        entry = self.get(ip)
        return entry is not None and entry.is_active

    # ------------------------------------------------------------------
    # Queries (handle permanent blocks correctly)
    # ------------------------------------------------------------------
    def active_blocks(self) -> list[BlockEntry]:
        """All currently active blocks (including permanent ones)."""
        now = time.time()
        cur = self._conn.execute(
            "SELECT * FROM blocks "
            "WHERE until > ? OR until = ? "
            "ORDER BY since DESC",
            (now, PERMANENT_UNTIL),
        )
        return [self._row_to_entry(r) for r in cur.fetchall()]

    def all_blocks(self) -> list[BlockEntry]:
        """All block records (active + expired)."""
        cur = self._conn.execute("SELECT * FROM blocks ORDER BY since DESC")
        return [self._row_to_entry(r) for r in cur.fetchall()]

    def expired_blocks(self) -> list[BlockEntry]:
        """Blocks whose TTL has passed (excludes permanent blocks)."""
        now = time.time()
        cur = self._conn.execute(
            "SELECT * FROM blocks "
            "WHERE until <= ? AND until != ? "
            "ORDER BY until DESC",
            (now, PERMANENT_UNTIL),
        )
        return [self._row_to_entry(r) for r in cur.fetchall()]

    def history(self, ip: str | None = None, limit: int = 100) -> list[dict]:
        """Fetch block/unblock history."""
        if ip:
            cur = self._conn.execute("""
                SELECT * FROM history WHERE ip = ? ORDER BY id DESC LIMIT ?
            """, (ip, limit))
        else:
            cur = self._conn.execute("""
                SELECT * FROM history ORDER BY id DESC LIMIT ?
            """, (limit,))
        return [dict(r) for r in cur.fetchall()]

    def count(self) -> dict:
        """Summary counts."""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM blocks"
        ).fetchone()[0]
        permanent = self._conn.execute(
            "SELECT COUNT(*) FROM blocks WHERE until = ?",
            (PERMANENT_UNTIL,),
        ).fetchone()[0]
        active = len(self.active_blocks())
        return {
            "total_records": total,
            "active": active,
            "permanent": permanent,
            "expired": total - active,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def purge_expired(self) -> int:
        """Delete records whose TTL has passed. Never removes permanent blocks."""
        now = time.time()
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM blocks WHERE until <= ? AND until != ?",
                (now, PERMANENT_UNTIL),
            )
            n = cur.rowcount
        if n:
            log.info(f"purged {n} expired blocks")
        return n

    def clear(self) -> None:
        """Wipe all block records (danger!)."""
        with self._conn:
            self._conn.execute("DELETE FROM blocks")
            self._conn.execute("DELETE FROM history")
        log.warning("block state cleared")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _row_to_entry(self, row: sqlite3.Row) -> BlockEntry:
        return BlockEntry(
            ip=row["ip"],
            since=row["since"],
            until=row["until"],
            reason=row["reason"],
            confidence=row["confidence"],
            strike=row["strike"],
            model=row["model"],
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Block state CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list active blocks")
    sub.add_parser("all", help="list all blocks (incl. expired)")
    sub.add_parser("count", help="summary counts")
    sub.add_parser("purge", help="remove expired records")

    add = sub.add_parser("add", help="manually add a block")
    add.add_argument("ip")
    add.add_argument("--ttl", type=float, default=60)
    add.add_argument("--reason", default="manual")

    perm = sub.add_parser("block-permanent", help="add a permanent block")
    perm.add_argument("ip")
    perm.add_argument("--reason", default="manual_permanent")

    rm = sub.add_parser("remove", help="remove a block")
    rm.add_argument("ip")

    hist = sub.add_parser("history", help="show history")
    hist.add_argument("ip", nargs="?")
    hist.add_argument("--limit", type=int, default=20)

    args = p.parse_args()
    state = BlockState()

    if args.cmd == "list":
        for e in state.active_blocks():
            tag = "PERM" if e.is_permanent else f"{e.ttl_remaining:5.1f}s"
            print(f"{e.ip:15s}  [{tag:>8s}]  strike={e.strike}  "
                  f"reason={e.reason}")
    elif args.cmd == "all":
        for e in state.all_blocks():
            if e.is_permanent:
                status = "PERMANENT"
            elif e.is_active:
                status = "ACTIVE"
            else:
                status = "expired"
            print(f"{e.ip:15s}  [{status:>9s}]  strike={e.strike}  "
                  f"reason={e.reason}")
    elif args.cmd == "count":
        print(json.dumps(state.count(), indent=2))
    elif args.cmd == "purge":
        n = state.purge_expired()
        print(f"purged {n} records")
    elif args.cmd == "add":
        entry = state.add_block(args.ip, args.ttl, args.reason)
        print(f"added: {entry.ip} until={entry.until:.0f}")
    elif args.cmd == "block-permanent":
        entry = state.add_block(args.ip, ttl=-1, reason=args.reason)
        print(f"permanently blocked: {entry.ip}")
    elif args.cmd == "remove":
        ok = state.remove_block(args.ip)
        print("removed" if ok else "not found")
    elif args.cmd == "history":
        for row in state.history(args.ip, args.limit):
            print(f"{row['event']:8s}  ip={row['ip']:15s}  "
                  f"reason={row['reason']}")