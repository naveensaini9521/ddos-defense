"""Redis-backed block state and IP history.

Drop-in replacement for blocker.state.BlockState with three advantages
for dynamic-IP defense:

    1. Native TTL on block records — Redis expires them automatically.
    2. Fast set membership — subnet/ASN/JA3 -> IP sets let us answer
       "how many IPs from this /24 in the last 5 min?" in O(log N).
    3. Shared across processes — collector, blocker, and API read/write
       the same state without file locks.

Environment overrides (for Docker):
    REDIS_HOST   — default "localhost"
    REDIS_PORT   — default 6379
    REDIS_DB     — default 0

Key layout (prefix ddos: by default):
    <p>:block:<target>            HASH  {scope, since, until, reason, strike,
                                         subnet, asn, ja3, confidence, model}
    <p>:blocks:active             ZSET  target -> until  (score = expiry ts)
    <p>:ip:<ip>                   HASH  {subnet, asn, ja3, first_seen,
                                         last_seen, hits}
    <p>:subnet:<cidr>:ips         ZSET  ip -> last_seen
    <p>:asn:<n>:ips               ZSET  ip -> last_seen
    <p>:ja3:<hash>:ips            ZSET  ip -> last_seen
    <p>:history                   LIST  (append-only audit, capped)
    <p>:pubsub:blocks             PUB/SUB channel for block events

Usage:
    from blocker.redis_state import RedisBlockState

    state = RedisBlockState()
    state.record_ip("1.2.3.4", subnet="1.2.3.0/24", asn=9009, ja3="abc")
    state.add_block("1.2.3.4", ttl=60, reason="dynamic_ip")
    state.is_blocked("1.2.3.4")        # True
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable

try:
    import redis
except ImportError as e:
    raise ImportError("redis-py not installed. Run: pip install redis") from e

from core.logging import get_logger

log = get_logger("blocker.redis_state")

PERMANENT_UNTIL = -1.0
DEFAULT_PREFIX = "ddos:"
HISTORY_MAX = 10_000


# ---------------------------------------------------------------------------
# Entry dataclass — same shape as blocker.state.BlockEntry
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
    subnet: str | None = None
    asn: int | None = None
    ja3: str | None = None
    scope: str = "ip"

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
            "subnet": self.subnet,
            "asn": self.asn,
            "ja3": self.ja3,
            "scope": self.scope,
            "active": self.is_active,
            "permanent": self.is_permanent,
            "ttl_remaining": self.ttl_remaining,
        }


# ---------------------------------------------------------------------------
# RedisBlockState
# ---------------------------------------------------------------------------

class RedisBlockState:
    """Redis-backed persistent block state + IP history."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
        prefix: str = DEFAULT_PREFIX,
        password: str | None = None,
        socket_timeout: float = 3.0,
        decode_responses: bool = True,
    ) -> None:
        # Env overrides (for Docker; explicit args still win if provided)
        host = host or os.environ.get("REDIS_HOST", "localhost")
        port = port or int(os.environ.get("REDIS_PORT", "6379"))
        db = db if db is not None else int(os.environ.get("REDIS_DB", "0"))
        password = password or os.environ.get("REDIS_PASSWORD") or None

        self.prefix = prefix
        self._r = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_timeout=socket_timeout,
            decode_responses=decode_responses,
        )
        self._r.ping()
        log.info(f"RedisBlockState connected: {host}:{port}/{db} prefix={prefix!r}")

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------
    def _k(self, *parts: str) -> str:
        return self.prefix + ":".join(parts)

    def _block_key(self, target: str) -> str:
        return self._k("block", target)

    def _ip_key(self, ip: str) -> str:
        return self._k("ip", ip)

    def _subnet_key(self, cidr: str) -> str:
        return self._k("subnet", cidr, "ips")

    def _asn_key(self, asn: int) -> str:
        return self._k("asn", str(asn), "ips")

    def _ja3_key(self, ja3: str) -> str:
        return self._k("ja3", ja3, "ips")

    # ------------------------------------------------------------------
    # IP history
    # ------------------------------------------------------------------
    def record_ip(
        self,
        ip: str,
        subnet: str | None = None,
        asn: int | None = None,
        ja3: str | None = None,
    ) -> None:
        now = time.time()
        pipe = self._r.pipeline()

        key = self._ip_key(ip)
        pipe.hsetnx(key, "first_seen", now)
        pipe.hset(key, mapping={
            "subnet": subnet or "",
            "asn": str(asn) if asn is not None else "",
            "ja3": ja3 or "",
            "last_seen": now,
        })
        pipe.hincrby(key, "hits", 1)

        if subnet:
            pipe.zadd(self._subnet_key(subnet), {ip: now})
        if asn is not None:
            pipe.zadd(self._asn_key(asn), {ip: now})
        if ja3:
            pipe.zadd(self._ja3_key(ja3), {ip: now})

        pipe.execute()

    def _count_in_zset(self, key: str, window: float) -> int:
        cutoff = time.time() - window
        return int(self._r.zcount(key, cutoff, "+inf"))

    def subnet_ip_count(self, subnet: str, window: float = 300.0) -> int:
        return self._count_in_zset(self._subnet_key(subnet), window)

    def asn_ip_count(self, asn: int, window: float = 300.0) -> int:
        return self._count_in_zset(self._asn_key(asn), window)

    def ja3_ip_count(self, ja3: str, window: float = 300.0) -> int:
        return self._count_in_zset(self._ja3_key(ja3), window)

    def subnet_members(self, subnet: str, window: float = 300.0) -> list[str]:
        cutoff = time.time() - window
        return list(self._r.zrangebyscore(self._subnet_key(subnet), cutoff, "+inf"))

    def asn_members(self, asn: int, window: float = 300.0) -> list[str]:
        cutoff = time.time() - window
        return list(self._r.zrangebyscore(self._asn_key(asn), cutoff, "+inf"))

    # ------------------------------------------------------------------
    # Block CRUD
    # ------------------------------------------------------------------
    def add_block(
        self,
        ip: str,
        ttl: float,
        reason: str = "",
        confidence: float = 0.0,
        model: str = "ensemble",
        strike: int | None = None,
        subnet: str | None = None,
        asn: int | None = None,
        ja3: str | None = None,
        scope: str = "ip",
    ) -> BlockEntry:
        now = time.time()
        until = PERMANENT_UNTIL if ttl == -1 else now + ttl

        if strike is None:
            existing = self.get(ip)
            strike = (existing.strike + 1) if existing else 1

        entry = BlockEntry(
            ip=ip, since=now, until=until, reason=reason,
            confidence=confidence, strike=strike, model=model,
            subnet=subnet, asn=asn, ja3=ja3, scope=scope,
        )

        key = self._block_key(ip)
        pipe = self._r.pipeline()

        pipe.hset(key, mapping={
            "ip": ip, "since": now, "until": until,
            "reason": reason, "confidence": confidence,
            "strike": strike, "model": model,
            "subnet": subnet or "",
            "asn": str(asn) if asn is not None else "",
            "ja3": ja3 or "", "scope": scope,
        })

        if until != PERMANENT_UNTIL:
            ttl_seconds = max(int(until - now), 1)
            pipe.expire(key, ttl_seconds)

        pipe.zadd(self._k("blocks", "active"), {ip: until})

        pipe.lpush(self._k("history"), json.dumps({
            "event": "block", "ip": ip, "since": now, "until": until,
            "reason": reason, "confidence": confidence, "strike": strike,
            "model": model, "scope": scope,
        }))
        pipe.ltrim(self._k("history"), 0, HISTORY_MAX - 1)

        pipe.publish(self._k("pubsub", "blocks"), json.dumps({
            "action": "block", "target": ip, "scope": scope,
            "until": until, "reason": reason,
        }))

        pipe.execute()

        log.info(f"block added: ip={ip} scope={scope} ttl={ttl:.0f}s "
                 f"strike={strike} reason={reason}")
        return entry

    def remove_block(self, ip: str) -> bool:
        entry = self.get(ip)
        if entry is None:
            return False

        key = self._block_key(ip)
        pipe = self._r.pipeline()
        pipe.delete(key)
        pipe.zrem(self._k("blocks", "active"), ip)
        pipe.lpush(self._k("history"), json.dumps({
            "event": "unblock", "ip": ip, "ts": time.time(),
        }))
        pipe.ltrim(self._k("history"), 0, HISTORY_MAX - 1)
        pipe.publish(self._k("pubsub", "blocks"), json.dumps({
            "action": "unblock", "target": ip,
        }))
        pipe.execute()

        log.info(f"block removed: ip={ip}")
        return True

    def get(self, ip: str) -> BlockEntry | None:
        h = self._r.hgetall(self._block_key(ip))
        if not h:
            return None
        return self._hash_to_entry(h)

    def is_blocked(self, ip: str) -> bool:
        entry = self.get(ip)
        return entry is not None and entry.is_active

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def active_blocks(self) -> list[BlockEntry]:
        now = time.time()
        active = self._r.zrangebyscore(self._k("blocks", "active"), now, "+inf")
        permanent = self._r.zrangebyscore(
            self._k("blocks", "active"), PERMANENT_UNTIL, PERMANENT_UNTIL
        )
        out: list[BlockEntry] = []
        for t in set(active) | set(permanent):
            e = self.get(t)
            if e is not None:
                out.append(e)
        out.sort(key=lambda x: x.since, reverse=True)
        return out

    def all_blocks(self) -> list[BlockEntry]:
        keys = self._r.keys(self._k("block", "*"))
        out: list[BlockEntry] = []
        for k in keys:
            target = k[len(self.prefix) + len("block:"):]
            e = self.get(target)
            if e is not None:
                out.append(e)
        return out

    def expired_blocks(self) -> list[BlockEntry]:
        return []

    def history(self, ip: str | None = None, limit: int = 100) -> list[dict]:
        raw = self._r.lrange(self._k("history"), 0, limit - 1)
        out = [json.loads(x) for x in raw]
        if ip:
            out = [r for r in out if r.get("ip") == ip]
        return out

    def count(self) -> dict:
        active_list = self.active_blocks()
        permanent = sum(1 for e in active_list if e.is_permanent)
        return {
            "total_records": len(active_list),
            "active": len(active_list),
            "permanent": permanent,
            "expired": 0,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def purge_expired(self) -> int:
        now = time.time()
        removed = self._r.zremrangebyscore(
            self._k("blocks", "active"), 0, now
        )
        cutoff = now - 86400
        for pattern in ("subnet:*:ips", "asn:*:ips", "ja3:*:ips"):
            for key in self._r.scan_iter(self.prefix + pattern):
                self._r.zremrangebyscore(key, 0, cutoff)
        if removed:
            log.info(f"purged {removed} expired zset entries")
        return int(removed)

    def clear(self) -> None:
        keys = list(self._r.scan_iter(self.prefix + "*"))
        if keys:
            self._r.delete(*keys)
        log.warning(f"redis state cleared ({len(keys)} keys)")

    # ------------------------------------------------------------------
    # Pub/Sub
    # ------------------------------------------------------------------
    def subscribe(self, callback: Callable[[dict], None]) -> None:
        pubsub = self._r.pubsub()
        pubsub.subscribe(self._k("pubsub", "blocks"))
        for msg in pubsub.listen():
            if msg["type"] != "message":
                continue
            try:
                callback(json.loads(msg["data"]))
            except Exception as e:
                log.warning(f"pubsub callback error: {e}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_to_entry(h: dict) -> BlockEntry:
        def _f(v, default=0.0):
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def _i(v, default=0):
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        return BlockEntry(
            ip=h.get("ip", ""),
            since=_f(h.get("since")),
            until=_f(h.get("until")),
            reason=h.get("reason", ""),
            confidence=_f(h.get("confidence")),
            strike=_i(h.get("strike"), 1),
            model=h.get("model", "ensemble"),
            subnet=h.get("subnet") or None,
            asn=_i(h["asn"]) if h.get("asn") else None,
            ja3=h.get("ja3") or None,
            scope=h.get("scope", "ip"),
        )

    def close(self) -> None:
        try:
            self._r.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Redis block state CLI")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping")
    sub.add_parser("list")
    sub.add_parser("count")
    sub.add_parser("purge")
    sub.add_parser("clear")

    add = sub.add_parser("add")
    add.add_argument("ip")
    add.add_argument("--ttl", type=float, default=60)
    add.add_argument("--reason", default="manual")
    add.add_argument("--subnet", default=None)
    add.add_argument("--asn", type=int, default=None)
    add.add_argument("--ja3", default=None)
    add.add_argument("--scope", default="ip")

    rm = sub.add_parser("remove")
    rm.add_argument("ip")

    rec = sub.add_parser("record")
    rec.add_argument("ip")
    rec.add_argument("--subnet", default=None)
    rec.add_argument("--asn", type=int, default=None)
    rec.add_argument("--ja3", default=None)

    subn = sub.add_parser("subnet-count")
    subn.add_argument("cidr")
    subn.add_argument("--window", type=float, default=300)

    asnc = sub.add_parser("asn-count")
    asnc.add_argument("asn", type=int)
    asnc.add_argument("--window", type=float, default=300)

    sub.add_parser("history")
    sub.add_parser("perms")

    args = p.parse_args()
    state = RedisBlockState(host=args.host, port=args.port, prefix=args.prefix)

    if args.cmd == "ping":
        print("PONG")
    elif args.cmd == "list":
        for e in state.active_blocks():
            tag = "PERM" if e.is_permanent else f"{e.ttl_remaining:6.1f}s"
            extra = ""
            if e.subnet:
                extra += f" subnet={e.subnet}"
            if e.asn:
                extra += f" asn={e.asn}"
            if e.ja3:
                extra += f" ja3={e.ja3[:8]}"
            print(f"{e.ip:20s} [{tag:>8s}] scope={e.scope} "
                  f"strike={e.strike} reason={e.reason}{extra}")
    elif args.cmd == "count":
        print(json.dumps(state.count(), indent=2))
    elif args.cmd == "purge":
        n = state.purge_expired()
        print(f"purged {n}")
    elif args.cmd == "add":
        e = state.add_block(
            args.ip, args.ttl, args.reason,
            subnet=args.subnet, asn=args.asn, ja3=args.ja3,
            scope=args.scope,
        )
        print(f"added: {e.ip} scope={e.scope} until={e.until:.0f}")
    elif args.cmd == "remove":
        ok = state.remove_block(args.ip)
        print("removed" if ok else "not found")
    elif args.cmd == "record":
        state.record_ip(args.ip, args.subnet, args.asn, args.ja3)
        print(f"recorded {args.ip}")
    elif args.cmd == "subnet-count":
        n = state.subnet_ip_count(args.cidr, window=args.window)
        print(f"subnet {args.cidr} -> {n} IPs in last {args.window:.0f}s")
    elif args.cmd == "asn-count":
        n = state.asn_ip_count(args.asn, window=args.window)
        print(f"ASN {args.asn} -> {n} IPs in last {args.window:.0f}s")
    elif args.cmd == "history":
        for row in state.history(limit=20):
            print(json.dumps(row))
    elif args.cmd == "perms":
        for e in state.active_blocks():
            if e.is_permanent:
                print(f"{e.ip:20s} scope={e.scope} reason={e.reason}")
    elif args.cmd == "clear":
        state.clear()

    state.close()