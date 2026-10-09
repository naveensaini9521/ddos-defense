"""Statistics endpoints."""
from __future__ import annotations

import time

from fastapi import APIRouter

from api.schemas import StatsOut
from core.config_loader import load as load_config

router = APIRouter()
_started_at = time.time()


def _state():
    """Build the state backend the same way BlockManager does."""
    cfg = load_config()
    redis_cfg = cfg.get("redis", {})
    if redis_cfg.get("host"):
        try:
            from blocker.redis_state import RedisBlockState
            return RedisBlockState(
                host=redis_cfg["host"],
                port=int(redis_cfg.get("port", 6379)),
                db=int(redis_cfg.get("db", 0)),
                prefix=redis_cfg.get("prefix", "ddos:"),
            )
        except Exception:
            pass
    from blocker.state import BlockState
    return BlockState()


@router.get("/summary", response_model=StatsOut)
def summary():
    state = _state()
    try:
        counts = state.count()
        return StatsOut(
            active_blocks=counts["active"],
            total_records=counts["total_records"],
            permanent_blocks=counts["permanent"],
            uptime_seconds=int(time.time() - _started_at),
            state_backend=type(state).__name__,
        )
    finally:
        state.close()


@router.get("/history")
def history(ip: str | None = None, limit: int = 50):
    state = _state()
    try:
        rows = state.history(ip, limit=limit)
        return {"count": len(rows), "history": rows}
    finally:
        state.close()


@router.get("/asns")
def list_blocked_asns():
    """Show which ASNs are currently blocked (via firewall index)."""
    try:
        from pathlib import Path
        import json
        p = Path("data/firewall_asn_index.json")
        if not p.exists():
            return {"count": 0, "asns": {}}
        with p.open() as f:
            data = json.load(f)
        return {
            "count": len(data),
            "asns": {k: len(v) for k, v in data.items()},
        }
    except Exception as e:
        return {"count": 0, "asns": {}, "error": str(e)}