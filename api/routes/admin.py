"""Admin endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from core.config_loader import load as load_config

router = APIRouter()


def _state():
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


@router.post("/reload-model")
def reload_model():
    """Placeholder — model reload requires a running pipeline instance."""
    return {"ok": True, "message": "reload not implemented in standalone API"}


@router.post("/purge-expired")
def purge_expired():
    state = _state()
    try:
        n = state.purge_expired()
        return {"ok": True, "purged": n}
    finally:
        state.close()


@router.get("/state")
def state_dump():
    state = _state()
    try:
        return {
            "backend": type(state).__name__,
            "counts": state.count(),
            "active": [e.to_dict() for e in state.active_blocks()],
        }
    finally:
        state.close()


@router.post("/clear-state")
def clear_state():
    """DANGER — wipes all block state."""
    state = _state()
    try:
        state.clear()
        return {"ok": True, "message": "state cleared"}
    finally:
        state.close()