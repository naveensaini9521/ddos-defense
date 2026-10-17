"""Health check endpoints."""
from __future__ import annotations

import os

from fastapi import APIRouter

from api.schemas import HealthOut
from core.config_loader import load as load_config

router = APIRouter()


@router.get("/health", response_model=HealthOut)
def health():
    """Liveness probe: is the API itself up?"""
    components: dict[str, str] = {"api": "ok"}

    # Redis — respect env override (Docker sets REDIS_HOST=redis)
    try:
        import redis
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        r = redis.Redis(host=host, port=port, socket_timeout=1)
        r.ping()
        components["redis"] = "ok"
    except Exception:
        components["redis"] = "down"

    # Firewall state file
    try:
        from blocker.firewall import BLOCKED_FILE
        components["firewall_state"] = "ok" if BLOCKED_FILE.exists() else "empty"
    except Exception:
        components["firewall_state"] = "unknown"

    # Config
    try:
        cfg = load_config()
        components["config"] = "ok" if cfg else "empty"
    except Exception:
        components["config"] = "down"

    overall = "ok" if all(v == "ok" for v in components.values()) else "degraded"
    return HealthOut(status=overall, components=components)