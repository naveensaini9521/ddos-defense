"""FastAPI control plane for the DDoS defense system."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from api.routes import admin, blocks, health, stats
from core.config_loader import load as load_config
from core.logging import get_logger

log = get_logger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.cfg = cfg
    log.info(f"API starting: {cfg.get('app', {}).get('name', 'ddos-defense')}")
    yield
    log.info("API shutting down")


app = FastAPI(
    title="DDoS Defense API",
    description="Control plane for the DDoS defense system",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/metrics", make_asgi_app())

app.include_router(health.router, tags=["health"])
app.include_router(blocks.router, prefix="/blocks", tags=["blocks"])
app.include_router(stats.router, prefix="/stats", tags=["stats"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])


@app.get("/")
def root():
    return {
        "service": "ddos-defense",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/blocks/",
            "/blocks/block",
            "/blocks/{ip}",
            "/blocks/{ip}/unblock",
            "/stats/summary",
            "/stats/history",
            "/stats/asns",
            "/admin/state",
            "/admin/purge-expired",
            "/admin/reload-model",
            "/admin/clear-state",
        ],
    }