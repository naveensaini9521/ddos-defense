"""FastAPI app entrypoint."""
from fastapi import FastAPI

from api.routes import admin, blocks, stats

app = FastAPI(title="ddos-defense API")
app.include_router(blocks.router, prefix="/blocks", tags=["blocks"])
app.include_router(stats.router, prefix="/stats", tags=["stats"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
