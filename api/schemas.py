"""Pydantic schemas for API requests and responses."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BlockEntryOut(BaseModel):
    ip: str
    since: float
    until: float
    reason: str
    confidence: float
    strike: int
    model: str
    active: bool
    permanent: bool
    ttl_remaining: float
    scope: str = "ip"
    subnet: str | None = None
    asn: int | None = None
    ja3: str | None = None


class BlockListOut(BaseModel):
    count: int
    blocks: list[BlockEntryOut]


class BlockRequest(BaseModel):
    ip: str = Field(..., description="IP or CIDR to block")
    ttl: int = Field(60, description="TTL in seconds; -1 for permanent")
    reason: str = Field("manual", description="Reason for the block")
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class UnblockRequest(BaseModel):
    ip: str


class BlockResultOut(BaseModel):
    ok: bool
    ip: str
    action: Literal["blocked", "unblocked", "skipped", "not-found", "error"]
    detail: str = ""


class StatsOut(BaseModel):
    active_blocks: int
    total_records: int
    permanent_blocks: int
    uptime_seconds: int
    state_backend: str = "unknown"


class HealthOut(BaseModel):
    status: Literal["ok", "degraded", "down"]
    components: dict[str, str]