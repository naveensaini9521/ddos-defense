"""Block management endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import BlockEntryOut, BlockListOut, BlockResultOut
from blocker.block_manager import BlockManager
from core.schema import Decision

router = APIRouter()


def _manager() -> BlockManager:
    return BlockManager(start_daemon=False)


@router.get("/", response_model=BlockListOut)
def list_blocks(include_expired: bool = False):
    mgr = _manager()
    try:
        entries = mgr.state.all_blocks() if include_expired else mgr.state.active_blocks()
        blocks = [BlockEntryOut(**e.to_dict()) for e in entries]
        return BlockListOut(count=len(blocks), blocks=blocks)
    finally:
        mgr.shutdown()


@router.get("/{ip}", response_model=BlockEntryOut)
def get_block(ip: str):
    mgr = _manager()
    try:
        entry = mgr.state.get(ip)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"no block for {ip}")
        return BlockEntryOut(**entry.to_dict())
    finally:
        mgr.shutdown()


@router.post("/{ip}/unblock", response_model=BlockResultOut)
def unblock(ip: str):
    mgr = _manager()
    try:
        result = mgr.manual_unblock(ip)
        ok = result.get("action") == "unblocked"
        return BlockResultOut(
            ok=ok,
            ip=ip,
            action="unblocked" if ok else "not-found",
            detail=result.get("action", ""),
        )
    finally:
        mgr.shutdown()


@router.post("/block", response_model=BlockResultOut)
def block_ip(payload: dict):
    ip = payload.get("ip")
    if not ip:
        raise HTTPException(status_code=400, detail="ip required")

    ttl = int(payload.get("ttl", 60))
    reason = payload.get("reason", "manual")
    confidence = float(payload.get("confidence", 1.0))
    scope = payload.get("scope", "ip")
    asn = payload.get("asn")

    mgr = _manager()
    try:
        d = Decision(
            ip=ip,
            action="block",
            confidence=confidence,
            reason=reason,
            ttl_seconds=ttl,
        )
        if scope != "ip":
            object.__setattr__(d, "scope", scope)
        if asn is not None:
            object.__setattr__(d, "asn", int(asn))

        result = mgr.enforce(d)
        return BlockResultOut(
            ok=result.get("fw_ok", False),
            ip=ip,
            action="blocked" if result.get("action") == "blocked" else "error",
            detail=result.get("fw_command", ""),
        )
    finally:
        mgr.shutdown()