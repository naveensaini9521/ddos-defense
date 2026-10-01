"""Pydantic request/response models."""
from pydantic import BaseModel


class BlockOut(BaseModel):
    ip: str
    since: float
    until: float
    reason: str
