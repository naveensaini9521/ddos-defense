"""Shared type aliases."""
from __future__ import annotations

from typing import NewType

IP = NewType("IP", str)
CIDR = NewType("CIDR", str)
Timestamp = NewType("Timestamp", float)
