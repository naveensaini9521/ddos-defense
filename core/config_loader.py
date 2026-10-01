"""Load config/config.yaml and expose typed accessors."""
from __future__ import annotations

from pathlib import Path

import yaml

_CACHE: dict | None = None


def load(path: str | Path = "config/config.yaml") -> dict:
    global _CACHE
    if _CACHE is None:
        with open(path) as f:
            _CACHE = yaml.safe_load(f) or {}
    return _CACHE


def get(key: str, default=None):
    cfg = load()
    cur = cfg
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur
