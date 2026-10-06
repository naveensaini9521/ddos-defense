"""
Load config/config.yaml and expose typed accessors.

Usage:
    from core.config_loader import get
    threshold = get("ml.threshold", default=0.7)
"""
from __future__ import annotations

from pathlib import Path

import yaml

_CACHE: dict | None = None
_DEFAULT_PATH = Path("config/config.yaml")


def load(path: str | Path | None = None, reload: bool = False) -> dict:
    """Load (and cache) the YAML config."""
    global _CACHE
    if _CACHE is None or reload:
        p = Path(path) if path else _DEFAULT_PATH
        if not p.exists():
            _CACHE = {}
        else:
            with p.open() as f:
                _CACHE = yaml.safe_load(f) or {}
    return _CACHE


def get(key: str, default=None):
    """Dotted-path accessor: get('ml.threshold')."""
    cur = load()
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur