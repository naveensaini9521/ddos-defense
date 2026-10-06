"""
Consistent JSON logging across all modules.
Every module does: from core.logging import get_logger
"""
from __future__ import annotations

import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line — easy to grep, parse, ship."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing JSON to stdout, configured once."""
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(level)
        log.propagate = False
    return log