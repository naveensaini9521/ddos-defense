"""Emit ground-truth labels correlated with traffic timestamps."""
# TODO: write data/labeled/attacks.jsonl
"""
simulator/labeler.py

Small helper library for reading labels written by traffic_generator.

Usage:
    from simulator.labeler import load_labels
    df = load_labels()   # pandas DataFrame
"""
from __future__ import annotations

import json
from pathlib import Path

LABEL_FILE = Path("data/labeled/labels.jsonl")


def load_labels(path: str | Path = LABEL_FILE) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_labels_df(path: str | Path = LABEL_FILE):
    import pandas as pd
    rows = load_labels(path)
    return pd.DataFrame(rows)


def clear_labels(path: str | Path = LABEL_FILE) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()