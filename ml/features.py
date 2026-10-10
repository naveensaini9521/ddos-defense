"""Feature engineering helpers for ML training and inference.

This module bridges the collector's IPFeatures objects and sklearn's
expected numpy arrays. It also provides the standard train/test split
used by ml/train.py.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from collector.aggregator import IPFeatures
from core.schema import FEATURE_NAMES


def features_to_matrix(features: Iterable[IPFeatures]) -> np.ndarray:
    """Convert an iterable of IPFeatures into a 2D numpy array.

    Row order matches the order of FEATURE_NAMES.
    """
    rows = [f.to_vector() for f in features]
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float)
    return np.asarray(rows, dtype=float)


def labels_to_array(labels: Iterable[int]) -> np.ndarray:
    """Convert label iterable (0 = normal, 1 = attack) to a numpy array."""
    return np.asarray(list(labels), dtype=int)


def feature_names() -> list[str]:
    """Return the canonical ordered feature names."""
    return list(FEATURE_NAMES)


def describe(features: Iterable[IPFeatures]) -> dict[str, dict[str, float]]:
    """Return per-feature summary statistics — handy for debugging."""
    X = features_to_matrix(features)
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i] if X.shape[0] else np.array([])
        if col.size == 0:
            out[name] = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
            continue
        out[name] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
        }
    return out