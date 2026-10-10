"""Load and prepare labeled training data.

Supports two label sources:
    1. data/labeled/labels.jsonl   — one JSON object per line
    2. data/labeled/*.csv          — flat CSV with 'ip' + 'label' columns

Merges labels with features extracted from data/raw/*.log files.

Usage:
    from ml.data_loader import load_training_data

    X, y, ips = load_training_data()          # default paths
    X, y, ips = load_training_data(raw_dir="data/raw",
                                   label_file="data/labeled/labels.jsonl")
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from collector.aggregator import IPFeatures, aggregate
from collector.nginx_parser import parse_file
from core.logging import get_logger
from core.schema import FEATURE_NAMES
from ml.features import features_to_matrix

log = get_logger("ml.data_loader")


# ---------------------------------------------------------------------------
# Feature extraction (re-uses existing collector pipeline)
# ---------------------------------------------------------------------------

def extract_features(raw_dir: str | Path = "data/raw") -> dict[str, IPFeatures]:
    """Parse every .log in raw_dir, aggregate per IP.

    Returns {ip: IPFeatures}.
    """
    raw_dir = Path(raw_dir)
    log_files = sorted(raw_dir.glob("*.log"))
    if not log_files:
        raise FileNotFoundError(f"no .log files in {raw_dir}")

    by_ip: dict[str, IPFeatures] = {}
    for lf in log_files:
        records = parse_file(lf)
        feats = aggregate(records)
        for ip, f in feats.items():
            # If IP appears in multiple files, keep the one with most requests
            if ip not in by_ip or f.http_reqs > by_ip[ip].http_reqs:
                by_ip[ip] = f
        log.info(f"extracted {len(feats)} IPs from {lf.name}")

    log.info(f"total unique IPs: {len(by_ip)}")
    return by_ip


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_labels_jsonl(path: str | Path) -> dict[str, int]:
    """Load labels from a JSONL file.

    Expected format (one object per line):
        {"ip": "1.2.3.4", "label": 1}
        {"ip": "5.6.7.8", "label": 0}

    Also accepts simulator-style labels and expands them by mode:
        {"mode": "http_flood", ...}   -> label=1
        {"mode": "normal", ...}       -> label=0
    """
    path = Path(path)
    if not path.exists():
        log.warning(f"label file missing: {path}")
        return {}

    labels: dict[str, int] = {}
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"bad JSON at line {lineno}: {e}")
                continue

            # Case A: explicit IP label
            if "ip" in obj and "label" in obj:
                labels[obj["ip"]] = int(obj["label"])
                continue

            # Case B: simulator label (mode-based) — used only for logging
            if "mode" in obj:
                # Simulator labels don't map to a specific IP; skip
                continue

    log.info(f"loaded {len(labels)} explicit labels from {path.name}")
    return labels


def load_labels_csv(path: str | Path) -> dict[str, int]:
    """Load labels from a CSV with 'ip' and 'label' columns."""
    path = Path(path)
    if not path.exists():
        return {}

    labels: dict[str, int] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ip = row.get("ip") or row.get("source_ip")
            lab = row.get("label") or row.get("class")
            if ip is None or lab is None:
                continue
            labels[ip] = int(lab)

    log.info(f"loaded {len(labels)} labels from {path.name}")
    return labels


def load_labels(label_path: str | Path | None = None) -> dict[str, int]:
    """Load labels from a file, auto-detecting format."""
    if label_path is None:
        return {}

    p = Path(label_path)
    if p.suffix == ".jsonl":
        return load_labels_jsonl(p)
    elif p.suffix == ".csv":
        return load_labels_csv(p)
    else:
        # try JSONL first
        return load_labels_jsonl(p)


# ---------------------------------------------------------------------------
# Heuristic fallback (used only if no labels available)
# ---------------------------------------------------------------------------

def heuristic_labels(by_ip: dict[str, IPFeatures],
                     rate_percentile: float = 70.0,
                     error_percentile: float = 70.0) -> dict[str, int]:
    """Mark the top X% of IPs by rate/error as attack (percentile-based).

    Why percentile-based? Because absolute thresholds don't adapt to
    the window duration, traffic volume, or log size.
    """
    if not by_ip:
        return {}

    ips = list(by_ip.keys())
    rates = np.asarray([by_ip[ip].pkt_rate for ip in ips], dtype=float)
    errors = np.asarray([by_ip[ip].error_ratio for ip in ips], dtype=float)

    rate_threshold = np.percentile(rates, rate_percentile)
    error_threshold = np.percentile(errors, error_percentile)

    labels: dict[str, int] = {}
    for ip in ips:
        f = by_ip[ip]
        is_attack = (f.pkt_rate >= rate_threshold) or (f.error_ratio >= error_threshold)
        labels[ip] = 1 if is_attack else 0

    # Ensure both classes are present
    n_attack = sum(labels.values())
    if n_attack == 0:
        top = max(by_ip.values(), key=lambda f: f.pkt_rate)
        labels[top.ip] = 1
    elif n_attack == len(labels):
        bottom = min(by_ip.values(), key=lambda f: f.pkt_rate)
        labels[bottom.ip] = 0

    n_attack = sum(labels.values())
    log.warning(f"using heuristic labels ({n_attack} attack / {len(labels)} total, "
                f"rate_p{rate_percentile:.0f}={rate_threshold:.2f}, "
                f"err_p{error_percentile:.0f}={error_threshold:.2f})")
    return labels

# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------

def load_training_data(
    raw_dir: str | Path = "data/raw",
    label_file: str | Path | None = "data/labeled/labels.jsonl",
    use_heuristic_if_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, ips) ready for sklearn/XGBoost training.

    Args:
        raw_dir: directory of .log files
        label_file: optional path to labels (.jsonl or .csv)
        use_heuristic_if_missing: fall back to heuristic labels
    """
    # 1. Extract features
    by_ip = extract_features(raw_dir)
    if not by_ip:
        raise ValueError("no features extracted")

    # 2. Get labels
    labels = load_labels(label_file) if label_file else {}

    # 3. If labels missing for some IPs, decide whether to fall back
    missing = [ip for ip in by_ip if ip not in labels]
    if missing:
        log.info(f"{len(missing)} IPs have no label")

    if not labels and use_heuristic_if_missing:
        labels = heuristic_labels(by_ip)
    elif missing and use_heuristic_if_missing:
        # Fill missing labels with heuristic
        heur = heuristic_labels({ip: by_ip[ip] for ip in missing})
        labels.update(heur)

    # 4. Build matrix
    ips = list(by_ip.keys())
    X = features_to_matrix([by_ip[ip] for ip in ips])
    y = np.asarray([labels.get(ip, 0) for ip in ips], dtype=int)

    log.info(f"loaded {X.shape[0]} samples, {X.shape[1]} features, "
             f"{int(y.sum())} attack / {int((y==0).sum())} normal")

    return X, y, ips

def load_features_from_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load features + labels from a features.csv.

    Auto-detects feature columns from the CSV header (everything except
    window_index, scenario, ip, label). Raises if duplicate columns exist.
    """
    import csv as _csv

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")

    with path.open() as f:
        reader = _csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        raise ValueError(f"empty CSV: {path}")

    meta_cols = {"window_index", "scenario", "ip", "label"}
    feature_cols = [c for c in columns if c not in meta_cols]

    # Guard against duplicate column names
    seen: set[str] = set()
    dupes: set[str] = set()
    for c in feature_cols:
        if c in seen:
            dupes.add(c)
        seen.add(c)
    if dupes:
        raise ValueError(f"duplicate feature columns in CSV: {sorted(dupes)}")

    X = np.asarray(
        [[float(r[name]) for name in feature_cols] for r in rows],
        dtype=float,
    )
    y = np.asarray([int(r["label"]) for r in rows], dtype=int)
    ips = [r.get("ip", "unknown") for r in rows]

    log.info(
        f"loaded {len(X)} rows from {path.name} "
        f"({len(feature_cols)} features, "
        f"{int(y.sum())} attack / {int((y == 0).sum())} normal)"
    )
    return X, y, ips

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Inspect labeled training data")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--labels", default="data/labeled/labels.jsonl")
    args = p.parse_args()

    X, y, ips = load_training_data(raw_dir=args.raw, label_file=args.labels)

    print()
    print(f"samples: {len(ips)}")
    print(f"features: {X.shape[1]}")
    print(f"attack:  {int(y.sum())}")
    print(f"normal:  {int((y == 0).sum())}")
    print()
    print("per-IP labels:")
    for ip, label in zip(ips, y):
        tag = "ATTACK" if label == 1 else "normal"
        print(f"  {ip:15s} -> {tag}")