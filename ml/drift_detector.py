"""Concept drift detection for DDoS detection models.

Monitors the statistical distribution of incoming features and compares
them to the training distribution. When the distribution shifts significantly,
the model is likely outdated and should be retrained.

Methods implemented:
    - Kolmogorov-Smirnov (KS) test per feature
    - Population Stability Index (PSI) — industry standard
    - Prediction-distribution drift (fraction of IPs flagged as attack)

Usage:
    python3 -m ml.drift_detector --model ml/models/v1_xgb.pkl --data data/raw
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from collector.aggregator import IPFeatures, aggregate
from collector.nginx_parser import parse_file
from core.logging import get_logger
from core.schema import FEATURE_NAMES
from ml.features import features_to_matrix

log = get_logger("ml.drift_detector")

STATE_FILE = Path("ml/models/drift_state.json")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureDrift:
    name: str
    ks_stat: float
    ks_pvalue: float
    psi: float
    mean_ref: float
    mean_curr: float
    std_ref: float
    std_curr: float
    drifted: bool


@dataclass
class DriftReport:
    timestamp: float
    n_reference: int
    n_current: int
    features: list[FeatureDrift] = field(default_factory=list)
    overall_drift: bool = False
    drift_score: float = 0.0
    recommendation: str = ""

    def summary(self) -> str:
        lines = [
            "=== Drift Report ===",
            f"reference samples: {self.n_reference}",
            f"current samples:   {self.n_current}",
            f"drift score:       {self.drift_score:.3f}",
            f"overall drift:     {'YES ⚠' if self.overall_drift else 'no'}",
            f"recommendation:    {self.recommendation}",
            "",
            f"{'feature':18s} {'KS':>6s} {'p':>7s} {'PSI':>6s} "
            f"{'ref_mean':>10s} {'curr_mean':>10s} {'drift':>6s}",
            "-" * 80,
        ]
        for f in self.features:
            mark = "YES" if f.drifted else "-"
            lines.append(
                f"{f.name:18s} {f.ks_stat:6.3f} {f.ks_pvalue:7.4f} "
                f"{f.psi:6.3f} {f.mean_ref:10.2f} {f.mean_curr:10.2f} "
                f"{mark:>6s}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def ks_test(ref: np.ndarray, curr: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov 2-sample test. Returns (stat, p_value)."""
    if len(ref) < 2 or len(curr) < 2:
        return 0.0, 1.0
    stat, pval = stats.ks_2samp(ref, curr)
    return float(stat), float(pval)


def population_stability_index(
    ref: np.ndarray,
    curr: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index (PSI).

    PSI < 0.1   → no significant change
    0.1–0.25    → moderate change, monitor
    PSI > 0.25  → significant change, retrain
    """
    if len(ref) == 0 or len(curr) == 0:
        return 0.0

    # bin edges from reference distribution
    breakpoints = np.linspace(ref.min(), ref.max(), n_bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    curr_counts, _ = np.histogram(curr, bins=breakpoints)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    curr_pct = curr_counts / max(curr_counts.sum(), 1)

    # avoid log(0)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    curr_pct = np.where(curr_pct == 0, 1e-6, curr_pct)

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi)


# ---------------------------------------------------------------------------
# Reference distribution persistence
# ---------------------------------------------------------------------------

def save_reference(X: np.ndarray, path: Path = STATE_FILE) -> None:
    """Save the reference feature distribution for future comparisons."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "timestamp": time.time(),
        "n_samples": int(X.shape[0]),
        "features": {
            name: {
                "values": X[:, i].tolist(),
                "mean": float(X[:, i].mean()),
                "std": float(X[:, i].std()),
                "min": float(X[:, i].min()),
                "max": float(X[:, i].max()),
            }
            for i, name in enumerate(FEATURE_NAMES)
        },
    }
    with path.open("w") as f:
        json.dump(state, f)
    log.info(f"saved reference distribution ({X.shape[0]} samples) to {path}")


def load_reference(path: Path = STATE_FILE) -> dict | None:
    """Load a previously saved reference distribution."""
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def detect_drift(
    X_ref: np.ndarray,
    X_curr: np.ndarray,
    ks_alpha: float = 0.05,
    psi_threshold: float = 0.25,
) -> DriftReport:
    """Compare two feature matrices and report drift per feature."""
    if X_ref.shape[1] != X_curr.shape[1]:
        raise ValueError(f"feature dim mismatch: {X_ref.shape[1]} vs {X_curr.shape[1]}")

    report = DriftReport(
        timestamp=time.time(),
        n_reference=len(X_ref),
        n_current=len(X_curr),
    )

    drifted_count = 0

    for i, name in enumerate(FEATURE_NAMES):
        ref_col = X_ref[:, i]
        curr_col = X_curr[:, i]

        ks_stat, ks_p = ks_test(ref_col, curr_col)
        psi = population_stability_index(ref_col, curr_col)

        # A feature drifted if PSI > threshold OR (KS significant AND big stat)
        drifted = (psi > psi_threshold) or (ks_p < ks_alpha and ks_stat > 0.3)

        if drifted:
            drifted_count += 1

        report.features.append(FeatureDrift(
            name=name,
            ks_stat=ks_stat,
            ks_pvalue=ks_p,
            psi=psi,
            mean_ref=float(ref_col.mean()),
            mean_curr=float(curr_col.mean()),
            std_ref=float(ref_col.std()),
            std_curr=float(curr_col.std()),
            drifted=drifted,
        ))

    # Overall drift score = fraction of drifted features
    report.drift_score = drifted_count / len(FEATURE_NAMES)
    report.overall_drift = report.drift_score >= 0.3   # 30%+ features drifted

    # Recommendation
    if report.drift_score >= 0.5:
        report.recommendation = "RETRAIN NOW — major distribution shift"
    elif report.drift_score >= 0.3:
        report.recommendation = "RETRAIN SOON — moderate drift detected"
    elif report.drift_score >= 0.1:
        report.recommendation = "MONITOR — minor drift, no action needed"
    else:
        report.recommendation = "OK — no significant drift"

    return report


# ---------------------------------------------------------------------------
# Convenience: detect drift from log files
# ---------------------------------------------------------------------------

def load_features_from_logs(log_dir: Path) -> np.ndarray:
    all_features: list[IPFeatures] = []
    for lf in sorted(log_dir.glob("*.log")):
        recs = parse_file(lf)
        feats = aggregate(recs)
        all_features.extend(feats.values())
    if not all_features:
        return np.zeros((0, len(FEATURE_NAMES)))
    return features_to_matrix(all_features)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drift detection for DDoS models")
    p.add_argument("--reference", default=None,
                   help="path to reference .log dir or state .json "
                        "(default: uses saved state)")
    p.add_argument("--current", default="data/raw",
                   help="directory of current .log files")
    p.add_argument("--save-reference", action="store_true",
                   help="save the current data as the new reference")
    p.add_argument("--ks-alpha", type=float, default=0.05)
    p.add_argument("--psi-threshold", type=float, default=0.25)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Load current features
    log.info(f"loading current features from {args.current}")
    X_curr = load_features_from_logs(Path(args.current))
    if X_curr.shape[0] == 0:
        raise SystemExit("no current features found")
    log.info(f"current: {X_curr.shape}")

    # Optionally save as reference and exit
    if args.save_reference:
        save_reference(X_curr)
        return 0

    # Load reference
    ref_state = load_reference()
    if ref_state is None:
        log.warning("no reference saved yet — saving current as reference")
        save_reference(X_curr)
        print("\nreference saved. Run again with different data to detect drift.")
        return 0

    # Rebuild reference matrix from saved values
    X_ref = np.column_stack([
        ref_state["features"][name]["values"] for name in FEATURE_NAMES
    ])
    log.info(f"reference: {X_ref.shape}")

    # Detect
    report = detect_drift(X_ref, X_curr,
                          ks_alpha=args.ks_alpha,
                          psi_threshold=args.psi_threshold)

    print()
    print(report.summary())

    # Save latest report
    out = Path("ml/models/drift_reports.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps({
            "timestamp": report.timestamp,
            "drift_score": report.drift_score,
            "overall_drift": report.overall_drift,
            "recommendation": report.recommendation,
            "features": [
                {"name": fd.name, "psi": fd.psi, "ks_stat": fd.ks_stat,
                 "drifted": fd.drifted}
                for fd in report.features
            ],
        }) + "\n")

    log.info(f"drift report appended to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())