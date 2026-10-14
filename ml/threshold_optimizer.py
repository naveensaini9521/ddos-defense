"""Threshold optimizer for DDoS detection.

Sweeps classification thresholds from 0.05 to 0.95, evaluating
precision/recall/F1 at each. Picks the best threshold according to
a chosen objective (f1, precision, recall, or custom cost matrix).

Usage:
    python3 -m ml.threshold_optimizer --model xgb
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from core.logging import get_logger
from ml.augmenter import augment_training_data
from ml.cross_validator import build_model
from ml.data_loader import load_training_data

log = get_logger("ml.threshold_optimizer")

OUT_FILE = Path("ml/models/thresholds.json")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ThresholdPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    fp: int
    fn: int
    tp: int
    tn: int


@dataclass
class OptimizationResult:
    model: str
    objective: str
    best_threshold: float
    best_score: float
    points: list[ThresholdPoint]


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[ThresholdPoint]:
    """Compute precision/recall/F1 at each threshold."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    points: list[ThresholdPoint] = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        points.append(ThresholdPoint(
            threshold=float(t),
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
            fp=fp, fn=fn, tp=tp, tn=tn,
        ))

    return points


# ---------------------------------------------------------------------------
# Objective selection
# ---------------------------------------------------------------------------

def pick_best(
    points: list[ThresholdPoint],
    objective: str = "f1",
    fp_cost: float = 5.0,
    fn_cost: float = 1.0,
) -> ThresholdPoint:
    """Pick best threshold per objective.

    Objectives:
        - 'f1'         : maximize F1
        - 'precision'  : maximize precision (few false positives)
        - 'recall'     : maximize recall (catch all attacks)
        - 'cost'       : minimize fp_cost * FP + fn_cost * FN
        - 'youden'     : maximize (recall + specificity - 1) = Youden's J
    """
    if not points:
        raise ValueError("no threshold points to choose from")

    if objective == "f1":
        return max(points, key=lambda p: p.f1)
    if objective == "precision":
        # prefer highest precision; break ties by recall
        return max(points, key=lambda p: (p.precision, p.recall))
    if objective == "recall":
        return max(points, key=lambda p: (p.recall, p.precision))
    if objective == "cost":
        return min(points, key=lambda p: fp_cost * p.fp + fn_cost * p.fn)
    if objective == "youden":
        # specificity = TN / (TN + FP)
        def youden(p):
            spec = p.tn / max(p.tn + p.fp, 1)
            return p.recall + spec - 1.0
        return max(points, key=youden)

    raise ValueError(f"unknown objective: {objective}")


# ---------------------------------------------------------------------------
# Full pipeline: train + sweep + pick
# ---------------------------------------------------------------------------

def optimize_threshold(
    X: np.ndarray,
    y: np.ndarray,
    model: str = "xgb",
    objective: str = "f1",
    tuned: bool = True,
    fp_cost: float = 5.0,
    fn_cost: float = 1.0,
) -> OptimizationResult:
    """Train a model on X, sweep thresholds on the same X (in-sample).

    For a proper out-of-sample sweep, use cross_validator to get probs.
    This in-sample version is fast and good for a first estimate.
    """
    log.info(f"training {model} for threshold sweep on {X.shape[0]} samples")

    m = build_model(model, tuned=tuned)
    m.fit(X, y)

    # Out-of-fold probabilities (simple 5-fold if enough data)
    y_prob = _out_of_fold_probs(X, y, model=model, tuned=tuned)
    if y_prob is None:
        # fallback to in-sample probs
        log.warning("using in-sample probabilities (dataset too small for OOF)")
        y_prob = m.predict_proba(X)[:, 1]

    points = sweep_thresholds(y, y_prob)
    best = pick_best(points, objective=objective,
                     fp_cost=fp_cost, fn_cost=fn_cost)

    log.info(f"best threshold={best.threshold:.2f} ({objective}: {best.f1:.3f})")

    return OptimizationResult(
        model=model,
        objective=objective,
        best_threshold=best.threshold,
        best_score=best.f1 if objective == "f1" else (
            best.precision if objective == "precision" else (
            best.recall if objective == "recall" else best.f1)),
        points=points,
    )


def _out_of_fold_probs(
    X: np.ndarray,
    y: np.ndarray,
    model: str,
    tuned: bool,
    n_folds: int = 5,
) -> np.ndarray | None:
    """Return out-of-fold P(attack) for each sample via K-fold CV."""
    from sklearn.model_selection import StratifiedKFold

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    folds = min(n_folds, min(n_pos, n_neg))
    if folds < 2:
        return None

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    y_prob = np.zeros(len(y), dtype=float)

    for tr_idx, te_idx in skf.split(X, y):
        m = build_model(model, tuned=tuned)
        m.fit(X[tr_idx], y[tr_idx])
        y_prob[te_idx] = m.predict_proba(X[te_idx])[:, 1]

    return y_prob


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_thresholds(result: OptimizationResult) -> Path:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if OUT_FILE.exists():
        with OUT_FILE.open() as f:
            existing = json.load(f)

    existing[result.model] = {
        "best_threshold": result.best_threshold,
        "best_score": result.best_score,
        "objective": result.objective,
        "sweep": [
            {
                "threshold": p.threshold,
                "f1": p.f1,
                "precision": p.precision,
                "recall": p.recall,
                "fp": p.fp,
                "fn": p.fn,
            }
            for p in result.points
        ],
    }

    with OUT_FILE.open("w") as f:
        json.dump(existing, f, indent=2)

    log.info(f"saved thresholds to {OUT_FILE}")
    return OUT_FILE


def load_threshold(model: str = "xgb") -> float | None:
    """Load the saved best threshold for a model."""
    if not OUT_FILE.exists():
        return None
    with OUT_FILE.open() as f:
        data = json.load(f)
    entry = data.get(model)
    return entry["best_threshold"] if entry else None


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_sweep(result: OptimizationResult) -> None:
    print()
    print(f"=== Threshold Sweep ({result.model}, objective={result.objective}) ===")
    print(f"{'thresh':>7s} {'precision':>10s} {'recall':>8s} {'f1':>6s}  {'fp':>4s} {'fn':>4s}")
    print('-' * 55)
    for p in result.points:
        marker = "  <--" if abs(p.threshold - result.best_threshold) < 0.001 else ""
        print(f"{p.threshold:>7.2f} {p.precision:>10.3f} {p.recall:>8.3f} "
              f"{p.f1:>6.3f}  {p.fp:>4d} {p.fn:>4d}{marker}")

    print()
    print(f"BEST: threshold={result.best_threshold:.2f} "
          f"(score={result.best_score:.3f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Threshold optimizer")
    p.add_argument("--model", choices=["xgb", "rf"], default="xgb")
    p.add_argument("--objective",
                   choices=["f1", "precision", "recall", "cost", "youden"],
                   default="f1")
    p.add_argument("--no-tuned", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--fp-cost", type=float, default=5.0,
                   help="cost of a false positive (block legit user)")
    p.add_argument("--fn-cost", type=float, default=1.0,
                   help="cost of a false negative (miss attacker)")
    p.add_argument("--data", default="data/raw")
    p.add_argument("--labels", default="data/labeled/labels.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    log.info("loading training data")
    X, y, _ = load_training_data(raw_dir=args.data, label_file=args.labels)

    if args.augment:
        r = augment_training_data(X, y, target_ratio=1.0)
        X, y = r.X, r.y
        log.info(f"augmented: shape={X.shape}, pos={int(y.sum())}")

    result = optimize_threshold(
        X, y,
        model=args.model,
        objective=args.objective,
        tuned=not args.no_tuned,
        fp_cost=args.fp_cost,
        fn_cost=args.fn_cost,
    )

    print_sweep(result)
    save_thresholds(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())