"""Robust cross-validation for DDoS detection models.

Provides:
    - Stratified K-Fold CV (standard, keeps class balance)
    - Time-Series CV (for temporal data — no future leakage)
    - Leave-One-Out (LOO) for very small datasets
    - Bootstrap confidence intervals

Reports mean ± std for F1, precision, recall, ROC-AUC.

Usage:
    python3 -m ml.cross_validator --model xgb --folds 5
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, TimeSeriesSplit

from core.logging import get_logger
from ml.augmenter import augment_training_data
from ml.data_loader import load_training_data
from ml.hyperparameter_tuner import load_tuned

log = get_logger("ml.cross_validator")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    f1: float
    precision: float
    recall: float
    roc_auc: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass
class CVReport:
    model_name: str
    method: str
    n_folds: int
    folds: list[FoldResult] = field(default_factory=list)

    # Aggregates
    f1_mean: float = 0.0
    f1_std: float = 0.0
    precision_mean: float = 0.0
    recall_mean: float = 0.0
    roc_auc_mean: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== CV Report: {self.model_name} ===",
            f"method:    {self.method}",
            f"n_folds:   {self.n_folds}",
            "",
            f"F1        : {self.f1_mean:.4f} ± {self.f1_std:.4f}",
            f"Precision : {self.precision_mean:.4f}",
            f"Recall    : {self.recall_mean:.4f}",
            f"ROC-AUC   : {self.roc_auc_mean:.4f}",
            "",
            "Per-fold:",
        ]
        for f in self.folds:
            lines.append(
                f"  fold {f.fold}: f1={f.f1:.3f} "
                f"p={f.precision:.3f} r={f.recall:.3f} "
                f"auc={f.roc_auc:.3f} "
                f"(tp={f.tp} fp={f.fp} tn={f.tn} fn={f.fn})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model: str, tuned: bool = True):
    """Build a fresh model instance with optional tuned params."""
    params = {}
    if tuned:
        t = load_tuned(model)
        if t:
            params = dict(t["best_params"])

    if model == "xgb":
        from xgboost import XGBClassifier
        base = dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        base.update(params)
        return XGBClassifier(**base)

    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        base = dict(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        base.update(params)
        return RandomForestClassifier(**base)

    raise ValueError(f"unknown model: {model}")


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------

def get_splitter(method: str, n_folds: int):
    if method == "stratified":
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    if method == "time-series":
        return TimeSeriesSplit(n_splits=n_folds)
    if method == "loo":
        return LeaveOneOut()
    raise ValueError(f"unknown CV method: {method}")


# ---------------------------------------------------------------------------
# Core CV loop
# ---------------------------------------------------------------------------

def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model: str = "xgb",
    method: str = "stratified",
    n_folds: int = 5,
    tuned: bool = True,
) -> CVReport:
    """Run cross-validation and return a report."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if method == "stratified" and min(n_pos, n_neg) < n_folds:
        n_folds = max(2, min(n_pos, n_neg))
        log.warning(f"reducing folds to {n_folds} (minority class size)")

    splitter = get_splitter(method, n_folds)

    # Number of folds: use splitter's get_n_splits when available
    try:
        n_splits = splitter.get_n_splits(X, y)
    except TypeError:
        try:
            n_splits = splitter.get_n_splits()
        except TypeError:
            n_splits = n_folds
            
    report = CVReport(model_name=model, method=method, n_folds=n_splits)

    f1s, precs, recalls, aucs = [], [], [], []

    for i, (tr_idx, te_idx) in enumerate(splitter.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if len(np.unique(y_tr)) < 2:
            log.warning(f"fold {i}: training set has only 1 class — skipping")
            continue

        model_obj = build_model(model, tuned=tuned)
        model_obj.fit(X_tr, y_tr)

        # Predictions
        y_pred = model_obj.predict(X_te)
        # Probability of attack (class 1)
        try:
            y_prob = model_obj.predict_proba(X_te)[:, 1]
        except (AttributeError, IndexError):
            y_prob = y_pred.astype(float)

        # Metrics
        f1 = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_te, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_te, y_pred, pos_label=1, zero_division=0)

        try:
            auc = roc_auc_score(y_te, y_prob)
        except ValueError:
            auc = 0.0

        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        report.folds.append(FoldResult(
            fold=i,
            n_train=len(X_tr),
            n_test=len(X_te),
            f1=float(f1),
            precision=float(prec),
            recall=float(rec),
            roc_auc=float(auc),
            tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        ))

        f1s.append(f1)
        precs.append(prec)
        recalls.append(rec)
        aucs.append(auc)

    if report.folds:
        report.f1_mean = float(np.mean(f1s))
        report.f1_std = float(np.std(f1s))
        report.precision_mean = float(np.mean(precs))
        report.recall_mean = float(np.mean(recalls))
        report.roc_auc_mean = float(np.mean(aucs))

    return report


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    X: np.ndarray,
    y: np.ndarray,
    model: str = "xgb",
    n_bootstrap: int = 100,
    random_state: int = 42,
) -> dict:
    """Bootstrap the F1 score to estimate confidence interval."""
    rng = np.random.default_rng(random_state)
    n = len(X)
    f1s = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_b, y_b = X[idx], y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            m = build_model(model)
            m.fit(X_b, y_b)
            pred = m.predict(X_b)
            f1s.append(f1_score(y_b, pred, pos_label=1, zero_division=0))
        except Exception:
            continue

    if not f1s:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}

    f1s = np.asarray(f1s)
    return {
        "mean": float(f1s.mean()),
        "std": float(f1s.std()),
        "lower": float(np.percentile(f1s, 2.5)),
        "upper": float(np.percentile(f1s, 97.5)),
        "n": len(f1s),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-validation for DDoS models")
    p.add_argument("--model", choices=["xgb", "rf"], default="xgb")
    p.add_argument("--method", choices=["stratified", "time-series", "loo"],
                   default="stratified")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--no-tuned", action="store_true",
                   help="ignore tuned params")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--bootstrap", action="store_true",
                   help="also run bootstrap CI")
    p.add_argument("--data", default="data/raw")
    p.add_argument("--labels", default="data/labeled/labels.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    log.info(f"loading training data (augment={args.augment})")
    X, y, _ = load_training_data(raw_dir=args.data, label_file=args.labels)

    if args.augment:
        r = augment_training_data(X, y, target_ratio=1.0)
        X, y = r.X, r.y
        log.info(f"augmented: shape={X.shape}, pos={int(y.sum())}")

    report = cross_validate(
        X, y,
        model=args.model,
        method=args.method,
        n_folds=args.folds,
        tuned=not args.no_tuned,
    )

    print()
    print(report.summary())

    if args.bootstrap:
        print()
        print("--- Bootstrap CI (95%) ---")
        ci = bootstrap_ci(X, y, model=args.model, n_bootstrap=100)
        print(f"F1 mean : {ci['mean']:.4f}")
        print(f"95% CI  : [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        print(f"n_boot  : {ci['n']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())