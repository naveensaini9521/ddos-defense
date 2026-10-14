"""Automatic hyperparameter tuning for XGBoost and RandomForest.

Runs RandomizedSearchCV (default) or GridSearchCV to find the best
hyperparameters for the given training data. Saves the winning config
to ml/models/tuned_params.json so it can be reused.

Usage:
    python3 -m ml.hyperparameter_tuner --model xgb --search random --n-iter 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

from core.logging import get_logger
from ml.augmenter import augment_training_data
from ml.data_loader import load_training_data

log = get_logger("ml.hyperparameter_tuner")

OUT_FILE = Path("ml/models/tuned_params.json")


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

XGB_PARAM_GRID = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [3, 4, 6, 8, 10],
    "learning_rate":     [0.01, 0.05, 0.1, 0.2],
    "subsample":         [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.7, 0.8, 0.9, 1.0],
    "min_child_weight":  [1, 3, 5, 7],
    "gamma":             [0, 0.1, 0.3],
}

RF_PARAM_GRID = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [None, 6, 12, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", None],
    "class_weight":      ["balanced", "balanced_subsample", None],
}


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def build_rf():
    return RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def tune(
    X: np.ndarray,
    y: np.ndarray,
    model: str = "xgb",
    search: str = "random",
    n_iter: int = 20,
    cv_folds: int = 3,
    random_state: int = 42,
) -> dict:
    """Run hyperparameter search and return best params + score.

    Args:
        X, y: training data
        model: 'xgb' or 'rf'
        search: 'random' or 'grid'
        n_iter: number of random samples (ignored for grid)
        cv_folds: cross-validation folds
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if min(n_pos, n_neg) < cv_folds:
        log.warning(f"only {min(n_pos, n_neg)} minority samples < {cv_folds} folds; "
                    f"reducing to {max(2, min(n_pos, n_neg))} folds")
        cv_folds = max(2, min(n_pos, n_neg))

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score, pos_label=1, zero_division=0)

    if model == "xgb":
        estimator = build_xgb()
        grid = XGB_PARAM_GRID
        # scale_pos_weight is computed from data, not searched
        estimator.set_params(scale_pos_weight=max(n_neg / max(n_pos, 1), 1.0))
    elif model == "rf":
        estimator = build_rf()
        grid = RF_PARAM_GRID
    else:
        raise ValueError(f"unknown model: {model}")

    log.info(f"tuning {model}: search={search}, "
             f"param space size={_grid_size(grid)}, "
             f"cv_folds={cv_folds}, n_iter={n_iter}")

    t0 = time.time()

    if search == "grid":
        searcher = GridSearchCV(
            estimator, grid,
            scoring=scorer, cv=cv,
            n_jobs=-1, verbose=0,
        )
    else:
        searcher = RandomizedSearchCV(
            estimator, grid,
            n_iter=min(n_iter, _grid_size(grid)),
            scoring=scorer, cv=cv,
            random_state=random_state,
            n_jobs=-1, verbose=0,
        )

    searcher.fit(X, y)

    elapsed = time.time() - t0

    best = searcher.best_params_
    best_score = float(searcher.best_score_)

    log.info(f"best f1={best_score:.3f}  ({elapsed:.1f}s)")
    log.info(f"best params: {best}")

    return {
        "model": model,
        "search": search,
        "n_iter": n_iter,
        "cv_folds": cv_folds,
        "best_score": best_score,
        "best_params": best,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": time.time(),
    }


def _grid_size(grid: dict) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_tuned(result: dict) -> Path:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if OUT_FILE.exists():
        with OUT_FILE.open() as f:
            existing = json.load(f)
    existing[result["model"]] = result
    with OUT_FILE.open("w") as f:
        json.dump(existing, f, indent=2)
    log.info(f"saved tuned params to {OUT_FILE}")
    return OUT_FILE


def load_tuned(model: str = "xgb") -> dict | None:
    """Load previously tuned params for a model, if any."""
    if not OUT_FILE.exists():
        return None
    with OUT_FILE.open() as f:
        data = json.load(f)
    return data.get(model)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparameter tuning")
    p.add_argument("--model", choices=["xgb", "rf"], default="xgb")
    p.add_argument("--search", choices=["random", "grid"], default="random")
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--augment", action="store_true",
                   help="augment training data before tuning")
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

    result = tune(
        X, y,
        model=args.model,
        search=args.search,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
    )

    save_tuned(result)

    print()
    print(f"=== TUNING RESULT ({args.model}) ===")
    print(f"best F1:    {result['best_score']:.4f}")
    print(f"elapsed:    {result['elapsed_seconds']}s")
    print(f"best params:")
    for k, v in result["best_params"].items():
        print(f"  {k} = {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())