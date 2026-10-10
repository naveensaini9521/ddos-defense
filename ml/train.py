"""Train ML models for DDoS detection.

Supports three model types:
    - iso : IsolationForest  (unsupervised, cold start)
    - rf  : RandomForest     (supervised)
    - xgb : XGBoost          (supervised, highest accuracy)

Supports two dataset formats:
    - Directory of .log files (uses ml.data_loader.load_training_data)
    - CSV file (uses ml.data_loader.load_features_from_csv)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from core.logging import get_logger
from core.schema import FEATURE_NAMES
from ml.data_loader import load_features_from_csv, load_training_data

log = get_logger("ml.train")

MODEL_DIR = Path("ml/models")
REGISTRY = MODEL_DIR / "registry.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_arg: str, labels_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load training data from either a CSV file, a directory with
    features.csv, or a directory of .log files."""
    path = Path(data_arg)

    # Case 1: direct CSV file
    if path.is_file() and path.suffix == ".csv":
        log.info(f"loading CSV dataset: {path}")
        return load_features_from_csv(path)

    # Case 2: directory containing features.csv
    if path.is_dir() and (path / "features.csv").exists():
        csv_path = path / "features.csv"
        log.info(f"loading CSV dataset: {csv_path}")
        return load_features_from_csv(csv_path)

    # Case 3: directory of .log files (heuristic labels)
    log.info(f"loading .log dataset from {path} (labels: {labels_path})")
    return load_training_data(
        raw_dir=path,
        label_file=labels_path,
        use_heuristic_if_missing=True,
    )


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------

def train_iso(X: np.ndarray, contamination: float = 0.1) -> IsolationForest:
    log.info(f"training IsolationForest on {X.shape[0]} samples, "
             f"contamination={contamination}")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    log.info("IsolationForest training complete")
    return model


def train_rf(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    log.info(f"training RandomForest on {X.shape[0]} samples, "
             f"class balance: {np.bincount(y)}")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    log.info("RandomForest training complete")
    return model


def train_xgb(X: np.ndarray, y: np.ndarray):
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise SystemExit("xgboost not installed. Run: pip install xgboost") from e

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale = max(n_neg / max(n_pos, 1), 1.0)

    log.info(f"training XGBoost on {X.shape[0]} samples, "
             f"pos={n_pos} neg={n_neg} scale_pos_weight={scale:.2f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    log.info("XGBoost training complete")
    return model


# ---------------------------------------------------------------------------
# Evaluation + persistence
# ---------------------------------------------------------------------------

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True,
                                   zero_division=0)
    log.info(f"evaluation: precision={report['macro avg']['precision']:.3f}, "
             f"recall={report['macro avg']['recall']:.3f}, "
             f"f1={report['macro avg']['f1-score']:.3f}")
    return report


def save_model(model, version: str, metrics: dict | None = None) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{version}.pkl"
    joblib.dump(model, path)
    log.info(f"model saved to {path}")

    registry: dict = {"current": version, "versions": {}}
    if REGISTRY.exists():
        with REGISTRY.open() as f:
            registry = json.load(f)

    registry["current"] = version
    registry["versions"][version] = {
        "path": str(path),
        "features": list(FEATURE_NAMES),
        "trained_at": time.time(),
        "metrics": metrics or {},
        "type": type(model).__name__,
    }
    with REGISTRY.open("w") as f:
        json.dump(registry, f, indent=2)
    log.info(f"registry updated: current={version}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DDoS detection models")
    p.add_argument("--data", default="data/raw",
                   help="CSV file, dir with features.csv, or dir of .log files")
    p.add_argument("--labels", default="data/labeled/labels.jsonl")
    p.add_argument("--model", choices=["iso", "rf", "xgb"], default="xgb")
    p.add_argument("--version", default="v1")
    p.add_argument("--contamination", type=float, default=0.1,
                   help="IsolationForest only")
    p.add_argument("--test-size", type=float, default=0.3)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    X, y, ips = load_dataset(args.data, args.labels)
    log.info(f"feature matrix: shape={X.shape}, "
             f"positives={int(y.sum())}, negatives={int((y == 0).sum())}")

    # IsolationForest — unsupervised
    if args.model == "iso":
        model = train_iso(X, contamination=args.contamination)
        save_model(model, args.version, {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "contamination": args.contamination,
        })
        log.info("training run complete")
        return 0

    # Supervised models
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0 or n_neg == 0:
        log.warning(f"single-class labels (pos={n_pos}, neg={n_neg}); "
                    f"training on full set")
        X_tr, y_tr = X, y
        X_te, y_te = X, y
    else:
        stratify = y if min(n_pos, n_neg) >= 2 else None
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=args.test_size,
                random_state=42, stratify=stratify,
            )
        except ValueError as e:
            log.warning(f"train_test_split failed ({e}); full set")
            X_tr, y_tr = X, y
            X_te, y_te = X, y

    if args.model == "rf":
        model = train_rf(X_tr, y_tr)
    elif args.model == "xgb":
        model = train_xgb(X_tr, y_tr)
    else:
        raise SystemExit(f"unknown model: {args.model}")

    metrics = evaluate(model, X_te, y_te)
    metrics.update({
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_positives": n_pos,
        "n_negatives": n_neg,
    })
    save_model(model, args.version, metrics)

    log.info("training run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())