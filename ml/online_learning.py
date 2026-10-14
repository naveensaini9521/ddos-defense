"""Online learning — incremental model updates from live feedback.

Provides:
    - Feedback storage (analyst corrections, auto-labeled samples)
    - Incremental updates (partial_fit where supported, else replay buffer)
    - Full-retrain triggers when buffer grows large
    - Rolling accuracy tracking

XGBoost doesn't support true partial_fit in older versions; we use a
replay buffer + periodic full retrain. RandomForest also retrains.

Usage:
    from ml.online_learning import FeedbackStore, OnlineLearner

    store = FeedbackStore()
    store.add_correction(ip="1.2.3.4", features=f, true_label=0)

    learner = OnlineLearner(store)
    if learner.should_retrain():
        learner.retrain()
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np

from collector.aggregator import IPFeatures
from core.logging import get_logger
from ml.data_loader import load_training_data
from ml.train import (
    save_model,
    train_random_forest,
    train_xgboost,
)

log = get_logger("ml.online_learning")

FEEDBACK_FILE = Path("data/labeled/feedback.jsonl")
REPLAY_LIMIT = 1000          # max samples kept in memory buffer
MIN_NEW_FOR_RETRAIN = 20     # retrain after N new corrections


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeedbackSample:
    """One labeled sample captured from live traffic or analyst input."""
    ts: float
    ip: str
    features: list[float]
    true_label: int          # 0 = normal, 1 = attack
    source: str              # "analyst" | "auto" | "ground_truth"
    note: str = ""


# ---------------------------------------------------------------------------
# Persistent feedback store
# ---------------------------------------------------------------------------

class FeedbackStore:
    """Append-only JSONL store of labeled samples."""

    def __init__(self, path: Path = FEEDBACK_FILE) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[FeedbackSample] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self._buffer.append(FeedbackSample(**obj))
                except Exception as e:
                    log.warning(f"bad feedback line skipped: {e}")
        log.info(f"loaded {len(self._buffer)} feedback samples from {self.path}")

    # ------------------------------------------------------------------
    # Adding
    # ------------------------------------------------------------------
    def add(
        self,
        ip: str,
        features: IPFeatures,
        true_label: int,
        source: str = "analyst",
        note: str = "",
    ) -> None:
        """Record a new labeled sample."""
        sample = FeedbackSample(
            ts=time.time(),
            ip=ip,
            features=list(features.to_vector()),
            true_label=int(true_label),
            source=source,
            note=note,
        )
        self._buffer.append(sample)

        with self.path.open("a") as f:
            f.write(json.dumps(asdict(sample)) + "\n")

        log.info(f"feedback added ip={ip} label={true_label} source={source}")

    def add_correction(self, ip: str, features: IPFeatures, true_label: int,
                       note: str = "") -> None:
        """Analyst said the model was wrong — record the correction."""
        self.add(ip, features, true_label, source="analyst", note=note)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buffer)

    def all(self) -> list[FeedbackSample]:
        return list(self._buffer)

    def since(self, ts: float) -> list[FeedbackSample]:
        return [s for s in self._buffer if s.ts > ts]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._buffer:
            return np.zeros((0, 0)), np.zeros(0, dtype=int)
        X = np.asarray([s.features for s in self._buffer], dtype=float)
        y = np.asarray([s.true_label for s in self._buffer], dtype=int)
        return X, y

    def clear(self) -> None:
        self._buffer = []
        if self.path.exists():
            self.path.unlink()
        log.info("feedback store cleared")


# ---------------------------------------------------------------------------
# Online learner
# ---------------------------------------------------------------------------

class OnlineLearner:
    """Combines base training data + live feedback to retrain models."""

    def __init__(
        self,
        store: FeedbackStore,
        base_raw_dir: str = "data/raw",
        base_labels: str = "data/labeled/labels.jsonl",
        replay_limit: int = REPLAY_LIMIT,
        min_new_for_retrain: int = MIN_NEW_FOR_RETRAIN,
    ) -> None:
        self.store = store
        self.base_raw_dir = base_raw_dir
        self.base_labels = base_labels
        self.replay_limit = replay_limit
        self.min_new_for_retrain = min_new_for_retrain

        self._last_retrain_ts: float = self._load_last_retrain_ts()
        log.info(f"OnlineLearner ready; last retrain ts={self._last_retrain_ts}")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _state_file(self) -> Path:
        return Path("ml/models/online_state.json")

    def _load_last_retrain_ts(self) -> float:
        p = self._state_file()
        if not p.exists():
            return 0.0
        with p.open() as f:
            state = json.load(f)
        return float(state.get("last_retrain_ts", 0.0))

    def _save_last_retrain_ts(self, ts: float) -> None:
        p = self._state_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump({"last_retrain_ts": ts}, f)

    # ------------------------------------------------------------------
    # Retrain decision
    # ------------------------------------------------------------------
    def new_samples_since_last_retrain(self) -> int:
        return len(self.store.since(self._last_retrain_ts))

    def should_retrain(self) -> tuple[bool, str]:
        n_new = self.new_samples_since_last_retrain()
        if n_new >= self.min_new_for_retrain:
            return True, f"{n_new} new feedback samples (>= {self.min_new_for_retrain})"
        return False, f"only {n_new} new samples since last retrain"

    # ------------------------------------------------------------------
    # Full retrain (base + feedback merged)
    # ------------------------------------------------------------------
    def build_combined_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Merge base training data with feedback samples."""
        log.info("loading base training data")
        try:
            X_base, y_base, _ = load_training_data(
                raw_dir=self.base_raw_dir,
                label_file=self.base_labels,
            )
        except Exception as e:
            log.warning(f"base data load failed: {e}")
            X_base = np.zeros((0, 0))
            y_base = np.zeros(0, dtype=int)

        X_fb, y_fb = self.store.as_arrays()

        if X_base.size == 0 and X_fb.size == 0:
            raise RuntimeError("no data available to train on")

        if X_fb.size == 0:
            log.info(f"using base only: {X_base.shape}")
            return X_base, y_base

        if X_base.size == 0:
            log.info(f"using feedback only: {X_fb.shape}")
            return X_fb, y_fb

        # Align columns (assume same feature order)
        X_comb = np.vstack([X_base, X_fb])
        y_comb = np.concatenate([y_base, y_fb])

        log.info(f"combined dataset: base={X_base.shape}, "
                 f"feedback={X_fb.shape} → total={X_comb.shape}")
        return X_comb, y_comb

    # ------------------------------------------------------------------
    # Retrain entry point
    # ------------------------------------------------------------------
    def retrain(
        self,
        model: str = "xgb",
        version: str | None = None,
    ) -> dict:
        """Retrain a model on base + feedback data, save it, update state."""
        X, y = self.build_combined_dataset()

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        log.info(f"retraining {model} on {len(y)} samples "
                 f"(pos={n_pos}, neg={n_neg})")

        if n_pos == 0 or n_neg == 0:
            raise RuntimeError(f"single-class labels (pos={n_pos}, neg={n_neg})")

        if model == "xgb":
            clf = train_xgboost(X, y)
        elif model == "rf":
            clf = train_random_forest(X, y)
        else:
            raise ValueError(f"unsupported model for online learning: {model}")

        # Version: timestamp-based if not provided
        if version is None:
            version = f"{model}_online_{int(time.time())}"

        metrics = {
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "feedback_samples": len(self.store),
            "timestamp": time.time(),
        }
        path = save_model(clf, version, metrics)

        self._last_retrain_ts = time.time()
        self._save_last_retrain_ts(self._last_retrain_ts)

        log.info(f"retrain complete: {path}")

        return {
            "model": model,
            "version": version,
            "path": str(path),
            "metrics": metrics,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Online learning management")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="show feedback store status")

    add_corr = sub.add_parser("add", help="add a manual correction")
    add_corr.add_argument("--ip", required=True)
    add_corr.add_argument("--label", type=int, choices=[0, 1], required=True)
    add_corr.add_argument("--note", default="")

    retrain = sub.add_parser("retrain", help="retrain from feedback")
    retrain.add_argument("--model", choices=["xgb", "rf"], default="xgb")

    sub.add_parser("clear", help="clear feedback store")

    args = p.parse_args()
    store = FeedbackStore()

    if args.cmd == "status":
        print(f"feedback samples: {len(store)}")
        n_new = 0
        for s in store.all():
            print(f"  {s.ip:15s} label={s.true_label} source={s.source} "
                  f"ts={s.ts:.0f} note={s.note}")

    elif args.cmd == "add":
        # This CLI form only records the label; features must be added programmatically
        print("use FeedbackStore.add_correction() from Python for full usage")

    elif args.cmd == "retrain":
        learner = OnlineLearner(store)
        ready, reason = learner.should_retrain()
        print(f"ready to retrain: {ready} ({reason})")
        if ready:
            result = learner.retrain(model=args.model)
            print(f"retrained → {result['path']}")
        else:
            print("skipping retrain — not enough new feedback")

    elif args.cmd == "clear":
        store.clear()
        print("feedback store cleared")