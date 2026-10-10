"""Online inference engine — loads a trained model and returns Decisions.

Usage:
    from ml.detector import Detector
    det = Detector(model_path="ml/models/v1_rf.pkl")
    decision = det.predict(features)   # features: IPFeatures
    if decision.action == "block":
        ...
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from collector.aggregator import IPFeatures
from core.logging import get_logger
from core.schema import ATTACK_THRESHOLD, DEFAULT_TTL, Decision

log = get_logger("ml.detector")


class Detector:
    """Wraps a trained sklearn/xgboost model for online inference."""

    def __init__(
        self,
        model_path: str | Path = "ml/models/v11_rf.pkl",
        threshold: float = ATTACK_THRESHOLD,
        ttl_seconds: int = DEFAULT_TTL,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        self.model_name = type(self.model).__name__
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds

        log.info(f"Detector loaded model={self.model_name} "
                 f"path={self.model_path} threshold={threshold}")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def predict(self, features: IPFeatures) -> Decision:
        """Return a Decision for a single IP's feature vector."""
        x = np.asarray([features.to_vector()], dtype=float)
        prob = self._attack_probability(x[0])

        action = "block" if prob >= self.threshold else "allow"
        reason = self._explain(features, prob)

        return Decision(
            ip=features.ip,
            action=action,
            confidence=float(prob),
            reason=reason,
            ttl_seconds=self.ttl_seconds,
        )

    def predict_many(self, features: dict[str, IPFeatures]) -> list[Decision]:
        """Return Decisions for multiple IPs."""
        return [self.predict(f) for f in features.values()]

    # ------------------------------------------------------------------
    # Model-specific probability conversion
    # ------------------------------------------------------------------
    def _attack_probability(self, x: np.ndarray) -> float:
        """Return P(attack) in [0, 1] regardless of model type."""
        if self.model_name == "IsolationForest":
            # decision_function: higher = more normal
            score = float(self.model.decision_function(x.reshape(1, -1))[0])
            # map score -> probability: score 0 = 0.5, positive = normal
            # use a logistic-like transform
            prob = 1.0 / (1.0 + np.exp(score * 5.0))
            return float(prob)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x.reshape(1, -1))[0]
            # assume class index 1 = attack
            if len(probs) >= 2:
                return float(probs[1])
            return float(probs[0])

        # fallback: use predict() output directly
        raw = float(self.model.predict(x.reshape(1, -1))[0])
        if self.model_name == "IsolationForest":
            return 0.9 if raw == -1 else 0.1
        return float(raw)

    # ------------------------------------------------------------------
    # Human-readable reason
    # ------------------------------------------------------------------
    def _explain(self, f: IPFeatures, prob: float) -> str:
        """Build a short explanation string."""
        parts = [f"model={self.model_name}", f"p={prob:.2f}"]
        if f.pkt_rate > 5.0:
            parts.append(f"high_rate={f.pkt_rate:.1f}/s")
        if f.error_ratio > 0.3:
            parts.append(f"high_errors={f.error_ratio:.2f}")
        if f.conn_duration > 30.0:
            parts.append(f"long_conn={f.conn_duration:.0f}s")
        if f.unique_dports > 20:
            parts.append(f"many_paths={f.unique_dports:.0f}")
        return " ".join(parts)