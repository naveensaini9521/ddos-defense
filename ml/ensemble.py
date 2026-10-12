"""Three-model ensemble detector: IsolationForest + RandomForest + XGBoost.

Each model votes on P(attack). Votes are combined via weighted average.
Disagreement between models triggers "throttle" instead of "block",
which is a cautious middle ground when the models don't agree.

Usage:
    from ml.ensemble import EnsembleDetector

    det = EnsembleDetector()
    decision = det.predict(features)   # features: IPFeatures
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from collector.aggregator import IPFeatures
from core.logging import get_logger
from core.schema import ATTACK_THRESHOLD, DEFAULT_TTL, Decision
from ml.detector import Detector

log = get_logger("ml.ensemble")


# Weight per model. Interpretation:
#   - XGBoost is the most accurate on labeled data       → highest weight
#   - RandomForest is a solid, robust second opinion      → medium weight
#   - IsolationForest is the "unknown attack" safety net  → lower weight
DEFAULT_WEIGHTS = {
    "iso": 0.20,
    "rf":  0.35,
    "xgb": 0.45,
}


class EnsembleDetector:
    """Weighted-vote ensemble of IsolationForest, RandomForest, XGBoost."""

    def __init__(
        self,
        model_paths: dict[str, str] | None = None,
        weights: dict[str, float] | None = None,
        threshold: float = ATTACK_THRESHOLD,
        ttl_seconds: int = DEFAULT_TTL,
        disagreement_band: float = 0.15,
    ) -> None:
        """
        Args:
            model_paths: mapping name → .pkl path (defaults to our 3 models).
            weights: mapping name → weight (sums normalized to 1.0).
            threshold: P(attack) above which ensemble returns "block".
            ttl_seconds: default block TTL.
            disagreement_band: std-dev of per-model probabilities above which
                the ensemble demotes action to "throttle".
        """
        self.model_paths = model_paths or {
            "v1_iso": "ml/models/v11_iso.pkl",
            "v1_rf":  "ml/models/v11_rf.pkl",
            "v1_xgb": "ml/models/v11_xgb.pkl",
        }
        self.weights = weights or DEFAULT_WEIGHTS
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.disagreement_band = disagreement_band

        # Normalize weights over the models we'll actually load
        total = sum(self.weights.get(k, 0.0) for k in self.model_paths)
        if total <= 0:
            raise ValueError("ensemble weights must sum to a positive value")
        self._norm_weights = {
            k: self.weights.get(k, 0.0) / total for k in self.model_paths
        }

        # Load each model into a Detector wrapper
        self.detectors: dict[str, Detector] = {}
        for name, path in self.model_paths.items():
            p = Path(path)
            if not p.exists():
                log.warning(f"ensemble: skipping {name} (missing {path})")
                continue
            try:
                self.detectors[name] = Detector(
                    model_path=path,
                    threshold=0.5,          # per-model threshold; ensemble decides final
                    ttl_seconds=ttl_seconds,
                )
                log.info(f"ensemble: loaded {name} → {self.detectors[name].model_name}")
            except Exception as e:
                log.warning(f"ensemble: failed to load {name}: {e}")

        if not self.detectors:
            raise RuntimeError("ensemble: no models could be loaded")

        # Re-normalize weights to only loaded detectors
        active_total = sum(self._norm_weights[k] for k in self.detectors)
        self._norm_weights = {
            k: self._norm_weights[k] / active_total for k in self.detectors
        }

        active = {k: round(v, 2) for k, v in self._norm_weights.items()}
        log.info(f"ensemble ready: {len(self.detectors)} model(s), weights={active}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, features: IPFeatures) -> Decision:
        """Combine all model votes into one Decision."""
        per_model: dict[str, float] = {}
        for name, det in self.detectors.items():
            per_model[name] = det.predict(features).confidence

        # Weighted average
        weighted_sum = 0.0
        for name, p in per_model.items():
            weighted_sum += self._norm_weights[name] * p
        p_attack = weighted_sum

        # Disagreement = std-dev of model probabilities
        p_values = np.array(list(per_model.values()))
        disagreement = float(p_values.std()) if len(p_values) > 1 else 0.0

        # Decide action
        if p_attack >= self.threshold:
            action = "block"
        elif disagreement > self.disagreement_band and p_attack >= 0.4:
            action = "throttle"
        else:
            action = "allow"

        log.info(
            f"ML DECISION ip={features.ip} "
            f"iso={per_model.get('v1_iso', 0):.3f} "
            f"rf={per_model.get('v1_rf', 0):.3f} "
            f"xgb={per_model.get('v1_xgb', 0):.3f} "
            f"p_attack={p_attack:.3f} "
            f"disagreement={disagreement:.3f} "
            f"action={action}"
        )
        
        reason = self._format_reason(per_model, p_attack, disagreement)

        return Decision(
            ip=features.ip,
            action=action,
            confidence=float(p_attack),
            reason=reason,
            ttl_seconds=self.ttl_seconds,
        )

    def predict_many(self, features: dict[str, IPFeatures]) -> list[Decision]:
        return [self.predict(f) for f in features.values()]

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def _format_reason(
        self,
        per_model: dict[str, float],
        p_attack: float,
        disagreement: float,
    ) -> str:
        parts = [f"p={p_attack:.2f}", f"disagree={disagreement:.2f}"]
        for name, p in per_model.items():
            parts.append(f"{name}={p:.2f}")
        return " ".join(parts)