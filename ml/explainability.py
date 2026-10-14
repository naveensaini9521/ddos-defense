"""Explainability for DDoS detection models.

Provides:
    - SHAP values (per-feature contribution to each prediction)
    - Feature importance (global + local)
    - Human-readable explanations ("why was this IP flagged?")

Works with XGBoost and RandomForest. Falls back to feature_importances_
if SHAP is unavailable.

Usage:
    from ml.explainability import explain_prediction

    explanation = explain_prediction(model, features, top_k=3)
    print(explanation["summary"])
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from collector.aggregator import IPFeatures
from core.logging import get_logger
from core.schema import FEATURE_NAMES

log = get_logger("ml.explainability")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureContribution:
    name: str
    value: float
    shap_value: float       # signed — positive pushes toward ATTACK
    direction: str          # "attack" or "normal"


@dataclass
class Explanation:
    ip: str
    predicted_attack_prob: float
    feature_values: list[float]
    contributions: list[FeatureContribution]
    top_k: int = 3

    @property
    def summary(self) -> str:
        """Human-readable explanation string."""
        if not self.contributions:
            return f"IP {self.ip}: p={self.predicted_attack_prob:.2f} (no details)"

        top = self.contributions[: self.top_k]
        parts = [f"IP={self.ip}", f"p={self.predicted_attack_prob:.2f}"]

        for c in top:
            sign = "+" if c.shap_value > 0 else "-"
            parts.append(f"{sign}{abs(c.shap_value):.3f}×{c.name}({c.value:.1f})")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

_shap_explainers: dict[int, object] = {}


def _get_shap_explainer(model):
    """Cache SHAP explainer per model (keyed by id)."""
    key = id(model)
    if key in _shap_explainers:
        return _shap_explainers[key]

    try:
        import shap
    except ImportError:
        log.warning("shap not installed; falling back to feature_importances_")
        return None

    try:
        model_name = type(model).__name__

        if model_name in ("XGBClassifier", "XGBRegressor"):
            explainer = shap.TreeExplainer(model)
        elif model_name == "RandomForestClassifier":
            explainer = shap.TreeExplainer(model)
        else:
            # use generic explainer with model-agnostic kernel
            # (slower, but works on anything with predict_proba)
            explainer = None

        if explainer is not None:
            _shap_explainers[key] = explainer
            log.info(f"SHAP TreeExplainer ready for {model_name}")
        return explainer

    except Exception as e:
        log.warning(f"SHAP explainer failed: {e}; using fallback")
        return None


# ---------------------------------------------------------------------------
# Fallback: feature_importances_ × feature_value
# ---------------------------------------------------------------------------

def _fallback_contributions(model, x: np.ndarray) -> np.ndarray:
    """Approximate per-feature contribution via importances × value."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return np.zeros_like(x)
    # Direction: positive feature value × importance
    return importances * x


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def explain_prediction(
    model,
    features: IPFeatures,
    top_k: int = 5,
) -> Explanation:
    """Explain a single prediction with SHAP (or fallback)."""
    x = np.asarray(features.to_vector(), dtype=float).reshape(1, -1)

    # Get P(attack) from the model
    try:
        prob = float(model.predict_proba(x)[0, 1])
    except (AttributeError, IndexError):
        prob = float(model.predict(x)[0])

    # SHAP values
    shap_vals = None
    explainer = _get_shap_explainer(model)

    if explainer is not None:
        try:
            sv = explainer.shap_values(x)
            # Different SHAP versions return different shapes
            if isinstance(sv, list):
                shap_vals = np.asarray(sv[1]).reshape(-1)   # class 1
            else:
                shap_vals = np.asarray(sv).reshape(-1)
        except Exception as e:
            log.warning(f"SHAP failed: {e}")
            shap_vals = None

    if shap_vals is None or len(shap_vals) != len(FEATURE_NAMES):
        shap_vals = _fallback_contributions(model, x.reshape(-1))

    # Build contributions
    contributions: list[FeatureContribution] = []
    for name, val, sv in zip(FEATURE_NAMES, features.to_vector(), shap_vals):
        contributions.append(FeatureContribution(
            name=name,
            value=float(val),
            shap_value=float(sv),
            direction="attack" if sv > 0 else "normal",
        ))

    # Sort by absolute SHAP value (biggest impact first)
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

    return Explanation(
        ip=features.ip,
        predicted_attack_prob=prob,
        feature_values=list(features.to_vector()),
        contributions=contributions[:top_k],
        top_k=top_k,
    )


def explain_batch(
    model,
    features_dict: dict[str, IPFeatures],
    top_k: int = 3,
) -> list[Explanation]:
    """Explain multiple IPs."""
    return [explain_prediction(model, f, top_k=top_k)
            for f in features_dict.values()]


# ---------------------------------------------------------------------------
# Global feature importance
# ---------------------------------------------------------------------------

def global_importance(model) -> dict[str, float]:
    """Return global feature importance (from model, not SHAP)."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {}
    return {name: float(imp) for name, imp in zip(FEATURE_NAMES, importances)}


def print_global_importance(model) -> None:
    imp = global_importance(model)
    if not imp:
        print("model does not expose feature_importances_")
        return

    print()
    print("=== Global Feature Importance ===")
    for name, val in sorted(imp.items(), key=lambda x: -x[1]):
        bar = "█" * int(val * 50)
        print(f"{name:18s} {val:.4f}  {bar}")


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import joblib

    p = argparse.ArgumentParser(description="Explain model predictions")
    p.add_argument("--model", default="ml/models/v1_xgb.pkl")
    p.add_argument("--data", default="data/raw")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    from collector.aggregator import aggregate
    from collector.nginx_parser import parse_file
    from pathlib import Path

    model = joblib.load(args.model)
    print(f"loaded model: {type(model).__name__}")

    # Load features from all .log files
    all_feats: dict = {}
    for lf in sorted(Path(args.data).glob("*.log")):
        recs = parse_file(lf)
        all_feats.update(aggregate(recs))

    print(f"explaining {len(all_feats)} IPs")
    print()

    explanations = explain_batch(model, all_feats, top_k=args.top_k)

    for ex in explanations:
        verdict = "BLOCK" if ex.predicted_attack_prob >= 0.5 else "allow"
        print(f"[{verdict}] {ex.summary}")
        for c in ex.contributions:
            sign = "+" if c.shap_value > 0 else "-"
            print(f"      {sign} {c.name:18s} value={c.value:>8.2f}  "
                  f"shap={c.shap_value:+.4f}  → {c.direction}")

    print()
    print_global_importance(model)