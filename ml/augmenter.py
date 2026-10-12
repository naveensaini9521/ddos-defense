"""Synthetic data augmentation for imbalanced DDoS datasets.

Implements a lightweight SMOTE-style oversampler that works with our
feature vectors. Interpolates between existing minority-class samples
to create new synthetic ones — this teaches the model to generalize
across the attack class rather than memorizing the few samples.

Also supports Gaussian jitter and boundary sampling.

Usage:
    from ml.augmenter import augment_training_data

    X_aug, y_aug = augment_training_data(X, y, target_ratio=1.0)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.logging import get_logger

log = get_logger("ml.augmenter")


@dataclass
class AugmentResult:
    X: np.ndarray
    y: np.ndarray
    n_synthetic: int
    method: str


# ---------------------------------------------------------------------------
# Core augmentation
# ---------------------------------------------------------------------------

def augment_training_data(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 1.0,
    method: str = "smote",
    jitter_std: float = 0.05,
    random_state: int = 42,
) -> AugmentResult:
    """Balance class distribution by synthesizing minority-class samples.

    Args:
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) labels (0=normal, 1=attack)
        target_ratio: desired attack/normal ratio. 1.0 = balanced.
        method: 'smote' or 'jitter'
        jitter_std: std-dev of Gaussian noise (as fraction of feature range)
        random_state: for reproducibility

    Returns:
        AugmentResult(X, y, n_synthetic, method)
    """
    rng = np.random.default_rng(random_state)

    y = np.asarray(y, dtype=int)
    X = np.asarray(X, dtype=float)

    n_normal = int((y == 0).sum())
    n_attack = int((y == 1).sum())

    if n_normal == 0 or n_attack == 0:
        log.warning(f"only one class present (normal={n_normal}, attack={n_attack}); "
                    f"no augmentation performed")
        return AugmentResult(X=X, y=y, n_synthetic=0, method="none")

    # Determine how many synthetic attacks we need
    target_attack = int(round(n_normal * target_ratio))
    n_synth_needed = max(target_attack - n_attack, 0)

    if n_synth_needed == 0:
        log.info(f"already balanced enough: normal={n_normal}, attack={n_attack}; "
                 f"target_ratio={target_ratio}")
        return AugmentResult(X=X, y=y, n_synthetic=0, method="none")

    log.info(f"augmenting: normal={n_normal}, attack={n_attack}, "
             f"need {n_synth_needed} synthetic attacks")

    attacks = X[y == 1]

    if method == "smote":
        synth = _smote_oversample(attacks, n_synth_needed, rng)
    elif method == "jitter":
        synth = _jitter_oversample(attacks, n_synth_needed, jitter_std, rng)
    else:
        raise ValueError(f"unknown method: {method}")

    X_aug = np.vstack([X, synth])
    y_aug = np.concatenate([y, np.ones(len(synth), dtype=int)])

    log.info(f"augmented: {len(X)} → {len(X_aug)} samples "
             f"({n_synth_needed} synthetic attacks)")

    return AugmentResult(
        X=X_aug, y=y_aug, n_synthetic=n_synth_needed, method=method
    )


# ---------------------------------------------------------------------------
# SMOTE-style interpolation
# ---------------------------------------------------------------------------

def _smote_oversample(
    attacks: np.ndarray,
    n_needed: int,
    rng: np.random.Generator,
    k_neighbors: int = 3,
) -> np.ndarray:
    """Interpolate between existing attack samples and their nearest neighbors.

    For each synthetic sample:
        1. Pick a random attack sample A
        2. Pick one of its k nearest neighbors B (also attack)
        3. Synthetic = A + rand(0, 1) * (B - A)
    """
    n_attacks = len(attacks)
    if n_attacks < 2:
        # fall back to jitter
        log.warning("only 1 attack sample — falling back to jitter")
        return _jitter_oversample(attacks, n_needed, 0.1, rng)

    # pairwise distances (simple — dataset is small)
    synthetic = np.zeros((n_needed, attacks.shape[1]), dtype=float)

    for i in range(n_needed):
        # pick sample A
        a_idx = rng.integers(0, n_attacks)
        A = attacks[a_idx]

        # find k nearest neighbors of A
        dists = np.linalg.norm(attacks - A, axis=1)
        dists[a_idx] = np.inf
        k = min(k_neighbors, n_attacks - 1)
        neighbors = np.argsort(dists)[:k]

        # pick B
        b_idx = int(rng.choice(neighbors))
        B = attacks[b_idx]

        # interpolate
        alpha = rng.random()
        synthetic[i] = A + alpha * (B - A)

    return synthetic


# ---------------------------------------------------------------------------
# Jitter-based (Gaussian noise) augmentation
# ---------------------------------------------------------------------------

def _jitter_oversample(
    attacks: np.ndarray,
    n_needed: int,
    jitter_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add small Gaussian noise to random attack samples."""
    n_attacks = len(attacks)
    synthetic = np.zeros((n_needed, attacks.shape[1]), dtype=float)

    # scale jitter by feature range so we don't distort absolute scales
    feature_range = np.ptp(attacks, axis=0)
    feature_range[feature_range == 0] = 1.0
    sigma = jitter_std * feature_range

    for i in range(n_needed):
        a_idx = rng.integers(0, n_attacks)
        base = attacks[a_idx]
        noise = rng.normal(0, sigma, size=base.shape)
        synthetic[i] = base + noise

    return synthetic


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def class_balance(y: np.ndarray) -> dict:
    """Return counts and ratio per class."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return {
        "n_normal": n_neg,
        "n_attack": n_pos,
        "ratio": (n_pos / n_neg) if n_neg else float("inf"),
        "total": len(y),
    }


# ---------------------------------------------------------------------------
# CLI (quick test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from ml.data_loader import load_training_data

    print("loading training data...")
    X, y, ips = load_training_data()
    print(f"before: {class_balance(y)}")

    result = augment_training_data(X, y, target_ratio=1.0, method="smote")
    print(f"after:  {class_balance(result.y)}")
    print(f"synthetic samples added: {result.n_synthetic}")
    print(f"new shape: {result.X.shape}")