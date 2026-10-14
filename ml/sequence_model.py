"""LSTM/GRU sequence model for temporal attack detection.

Tree-based models (XGBoost, RF) treat each feature vector independently.
This model sees a SEQUENCE of feature vectors over time, so it can detect:

    - Ramping attacks (gradually increasing rate)
    - Regular intervals (botnet timing)
    - Burst patterns
    - Slowloris (gradual connection buildup)

Architecture:
    Input: [batch, seq_len, n_features]
    LSTM/GRU: hidden size 64
    Dense: → 1 (probability of attack)

Usage:
    python3 -m ml.sequence_model --train --model lstm
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from collector.aggregator import IPFeatures
from core.logging import get_logger
from core.schema import FEATURE_NAMES

log = get_logger("ml.sequence_model")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not installed — sequence_model will not work")

MODEL_DIR = Path("ml/models")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SeqConfig:
    model: str = "lstm"          # 'lstm' or 'gru'
    seq_len: int = 8             # how many timesteps to look back
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 0.001
    epochs: int = 30
    batch_size: int = 16
    n_features: int = 9


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class AttackSeqModel(nn.Module):
        def __init__(self, cfg: SeqConfig):
            super().__init__()
            self.cfg = cfg

            rnn_cls = nn.LSTM if cfg.model == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=cfg.n_features,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            )

            self.head = nn.Sequential(
                nn.Linear(cfg.hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            # x: [batch, seq_len, n_features]
            out, _ = self.rnn(x)                 # [batch, seq_len, hidden]
            last = out[:, -1, :]                 # last timestep
            logit = self.head(last).squeeze(-1)  # [batch]
            return logit


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 8,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences from a 2D feature matrix.

    Each sequence has `seq_len` consecutive rows. The label of a sequence
    is the label of the LAST row (the one we're predicting).
    """
    n = len(X)
    if n < seq_len:
        raise ValueError(f"need at least {seq_len} samples, have {n}")

    seqs, labels = [], []
    for i in range(0, n - seq_len + 1, stride):
        seqs.append(X[i : i + seq_len])
        labels.append(y[i + seq_len - 1])

    return np.asarray(seqs, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def sequences_from_ip_history(
    history: dict[str, list[IPFeatures]],
    seq_len: int = 8,
    label_map: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sequences from per-IP temporal history.

    history: {ip: [IPFeatures over time]}
    label_map: {ip: 0/1}, defaults to all 0
    """
    seqs, labels = [], []
    label_map = label_map or {}

    for ip, features_list in history.items():
        if len(features_list) < seq_len:
            continue
        arr = np.asarray([f.to_vector() for f in features_list], dtype=np.float32)
        label = int(label_map.get(ip, 0))

        for i in range(len(arr) - seq_len + 1):
            seqs.append(arr[i : i + seq_len])
            labels.append(label)

    if not seqs:
        return np.zeros((0, seq_len, len(FEATURE_NAMES)), dtype=np.float32), \
               np.zeros(0, dtype=np.float32)

    return np.asarray(seqs, dtype=np.float32), np.asarray(labels, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sequence_model(
    X: np.ndarray,
    y: np.ndarray,
    cfg: SeqConfig | None = None,
    version: str = "seq_v1",
) -> dict:
    """Train an LSTM/GRU on sequence data."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")

    cfg = cfg or SeqConfig()
    log.info(f"training {cfg.model} on X={X.shape}, y={y.shape}")

    # Adjust cfg to actual data shape
    cfg.n_features = X.shape[2]

    # Train/val split (simple)
    n = len(X)
    n_val = max(1, int(n * 0.2))
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)

    tr_idx, val_idx = idx[:-n_val], idx[-n_val:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")

    model = AttackSeqModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Class weighting via pos_weight
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        log.info(f"pos_weight={float(pos_weight[0]):.2f}")

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience = 5
    patience_counter = 0

    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(len(tr_loader), 1),
            "val_loss": val_loss,
        })

        if epoch % 5 == 0 or epoch == 1:
            log.info(f"epoch {epoch:3d} train={history[-1]['train_loss']:.4f} "
                     f"val={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{version}.pt"
    torch.save({
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
        "history": history,
    }, path)
    log.info(f"sequence model saved to {path}")

    return {
        "version": version,
        "path": str(path),
        "config": asdict(cfg),
        "best_val_loss": best_val_loss,
        "epochs_run": len(history),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class SequenceDetector:
    """Loads a saved sequence model and produces per-IP attack probabilities."""

    def __init__(self, model_path: str | Path):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")

        self.path = Path(model_path)
        if not self.path.exists():
            raise FileNotFoundError(f"model not found: {self.path}")

        checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)
        self.cfg = SeqConfig(**checkpoint["config"])

        self.model = AttackSeqModel(self.cfg)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.device = torch.device("cpu")
        self.model.to(self.device)

        log.info(f"SequenceDetector loaded: {self.cfg.model} "
                 f"seq_len={self.cfg.seq_len}")

    def predict_sequence(self, seq: np.ndarray) -> float:
        """Predict P(attack) for one sequence of shape [seq_len, n_features]."""
        x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(x)
            prob = torch.sigmoid(logit).item()
        return float(prob)

    def predict_history(self, history: list[IPFeatures]) -> float:
        """Predict P(attack) from a list of IPFeatures (temporal order)."""
        if len(history) < self.cfg.seq_len:
            # pad by repeating the first sample
            pad = [history[0]] * (self.cfg.seq_len - len(history))
            history = pad + list(history)

        arr = np.asarray([f.to_vector() for f in history[-self.cfg.seq_len:]],
                         dtype=np.float32)
        return self.predict_sequence(arr)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict P(attack) for a batch of sequences [batch, seq_len, n_feat]."""
        x = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequence model for DDoS")
    p.add_argument("--train", action="store_true")
    p.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--version", default="seq_v1")
    p.add_argument("--data", default="data/raw")
    p.add_argument("--labels", default="data/labeled/labels.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not TORCH_AVAILABLE:
        raise SystemExit("PyTorch not installed. "
                         "Run: pip install torch --index-url "
                         "https://download.pytorch.org/whl/cpu")

    if not args.train:
        print("Use --train to build a sequence model.")
        return 0

    # Load base features + labels
    from ml.data_loader import load_training_data
    X, y, _ = load_training_data(raw_dir=args.data, label_file=args.labels)
    log.info(f"base data: X={X.shape}, y={y.shape}, pos={int(y.sum())}")

    # Small dataset: duplicate to create sequences
    # In production you'd have real temporal data
    if len(X) < args.seq_len:
        log.warning(f"only {len(X)} samples; padding to {args.seq_len} by repetition")
        reps = (args.seq_len // len(X)) + 1
        X = np.tile(X, (reps, 1))[:args.seq_len]
        y = np.tile(y, reps)[:args.seq_len]

    X_seq, y_seq = build_sequences(X, y, seq_len=args.seq_len, stride=1)
    log.info(f"built {len(X_seq)} sequences of length {args.seq_len}")

    cfg = SeqConfig(
        model=args.model,
        seq_len=args.seq_len,
        hidden_size=args.hidden,
        epochs=args.epochs,
        n_features=X_seq.shape[2],
    )

    result = train_sequence_model(X_seq, y_seq, cfg, version=args.version)

    print()
    print(f"=== Sequence Model Trained ===")
    print(f"version: {result['version']}")
    print(f"path:    {result['path']}")
    print(f"epochs:  {result['epochs_run']}")
    print(f"val loss: {result['best_val_loss']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())