#!/usr/bin/env python3
"""
Phase 5: Train the BiLSTM phase detector.

Generates synthetic training data from Phase 0/1 CSV results (if available)
or from heuristic distributions, then trains the BiLSTM to classify
kernel execution phases.

Usage:
    python training/train_phase_detector.py
    python training/train_phase_detector.py --epochs 100 --lr 1e-3

Outputs:
    results/models/phase_detector.pt       — trained model weights
    results/tables/phase5_eval.csv         — per-class evaluation metrics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.phase_detector import (
    PhaseDetector,
    label_from_kernel_and_size,
)

_RESULTS = _REPO_ROOT / "results"
_MODELS  = _RESULTS / "models"
_TABLES  = _RESULTS / "tables"


# ── Synthetic data generation ────────────────────────────────────────


def _synthesize_cupti_window(
    phase_label: int,
    window_size: int = 20,
    input_dim: int = 5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a synthetic CUPTI counter window for a given phase.

    Column order: [achieved_occ, l2_hit_rate, dram_bw_pct, warp_exec_eff, sm_active_pct]

    The distributions are empirically plausible for an RTX 3050 Ti running
    the project's benchmark kernels.  They are NOT real profiler data — the
    purpose is to give the BiLSTM something to learn from so it can later
    be fine-tuned on real traces.
    """
    if rng is None:
        rng = np.random.default_rng()

    window = np.zeros((window_size, input_dim), dtype=np.float32)

    if phase_label == 0:
        # Compute-bound: high occupancy, low DRAM BW, high SM active
        window[:, 0] = rng.uniform(0.55, 0.95, window_size)  # achieved_occ
        window[:, 1] = rng.uniform(0.3,  0.7,  window_size)  # l2_hit_rate
        window[:, 2] = rng.uniform(0.05, 0.35, window_size)  # dram_bw_pct
        window[:, 3] = rng.uniform(0.8,  1.0,  window_size)  # warp_exec_eff
        window[:, 4] = rng.uniform(0.7,  0.95, window_size)  # sm_active_pct

    elif phase_label == 1:
        # Memory-bound: moderate occupancy, high DRAM BW
        window[:, 0] = rng.uniform(0.3,  0.7,  window_size)  # achieved_occ
        window[:, 1] = rng.uniform(0.1,  0.5,  window_size)  # l2_hit_rate
        window[:, 2] = rng.uniform(0.55, 0.95, window_size)  # dram_bw_pct
        window[:, 3] = rng.uniform(0.6,  0.9,  window_size)  # warp_exec_eff
        window[:, 4] = rng.uniform(0.4,  0.7,  window_size)  # sm_active_pct

    elif phase_label == 2:
        # Latency-bound: low occupancy, low DRAM BW, low SM active
        window[:, 0] = rng.uniform(0.05, 0.3,  window_size)  # achieved_occ
        window[:, 1] = rng.uniform(0.2,  0.8,  window_size)  # l2_hit_rate
        window[:, 2] = rng.uniform(0.01, 0.25, window_size)  # dram_bw_pct
        window[:, 3] = rng.uniform(0.3,  0.7,  window_size)  # warp_exec_eff
        window[:, 4] = rng.uniform(0.05, 0.35, window_size)  # sm_active_pct

    else:
        # Mixed: overlapping ranges, harder to classify
        window[:, 0] = rng.uniform(0.2,  0.6,  window_size)  # achieved_occ
        window[:, 1] = rng.uniform(0.2,  0.6,  window_size)  # l2_hit_rate
        window[:, 2] = rng.uniform(0.2,  0.6,  window_size)  # dram_bw_pct
        window[:, 3] = rng.uniform(0.5,  0.8,  window_size)  # warp_exec_eff
        window[:, 4] = rng.uniform(0.3,  0.6,  window_size)  # sm_active_pct

    # Add temporal noise (small walk) to simulate counter jitter
    noise = rng.normal(0, 0.02, window.shape).astype(np.float32)
    window = np.clip(window + np.cumsum(noise, axis=0), 0.0, 1.0)

    return window


def generate_synthetic_dataset(
    n_samples: int = 2000,
    window_size: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced synthetic dataset.

    Returns:
        X: (n_samples, window_size, 5)
        y: (n_samples,) int labels in {0,1,2,3}
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 4

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for phase in range(4):
        for _ in range(n_per_class):
            w = _synthesize_cupti_window(phase, window_size=window_size, rng=rng)
            X_list.append(w)
            y_list.append(phase)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def try_load_phase0_labels() -> list[dict] | None:
    """
    Attempt to load Phase 0 CSV and create augmented training examples
    using roofline-based labeling.

    Returns None if Phase 0 CSV is unavailable.
    """
    csv_path = _TABLES / "phase0_baseline.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        rows = []
        for _, row in df.iterrows():
            kernel = str(row.get("kernel", "gemm"))
            size = int(row.get("matrix_size", 256))
            label = label_from_kernel_and_size(kernel, size)
            rows.append({"kernel": kernel, "matrix_size": size, "label": label})
        return rows
    except Exception:
        return None


# ── Training loop ────────────────────────────────────────────────────


def train_phase_detector(
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    n_train: int = 1600,
    n_val: int = 400,
    seed: int = 42,
) -> PhaseDetector:
    """Train the BiLSTM phase detector and return the model."""

    print("=" * 60)
    print("Phase 5: BiLSTM Phase Detector Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate data
    print(f"Generating synthetic training data ({n_train + n_val} samples)...")
    X_all, y_all = generate_synthetic_dataset(n_samples=n_train + n_val, seed=seed)

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val,   y_val   = X_all[n_train:], y_all[n_train:]

    # Check class balance
    for phase in range(4):
        n_tr = (y_train == phase).sum()
        n_va = (y_val == phase).sum()
        print(f"  Phase {phase} ({PhaseDetector.PHASE_NAMES[phase]:>15s}): train={n_tr}, val={n_va}")

    # Augment with Phase 0 data if available
    phase0_rows = try_load_phase0_labels()
    if phase0_rows is not None:
        print(f"  Augmenting with {len(phase0_rows)} Phase 0 roofline labels")
        rng = np.random.default_rng(seed + 1)
        extra_X, extra_y = [], []
        for row in phase0_rows:
            w = _synthesize_cupti_window(row["label"], rng=rng)
            extra_X.append(w)
            extra_y.append(row["label"])
        extra_X = np.stack(extra_X, axis=0)
        extra_y = np.array(extra_y, dtype=np.int64)
        X_train = np.concatenate([X_train, extra_X], axis=0)
        y_train = np.concatenate([y_train, extra_y], axis=0)
        print(f"  Total training samples: {len(X_train)}")

    # Datasets
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Model
    model = PhaseDetector(hidden_dim=64, num_layers=2, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    print(f"\nTraining for {epochs} epochs...")
    print(f"{'Epoch':>6s}  {'Train Loss':>10s}  {'Val Loss':>10s}  {'Val Acc':>8s}")
    print("-" * 40)

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # Single LSTM forward pass, then extract logits for cross-entropy
            lstm_out, _ = model.lstm(xb)
            fwd = lstm_out[:, -1, :model.hidden_dim]
            bwd = lstm_out[:, 0,  model.hidden_dim:]
            combined = torch.cat([fwd, bwd], dim=-1)
            logits = model.phase_head(combined)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                lstm_out, _ = model.lstm(xb)
                fwd = lstm_out[:, -1, :model.hidden_dim]
                bwd = lstm_out[:, 0,  model.hidden_dim:]
                combined = torch.cat([fwd, bwd], dim=-1)
                logits = model.phase_head(combined)

                loss = criterion(logits, yb)
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_loss /= max(len(val_loader), 1)
        val_acc = correct / max(total, 1)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"{epoch:>6d}  {train_loss:>10.4f}  {val_loss:>10.4f}  {val_acc:>7.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest validation accuracy: {best_val_acc:.1%}")

    # ── Save model ──
    _MODELS.mkdir(parents=True, exist_ok=True)
    model_path = _MODELS / "phase_detector.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # ── Per-class evaluation ──
    model.eval()
    per_class = {p: {"tp": 0, "fp": 0, "fn": 0} for p in range(4)}

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            lstm_out, _ = model.lstm(xb)
            fwd = lstm_out[:, -1, :model.hidden_dim]
            bwd = lstm_out[:, 0,  model.hidden_dim:]
            combined = torch.cat([fwd, bwd], dim=-1)
            preds = model.phase_head(combined).argmax(dim=-1)

            for p, y in zip(preds.cpu().numpy(), yb.cpu().numpy()):
                if p == y:
                    per_class[y]["tp"] += 1
                else:
                    per_class[p]["fp"] += 1
                    per_class[y]["fn"] += 1

    eval_rows = []
    for phase in range(4):
        tp = per_class[phase]["tp"]
        fp = per_class[phase]["fp"]
        fn = per_class[phase]["fn"]
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        eval_rows.append({
            "phase": phase,
            "phase_name": PhaseDetector.PHASE_NAMES[phase],
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   tp + fn,
        })

    _TABLES.mkdir(parents=True, exist_ok=True)
    eval_df = pd.DataFrame(eval_rows)
    eval_path = _TABLES / "phase5_eval.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"\nPer-class evaluation saved to: {eval_path}")
    print(eval_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Phase 5 training complete!")
    print("=" * 60)

    return model


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Phase 5: Train BiLSTM phase detector")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    p.add_argument("--n-train", type=int, default=1600, help="Training samples (default: 1600)")
    p.add_argument("--n-val", type=int, default=400, help="Validation samples (default: 400)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = p.parse_args()

    train_phase_detector(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
