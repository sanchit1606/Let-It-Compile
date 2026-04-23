"""
BiLSTM-based temporal phase detector.

Input:  Sliding window of T=20 CUPTI counter vectors, shape (T, 5)
Output: Phase probability distribution over {compute, memory, latency, mixed}
        + uncertainty scalar

Phase labels (ground truth from roofline):
  0 = compute-bound  (high IPC, low DRAM BW, kernel on compute ceiling)
  1 = memory-bound   (low IPC, high DRAM BW, kernel on memory BW slope)
  2 = latency-bound  (low IPC, low DRAM BW, warp stalls dominate)
  3 = mixed          (transitions, unclear regime)

Training data:
  - Collect CUPTI traces for GEMM at various sizes
  - Label by arithmetic intensity vs roofline ridge point:
    RTX 3050 Ti ridge ≈ 7.8 TFLOP/s ÷ 192 GB/s ≈ 40.6 FLOP/byte
  - Small matrices (N<256): latency-bound
  - Medium matrices (256<N<1024): memory-bound
  - Large matrices (N>1024): compute-bound (for GEMM)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class PhaseDetector(nn.Module):
    """
    Bidirectional LSTM for GPU kernel phase detection.

    Architecture:
      Input:  (batch, T, 5)         ← T timesteps of 5 CUPTI counters
      BiLSTM: (batch, T, 2*hidden)  ← bidirectional
      MLP:    (batch, 4)            ← phase probabilities
              (batch, 1)            ← uncertainty [0,1]
    """

    WINDOW_SIZE = 20
    INPUT_DIM   = 5    # CUPTI counter dimensions
    NUM_PHASES  = 4    # compute, memory, latency, mixed

    PHASE_NAMES = {
        0: "compute-bound",
        1: "memory-bound",
        2: "latency-bound",
        3: "mixed",
    }

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.INPUT_DIM,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Phase classification head
        self.phase_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, self.NUM_PHASES),
        )

        # Uncertainty head (epistemic — higher = less confident)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, T, 5)
               T = window size (20 timesteps)
               5 = CUPTI counter dimensions

        Returns:
            phase_probs:  (batch, 4) — softmax probabilities
            uncertainty:  (batch, 1) — scalar uncertainty [0, 1]
        """
        # BiLSTM: output shape (batch, T, 2*hidden)
        lstm_out, _ = self.lstm(x)

        # Use last forward + last backward hidden states
        # Forward: lstm_out[:, -1, :hidden_dim]
        # Backward: lstm_out[:, 0, hidden_dim:]
        fwd = lstm_out[:, -1, :self.hidden_dim]
        bwd = lstm_out[:, 0,  self.hidden_dim:]
        combined = torch.cat([fwd, bwd], dim=-1)   # (batch, 2*hidden)

        phase_logits = self.phase_head(combined)
        phase_probs  = torch.softmax(phase_logits, dim=-1)
        uncertainty  = self.uncertainty_head(combined)

        return phase_probs, uncertainty

    def predict(self, counter_window: np.ndarray) -> Tuple[int, float, float]:
        """
        Single-window inference.

        Args:
            counter_window: np.ndarray of shape (T, 5) or (5,) for single step

        Returns:
            phase_label:  int in {0,1,2,3}
            confidence:   float in [0,1]
            uncertainty:  float in [0,1]
        """
        self.eval()
        with torch.no_grad():
            if counter_window.ndim == 1:
                # Single timestep — pad to window
                window = np.zeros((self.WINDOW_SIZE, self.INPUT_DIM), dtype=np.float32)
                window[-1] = counter_window
            else:
                window = counter_window[-self.WINDOW_SIZE:]  # Take last T steps
                # Pad if shorter than window
                if window.shape[0] < self.WINDOW_SIZE:
                    pad = np.zeros(
                        (self.WINDOW_SIZE - window.shape[0], self.INPUT_DIM),
                        dtype=np.float32,
                    )
                    window = np.concatenate([pad, window], axis=0)

            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, T, 5)
            probs, unc = self.forward(x)

            phase_label = int(probs.argmax(dim=-1).item())
            confidence  = float(probs.max().item())
            uncertainty = float(unc.item())

        return phase_label, confidence, uncertainty

    def phase_name(self, label: int) -> str:
        return self.PHASE_NAMES.get(label, "unknown")


# ─── Roofline-based labeling utilities ────────────────────────────────


def create_training_labels(
    cupti_traces: np.ndarray,
    arithmetic_intensities: np.ndarray,
    ridge_point: float = 40.6,
) -> np.ndarray:
    """
    Generate phase labels for training data based on roofline model.

    RTX 3050 Ti roofline ridge point:
      Peak compute: ~7.8 TFLOP/s (FP32)
      Peak memory BW: ~192 GB/s
      Ridge point = 7.8e12 / 192e9 ≈ 40.6 FLOP/byte

    Args:
        cupti_traces: (N, 5) array of normalized CUPTI counter measurements
                      Column order: [achieved_occ, l2_hit_rate, dram_bw_pct,
                                     warp_exec_eff, sm_active_pct]
        arithmetic_intensities: (N,) array of FLOP/byte for each measurement
        ridge_point: FLOP/byte at compute/memory boundary

    Returns:
        labels: (N,) int array with values in {0,1,2,3}
    """
    labels = np.zeros(len(arithmetic_intensities), dtype=int)

    for i, ai in enumerate(arithmetic_intensities):
        achieved_occ = cupti_traces[i, 0]
        dram_bw_pct  = cupti_traces[i, 2]

        if ai > ridge_point and achieved_occ > 0.5:
            labels[i] = 0  # compute-bound
        elif ai < ridge_point and dram_bw_pct > 0.5:
            labels[i] = 1  # memory-bound
        elif achieved_occ < 0.3 and dram_bw_pct < 0.3:
            labels[i] = 2  # latency-bound
        else:
            labels[i] = 3  # mixed

    return labels


def estimate_arithmetic_intensity(kernel_name: str, matrix_size: int) -> float:
    """
    Estimate arithmetic intensity (FLOP/byte) for a given kernel and problem size.

    This is used to generate roofline-based phase labels for training data
    when real CUPTI traces are unavailable.

    For GEMM (C = A × B, all NxN):
      FLOPs = 2 * N^3  (one multiply + one add per output element, N^2 outputs, N inner products)
      Bytes  = 3 * N^2 * 4  (read A, read B, write C — float32)
      AI = 2N^3 / (12 N^2) = N / 6

    For Reduction (sum of N elements):
      FLOPs = N - 1 ≈ N
      Bytes  = N * 4  (read input)
      AI = N / (4N) = 0.25  (always memory-bound)

    For Softmax (row-wise over NxN):
      FLOPs ≈ 5 * N^2  (exp, sum, div per element — approximate)
      Bytes  = 2 * N^2 * 4  (read input, write output)
      AI = 5N^2 / (8N^2) = 0.625  (always memory-bound)
    """
    if kernel_name == "gemm":
        return matrix_size / 6.0
    elif kernel_name == "reduction":
        return 0.25
    elif kernel_name == "softmax":
        return 0.625
    else:
        return 1.0  # Default: assume memory-bound


def label_from_kernel_and_size(kernel_name: str, matrix_size: int, ridge_point: float = 40.6) -> int:
    """
    Assign a phase label based on kernel type and problem size using the roofline model.

    Returns:
        int: Phase label in {0, 1, 2, 3}
    """
    ai = estimate_arithmetic_intensity(kernel_name, matrix_size)

    if kernel_name == "gemm":
        if matrix_size < 128:
            return 2  # latency-bound (too small to saturate GPU)
        elif ai > ridge_point:
            return 0  # compute-bound
        else:
            return 1  # memory-bound
    elif kernel_name == "reduction":
        if matrix_size < 128:
            return 2  # latency-bound
        return 1  # memory-bound (always low AI)
    elif kernel_name == "softmax":
        if matrix_size < 128:
            return 2  # latency-bound
        return 1  # memory-bound (always low AI)
    else:
        return 3  # mixed / unknown
