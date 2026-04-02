from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardResult:
    reward: float
    speedup: float


def speedup_reward(*, baseline_ms: float, measured_ms: float) -> RewardResult:
    """Compute reward as speedup relative to baseline.

    speedup = baseline / measured
    reward  = speedup - 1

    This keeps the baseline action near 0 reward.
    """

    eps = 1e-9
    denom = max(measured_ms, eps)
    speedup = float(baseline_ms / denom)
    reward = float(speedup - 1.0)
    return RewardResult(reward=reward, speedup=speedup)
