from __future__ import annotations

from dataclasses import dataclass
from typing import List

import gymnasium as gym
from gymnasium import spaces


# Canonical action knobs explored in Phase 0/1.
BLOCK_SIZES: List[int] = [64, 128, 256]

# 0 means "PTXAS default" (no requested cap).
REG_CAPS: List[int] = [0, 32, 64]

KERNEL_NAMES: List[str] = ["gemm", "reduction", "softmax"]


@dataclass(frozen=True)
class Action:
    block_size: int
    reg_cap: int


def make_action_space() -> spaces.MultiDiscrete:
    """Return a discrete action space for (block_size, reg_cap)."""

    return spaces.MultiDiscrete([len(BLOCK_SIZES), len(REG_CAPS)])


def decode_action(action: gym.core.ActType) -> Action:
    """Decode action indices into concrete knob values."""

    block_idx = int(action[0])
    reg_idx = int(action[1])

    if block_idx < 0 or block_idx >= len(BLOCK_SIZES):
        raise ValueError(f"Invalid block index {block_idx}")
    if reg_idx < 0 or reg_idx >= len(REG_CAPS):
        raise ValueError(f"Invalid reg_cap index {reg_idx}")

    return Action(block_size=BLOCK_SIZES[block_idx], reg_cap=REG_CAPS[reg_idx])
