from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces

from environment.action_space import KERNEL_NAMES


DEFAULT_CUPTI_KEYS: List[str] = [
    "achieved_occupancy",
    "l2_hit_rate",
    "dram_bw_pct",
    "sm_active_pct",
]


@dataclass(frozen=True)
class ObservationSpec:
    cupti_keys: Tuple[str, ...] = tuple(DEFAULT_CUPTI_KEYS)
    include_nvml: bool = True
    include_prev_action: bool = True


def kernel_onehot(kernel_name: str) -> np.ndarray:
    v = np.zeros(len(KERNEL_NAMES), dtype=np.float32)
    try:
        idx = KERNEL_NAMES.index(kernel_name)
    except ValueError:
        return v
    v[idx] = 1.0
    return v


def make_observation_space(spec: ObservationSpec) -> spaces.Box:
    dim = len(spec.cupti_keys) + (4 if spec.include_nvml else 0) + len(KERNEL_NAMES)
    if spec.include_prev_action:
        dim += 2

    # Everything is normalized into [0, 1] (best-effort). Use a bounded Box.
    low = np.zeros(dim, dtype=np.float32)
    high = np.ones(dim, dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def build_observation(
    *,
    kernel_name: str,
    cupti_norm: Sequence[float],
    nvml_norm: Optional[Sequence[float]],
    prev_action_norm: Optional[Sequence[float]],
    spec: ObservationSpec,
) -> np.ndarray:
    parts: List[np.ndarray] = []

    parts.append(np.asarray(cupti_norm, dtype=np.float32))

    if spec.include_nvml:
        if nvml_norm is None:
            parts.append(np.zeros(4, dtype=np.float32))
        else:
            parts.append(np.asarray(nvml_norm, dtype=np.float32))

    parts.append(kernel_onehot(kernel_name))

    if spec.include_prev_action:
        if prev_action_norm is None:
            parts.append(np.zeros(2, dtype=np.float32))
        else:
            parts.append(np.asarray(prev_action_norm, dtype=np.float32))

    out = np.concatenate(parts, axis=0)

    # Ensure we never leak NaNs/infs into the RL pipeline.
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    out = np.clip(out, 0.0, 1.0)
    return out


def cupti_dict_to_vector(cupti_norm: Dict[str, float], keys: Sequence[str]) -> np.ndarray:
    return np.array([float(cupti_norm.get(k, 0.0)) for k in keys], dtype=np.float32)
