#!/usr/bin/env python3
"""Evaluate a trained PPO policy and print basic insights.

This script is intentionally lightweight and does NOT run full rollouts.
It loads a model, prints its spaces, and probes the policy with a few
synthetic observations to show which (block_size, reg_cap) pairs it tends
to choose for each kernel identity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from environment.action_space import BLOCK_SIZES, REG_CAPS, KERNEL_NAMES


_REPO_ROOT = Path(__file__).resolve().parent
_MODELS_DIR = _REPO_ROOT / "results" / "models"


def _default_model_path(models_dir: Path) -> Path:
    candidates = []
    for p in models_dir.glob("*.zip"):
        name = p.name
        # Skip SB3 checkpoint files like <run_tag>_12345_steps.zip
        if name.endswith("_steps.zip"):
            continue
        candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No final model zip found in {models_dir}. "
            f"Expected something like results/models/<run_tag>.zip"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        default=None,
        help="Path to a trained SB3 PPO .zip model (default: latest results/models/<run_tag>.zip)",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="SB3 device: auto|cpu|cuda (default: auto)",
    )
    args = p.parse_args()

    print("\n" + "=" * 80)
    print("POLICY EVALUATION - Learned Configuration Preferences")
    print("=" * 80)

    model_path = Path(args.model_path) if args.model_path else _default_model_path(_MODELS_DIR)
    print(f"\n[1] Loading model: {model_path}")
    model = PPO.load(str(model_path), device=args.device)

    print("\n[2] Model spaces")
    print(f"    Observation space: {model.observation_space}")
    print(f"    Action space: {model.action_space}")

    print("\n[3] Probing policy with synthetic observations")
    print("    Observation layout: CUPTI(4) + NVML(4) + kernel_one_hot(3) + prev_action(2)")

    for kernel_idx, kernel_name in enumerate(KERNEL_NAMES):
        print(f"\n  [{kernel_name.upper()}]")
        for scenario in range(3):
            cupti = np.zeros(4, dtype=np.float32)

            # Mock NVML telemetry (normalized to [0,1])
            nvml = np.array(
                [
                    0.5 + 0.2 * np.sin(scenario),
                    0.3 + 0.1 * np.cos(scenario),
                    0.6,
                    0.4,
                ],
                dtype=np.float32,
            )

            kernel_onehot = np.zeros(len(KERNEL_NAMES), dtype=np.float32)
            kernel_onehot[kernel_idx] = 1.0

            prev_block_idx = scenario % len(BLOCK_SIZES)
            prev_reg_idx = (scenario + 1) % len(REG_CAPS)
            prev_action = np.array(
                [
                    prev_block_idx / max(1, (len(BLOCK_SIZES) - 1)),
                    prev_reg_idx / max(1, (len(REG_CAPS) - 1)),
                ],
                dtype=np.float32,
            )

            obs = np.concatenate([cupti, nvml, kernel_onehot, prev_action]).reshape(1, -1)

            action, _ = model.predict(obs.astype(np.float32), deterministic=True)
            block_idx = int(action[0])
            reg_idx = int(action[1])

            block_size = BLOCK_SIZES[block_idx]
            reg_cap = REG_CAPS[reg_idx]

            print(
                f"    Scenario {scenario + 1}: action=[{block_idx},{reg_idx}] -> "
                f"block_size={block_size} reg_cap={reg_cap}"
            )

    print("\n[4] Next steps")
    print("    - Evaluate with real rollouts: python phase3_rollout_log.py ...")
    print("    - View training curves: tensorboard --logdir results/logs/tensorboard --port 6006")
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
