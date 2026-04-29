#!/usr/bin/env python3
"""Summarize the latest PPO training run based on the saved summary JSON."""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_LOGS_DIR = _REPO_ROOT / "results" / "logs"
_TB_DIR = _LOGS_DIR / "tensorboard"


def _find_latest_training_summary(logs_dir: Path) -> Path | None:
    candidates = list(logs_dir.glob("*/training_summary_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

print("\n" + "=" * 80)
print("TRAINING ANALYSIS - PPO Run Summary")
print("=" * 80)

# Read training summary
summary_path = _find_latest_training_summary(_LOGS_DIR)
if summary_path is None:
    print("\n[ERROR] No training_summary_*.json found under results/logs/*/")
    print("       Run training first: python train_rl.py --total-steps 50000 --max-episode-len 50 --use-nvml")
    raise SystemExit(1)

with open(summary_path) as f:
    summary = json.load(f)

print("\n[LATEST SUMMARY]")
print(f"  {summary_path.relative_to(_REPO_ROOT)}")

print("\n[CONFIGURATION]")
for key in [
    "run_tag",
    "device_name",
    "total_steps",
    "batch_size",
    "learning_rate",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "entropy_coeff",
    "max_episode_len",
    "use_cupti",
    "use_nvml",
]:
    if key in summary:
        print(f"  {key:16s}: {summary[key]}")

print("\n[ARTIFACTS]")
for key in ["model_path", "best_model_path", "log_dir", "tensorboard_dir"]:
    val = summary.get(key)
    if not val:
        continue
    try:
        rel = str(Path(val).resolve().relative_to(_REPO_ROOT))
    except Exception:
        rel = str(val)
    print(f"  {key:16s}: {rel}")

print("\n[VISUALIZE WITH TENSORBOARD]")
print(f"  Run this command in a terminal:")
print(f"  ")
print(f"  tensorboard --logdir {_TB_DIR} --port 6006")
print(f"  ")
print(f"  Then open: http://localhost:6006")
print(f"  ")
print(f"  You'll see:")
print(f"  - Episode reward over time")
print(f"  - Training & value loss convergence")
print(f"  - Learning curves and policy gradients")
print(f"  - PPO-specific metrics (approx_kl, clip_fraction, etc.)")

print("\n[NEXT STEPS]")
print("  1. Evaluate policy with real rollouts: python phase3_rollout_log.py ...")
print("  2. Compare against Phase 0 baseline: results/tables/phase0_baseline.csv")
print("  3. Use the summary JSON to track run-to-run changes")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - Policy Ready For Deployment")
print("=" * 80 + "\n")
