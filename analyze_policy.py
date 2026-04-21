#!/usr/bin/env python3
"""Analyze the trained PPO policy and generate insights."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import torch

_REPO_ROOT = Path(__file__).resolve().parent
_MODELS_DIR = _REPO_ROOT / "results" / "models"
_LOGS_DIR = _REPO_ROOT / "results" / "logs"


def _find_latest_training_summary(logs_dir: Path) -> Path | None:
    candidates = list(logs_dir.glob("*/training_summary_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

print("=" * 80)
print("Phase 3: Analyzing Trained PPO Policy")
print("=" * 80)

# Load the trained model
summary_path = _find_latest_training_summary(_LOGS_DIR)
summary = None
if summary_path is not None and summary_path.exists():
    with open(summary_path) as f:
        summary = json.load(f)

model_path = None
best_model_path = None
if summary is not None:
    model_path = Path(summary.get("model_path", "")) if summary.get("model_path") else None
    best_model_path = Path(summary.get("best_model_path", "")) if summary.get("best_model_path") else None

chosen_model_path = None
if best_model_path is not None and best_model_path.exists():
    chosen_model_path = best_model_path
elif model_path is not None and model_path.exists():
    chosen_model_path = model_path
else:
    # Fallback: newest .zip in results/models (prefer non-checkpoint files).
    candidates = [p for p in _MODELS_DIR.glob("*.zip") if not p.name.endswith("_steps.zip")]
    if candidates:
        chosen_model_path = max(candidates, key=lambda p: p.stat().st_mtime)

if chosen_model_path is None or not chosen_model_path.exists():
    print("[ERROR] Could not find a trained model to load.")
    print(f"  Looked for: {summary_path if summary_path else 'no training summary found'}")
    exit(1)

print(f"\n[LOAD] Loading trained model from: {chosen_model_path}")
model = PPO.load(str(chosen_model_path), device="cuda" if torch.cuda.is_available() else "cpu")
print(f"[OK] Model loaded successfully")

# Get model info
print(f"\n[MODEL] Architecture:")
print(f"  Policy: {model.policy.__class__.__name__}")
print(f"  Learning rate: {model.learning_rate}")
print(f"  Batch size: {model.batch_size}")
print(f"  N epochs: {model.n_epochs}")
print(f"  Gamma: {model.gamma}")

# Load training summary
if summary is not None:
    print(f"[TRAIN] Configuration:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Max episode length: {summary['max_episode_len']}")
    print(f"  CUPTI enabled: {summary['use_cupti']}")
    print(f"  NVML enabled: {summary['use_nvml']}")
    print(f"  Batch size: {summary['batch_size']}")

# Analyze policy network weights
print(f"\n[NETWORK] Policy Analysis:")
policy_net = model.policy.pi_features_extractor
value_net = model.policy.value_features_extractor

print(f"  Policy feature extractor layers:")
for name, module in policy_net.named_modules():
    if isinstance(module, torch.nn.Linear):
        weight = module.weight.data.cpu().numpy()
        print(f"    {name}: shape={weight.shape}, mean={np.mean(weight):.4f}, std={np.std(weight):.4f}")

# Get action distribution
print(f"\n[ACTION] Space Analysis:")
print(f"  Action space: {model.env.action_space}")
if hasattr(model.env, 'action_space') and hasattr(model.env.action_space, 'nvec'):
    block_sizes = model.env.envs[0].action_space_config['block_sizes']
    reg_caps = model.env.envs[0].action_space_config['reg_caps']
    print(f"  Block sizes: {block_sizes}")
    print(f"  Register caps: {reg_caps}")
    print(f"  Total actions: {len(block_sizes) * len(reg_caps)}")

# Analyze policy in inference mode
print(f"\n[INFERENCE] Test:")
print(f"  Testing on sample observations...")

try:
    # Get a test observation from environment
    obs = model.env.reset()
    actions, _states = model.predict(obs, deterministic=True)
    print(f"  Sample action: {actions}")
    print(f"  Observation shape: {obs.shape}")
    
    # Run a few inference steps
    print(f"\n  Running 10 inference steps (deterministic):")
    for i in range(10):
        obs, rewards, dones, info = model.env.step(actions)
        actions, _states = model.predict(obs, deterministic=True)
        print(f"    Step {i+1}: action={actions[0]}, reward={rewards[0]:.3f}")
        if dones[0]:
            obs = model.env.reset()
except Exception as e:
    print(f"  [WARN] Could not run inference test: {e}")

print(f"\n" + "=" * 80)
print("[OK] Analysis Complete")
print("=" * 80)

# TensorBoard info
print(f"\n[TENSORBOARD] To visualize training metrics:")
print(f"  Command: tensorboard --logdir {_LOGS_DIR}/tensorboard --port 6006")
print(f"  Then open: http://localhost:6006 in your browser")

print(f"\n[ARTIFACTS] Files:")
print(f"  Model: {chosen_model_path}")
if best_model_path is not None:
    print(f"  Best model: {best_model_path}")
print(f"  Training logs: {_LOGS_DIR / 'tensorboard'}")
print(f"  Summary: {summary_path if summary_path else 'N/A'}")
