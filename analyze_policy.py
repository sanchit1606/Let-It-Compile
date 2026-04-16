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

print("=" * 80)
print("Phase 3: Analyzing Trained PPO Policy")
print("=" * 80)

# Load the trained model
model_path = _MODELS_DIR / "ppo_final.zip"
if not model_path.exists():
    print(f"[ERROR] Model not found at {model_path}")
    exit(1)

print(f"\n[LOAD] Loading trained model from: {model_path}")
model = PPO.load(str(model_path), device="cuda")
print(f"[OK] Model loaded successfully")

# Get model info
print(f"\n[MODEL] Architecture:")
print(f"  Policy: {model.policy.__class__.__name__}")
print(f"  Learning rate: {model.learning_rate}")
print(f"  Batch size: {model.batch_size}")
print(f"  N epochs: {model.n_epochs}")
print(f"  Gamma: {model.gamma}")

# Load training summary
summary_path = _LOGS_DIR / "training_summary.json"
if summary_path.exists():
    with open(summary_path) as f:
        summary = json.load(f)
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
print(f"  Model: {model_path}")
print(f"  Best model: {_MODELS_DIR / 'best' / 'best_model.zip'}")
print(f"  Training logs: {_LOGS_DIR / 'tensorboard'}")
print(f"  Summary: {summary_path}")
