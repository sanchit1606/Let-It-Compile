#!/bin/bash
# REFERENCE: Correct Commands for GPU Kernel Optimization Project

# ============================================================================
# SETUP (Run Once)
# ============================================================================

# Create conda environment
conda create -n gpu-jit-opt python=3.10 -y
conda activate gpu-jit-opt

# Install dependencies
pip install -r requirements.txt

# Verify setup
python check_cuda.py  # Should show: CUDA available: True


# ============================================================================
# PHASE 0: BASELINE (3 seconds)
# ============================================================================

python experiments/phase0_baseline_table.py


# ============================================================================
# PHASE 2: KERNEL VALIDATION (5-10 seconds)
# ============================================================================

# Quick tests
pytest tests/test_kernels.py -v

# Performance-scale tests (optional, slower)
pytest tests/test_kernels.py -v --runslow


# ============================================================================
# PHASE 3: TRAINING
# ============================================================================

# ✅ RECOMMENDED: NVML-ONLY (15-30 minutes)
python train_rl.py --total-steps 50000 --max-episode-len 50 --use-nvml --eval-freq 5000 --n-eval-episodes 5

# Quick test (1-2 minutes)
python train_rl.py --total-steps 1000 --max-episode-len 20 --use-nvml

# Minimal viable (5-10 minutes)
python train_rl.py --total-steps 5000 --max-episode-len 30 --use-nvml


# ============================================================================
# PHASE 3: ANALYSIS (Optional, 30-60 minutes)
# ============================================================================

# After training completes, collect CUPTI metrics on small subset
python phase3_rollout_log.py --use-cupti --use-nvml --kernels gemm reduction softmax --matrix-sizes 256 512 --episodes-per-case 3 --max-steps 10


# ============================================================================
# COMPLETE WORKFLOW (~45 minutes total)
# ============================================================================

# 1. Baseline (3 sec)
python experiments/phase0_baseline_table.py

# 2. Training (20-30 min)
python train_rl.py --total-steps 50000 --max-episode-len 50 --use-nvml

# 3. Analysis [OPTIONAL] (30-60 min)
python phase3_rollout_log.py --use-cupti --use-nvml --kernels gemm reduction softmax --matrix-sizes 256 512 --episodes-per-case 3 --max-steps 10


# ============================================================================
# TENSORBOARD (While training in another terminal)
# ============================================================================

tensorboard --logdir results/logs/tensorboard


# ============================================================================
# OUTPUT LOCATIONS
# ============================================================================

# Phase 0 baseline
results/tables/phase0_baseline.csv

# Phase 3 rollout logs
results/tables/phase3_rollout.csv
results/tables/phase3_episode_summary.csv

# Trained models
results/models/ppo_final.zip
results/models/best/best_model.zip (if eval enabled)

# Training logs
results/logs/train_rl.log
results/logs/tensorboard/


# ============================================================================
# ⚠️ DO NOT USE (EXTREMELY SLOW - 70-400 hours!)
# ============================================================================

# ❌ NEVER: python train_rl.py --total-steps 50000 --use-cupti --use-nvml
# This calls ncu profiling on every step = 50k × 5-30sec = days to months!


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# CUDA not available
python check_cuda.py

# If CUDA not found, reinstall PyTorch:
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# If training was interrupted, restart from beginning:
rm -rf results/models/*
rm -rf results/logs/*
python train_rl.py --total-steps 50000 --use-nvml
