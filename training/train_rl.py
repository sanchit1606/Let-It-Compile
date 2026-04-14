"""Phase 3: PPO training entry point (Stable-Baselines3).

Purpose
- Train a PPO agent on the kernel optimization Gymnasium environment.
- Saves checkpoints, training logs, and evaluation results.

Example (Windows CMD):
    cd /d "C:\\Users\\HP\\Desktop\\CD PROBLEM STATEMENT\\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && python training\\train_rl.py --total-steps 50000 --use-nvml --log-interval 1000

Features:
- Checkpoints saved to results/models/
- Training logs saved to results/logs/
- Optional CUPTI metrics (slow; requires admin on Windows)
- JSON artifact with training summary
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from environment.kernel_env import EpisodeConfig, KernelOptimizationEnv


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _REPO_ROOT / "results"
_MODELS_DIR = _RESULTS_DIR / "models"
_LOGS_DIR = _RESULTS_DIR / "logs"


def setup_logging() -> logging.Logger:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("train_rl")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(_LOGS_DIR / "train_rl.log")
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger


def train_ppo(
    *,
    total_steps: int,
    batch_size: int,
    learning_rate: float,
    n_epochs: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    entropy_coeff: float,
    max_episode_len: int,
    use_cupti: bool,
    use_nvml: bool,
    cupti_timeout_s: int,
    warmup: int,
    repeats: int,
    eval_freq: Optional[int],
    n_eval_episodes: Optional[int],
) -> None:
    logger = setup_logging()
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Phase 3: PPO Training (Stable-Baselines3)")
    logger.info("=" * 80)
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"N epochs: {n_epochs}")
    logger.info(f"Gamma: {gamma}")
    logger.info(f"GAE lambda: {gae_lambda}")
    logger.info(f"Clip range: {clip_range}")
    logger.info(f"Entropy coeff: {entropy_coeff}")
    logger.info(f"CUPTI enabled: {use_cupti}")
    logger.info(f"NVML enabled: {use_nvml}")
    logger.info(f"Max episode length: {max_episode_len}")
    logger.info("=" * 80)
    
    # Training environment
    cfg = EpisodeConfig(
        kernel_name="random",
        matrix_size=256,
        max_steps=max_episode_len,
        warmup=warmup,
        repeats=repeats,
        use_cupti=use_cupti,
        use_nvml=use_nvml,
        cupti_timeout_s=cupti_timeout_s,
    )
    
    logger.info(f"Creating training environment...")
    env = KernelOptimizationEnv(cfg)
    
    # (Optional) Evaluation environment
    eval_env = None
    eval_callback = None
    
    if eval_freq is not None and n_eval_episodes is not None:
        logger.info(f"Creating evaluation environment (eval_freq={eval_freq})...")
        eval_cfg = EpisodeConfig(
            kernel_name="random",
            matrix_size=256,
            max_steps=max_episode_len,
            warmup=warmup,
            repeats=repeats,
            use_cupti=False,  # Eval is NVML-only for speed
            use_nvml=use_nvml,
            cupti_timeout_s=cupti_timeout_s,
        )
        eval_env = KernelOptimizationEnv(eval_cfg)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(_MODELS_DIR / "best"),
            log_path=str(_LOGS_DIR),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=False,
            render=False,
        )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, total_steps // 10),  # ~10 checkpoints over training
        save_path=str(_MODELS_DIR),
        name_prefix="ppo_checkpoint",
    )
    
    callbacks = [checkpoint_callback]
    if eval_callback is not None:
        callbacks.append(eval_callback)
    
    # Detect and set device for PyTorch/SB3
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        logger.warning("CUDA not available. Falling back to CPU (training will be SLOW).")
    
    logger.info(f"Training device: {device}")
    
    # PPO agent
    logger.info(f"Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=batch_size,
        batch_size=min(batch_size, 64),
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=entropy_coeff,
        verbose=1,
        tensorboard_log=str(_LOGS_DIR / "tensorboard"),
        device=device,
    )
    
    logger.info(f"Starting training for {total_steps} steps...")
    model.learn(
        total_timesteps=total_steps,
        log_interval=1,
        callback=callbacks,
        progress_bar=True,
    )
    
    logger.info(f"Training complete. Saving final model...")
    model.save(str(_MODELS_DIR / "ppo_final"))
    
    # Summary artifact
    summary = {
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "entropy_coeff": entropy_coeff,
        "max_episode_len": max_episode_len,
        "use_cupti": use_cupti,
        "use_nvml": use_nvml,
        "warmup": warmup,
        "repeats": repeats,
        "model_path": str(_MODELS_DIR / "ppo_final.zip"),
        "best_model_path": str(_MODELS_DIR / "best" / "best_model.zip") if eval_callback else None,
        "log_path": str(_LOGS_DIR),
    }
    
    summary_path = _LOGS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Training summary saved to {summary_path}")
    logger.info("=" * 80)
    logger.info("Phase 3: PPO Training Complete")
    logger.info("=" * 80)


def main():
    p = argparse.ArgumentParser(description="Phase 3: PPO training entry point")
    
    # Training configuration
    p.add_argument("--total-steps", type=int, default=100000,
                   help="Total environment steps to train on (default: 100000)")
    p.add_argument("--batch-size", type=int, default=2048,
                   help="Number of steps to collect per update (default: 2048)")
    p.add_argument("--learning-rate", type=float, default=3e-4,
                   help="PPO learning rate (default: 3e-4)")
    p.add_argument("--n-epochs", type=int, default=10,
                   help="Number of epochs per update (default: 10)")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor (default: 0.99)")
    p.add_argument("--gae-lambda", type=float, default=0.95,
                   help="GAE lambda (default: 0.95)")
    p.add_argument("--clip-range", type=float, default=0.2,
                   help="PPO clip range (default: 0.2)")
    p.add_argument("--entropy-coeff", type=float, default=0.01,
                   help="Entropy coefficient (default: 0.01)")
    p.add_argument("--max-episode-len", type=int, default=50,
                   help="Max steps per episode (default: 50)")
    
    # Observation configuration
    p.add_argument("--use-cupti", action="store_true",
                   help="Enable CUPTI/ncu collection (slow; may need admin)")
    p.add_argument("--use-nvml", action="store_true", default=True,
                   help="Enable NVML telemetry (default: True)")
    p.add_argument("--cupti-timeout-s", type=int, default=120,
                   help="CUPTI timeout in seconds (default: 120)")
    p.add_argument("--warmup", type=int, default=1,
                   help="Kernel warmup runs (default: 1)")
    p.add_argument("--repeats", type=int, default=5,
                   help="Kernel measurement repeats (default: 5)")
    
    # Evaluation configuration
    p.add_argument("--eval-freq", type=int, default=None,
                   help="Evaluation frequency (steps between evals). If None, no evaluation.")
    p.add_argument("--n-eval-episodes", type=int, default=None,
                   help="Number of evaluation episodes. If None, no evaluation.")
    
    args = p.parse_args()
    
    train_ppo(
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        entropy_coeff=args.entropy_coeff,
        max_episode_len=args.max_episode_len,
        use_cupti=args.use_cupti,
        use_nvml=args.use_nvml,
        cupti_timeout_s=args.cupti_timeout_s,
        warmup=args.warmup,
        repeats=args.repeats,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
    )


if __name__ == "__main__":
    main()
