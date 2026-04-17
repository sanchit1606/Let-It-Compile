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
import gc
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure as sb3_configure_logger

from environment.kernel_env import EpisodeConfig, KernelOptimizationEnv


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _REPO_ROOT / "results"
_MODELS_DIR = _RESULTS_DIR / "models"
_LOGS_DIR = _RESULTS_DIR / "logs"


def _slugify_device_name(device_name: str) -> str:
    name = device_name.lower()
    for token in ("nvidia", "geforce", "laptop gpu", "graphics", "gpu"):
        name = name.replace(token, " ")
    name = re.sub(r"\s+", " ", name).strip()

    m = re.search(r"\b(rtx|gtx)\s*([0-9]{3,4})\b", name)
    if m:
        return f"{m.group(1)}{m.group(2)}"

    # Fallback: keep only alphanumerics.
    compact = re.sub(r"[^a-z0-9]+", "", name)
    return compact[:24] if compact else "device"


def _allocate_run_tag(*, base: str, roots: list[Path]) -> str:
    used: set[int] = set()
    pat = re.compile(rf"^{re.escape(base)}_(\d{{2}})\b")
    for root in roots:
        if not root.exists():
            continue
        for p in root.iterdir():
            m = pat.match(p.name)
            if m:
                used.add(int(m.group(1)))

    run_idx = 1
    while run_idx in used:
        run_idx += 1
    return f"{base}_{run_idx:02d}"


def setup_logging(*, run_log_dir: Path, run_tag: str) -> logging.Logger:
    run_log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"train_rl.{run_tag}")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()
    logger.propagate = False
    
    fh = logging.FileHandler(run_log_dir / "train_rl.log")
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class GCCallback(BaseCallback):
    """Periodically perform garbage collection to prevent GPU memory degradation."""
    
    def __init__(self, interval: int = 1000):
        super().__init__()
        self.interval = interval
        self._logger = logging.getLogger(__name__)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.interval == 0:
            gc.collect()
            try:
                import numba.cuda as cuda
                cuda.default_stream().synchronize()
                self._logger.debug(f"GC cleanup at step {self.n_calls}")
            except Exception as e:
                self._logger.debug(f"GC cleanup error at step {self.n_calls}: {e}")
        return True


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
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_slug = _slugify_device_name(gpu_name)
    else:
        gpu_name = "cpu"
        gpu_slug = "cpu"

    run_tag = _allocate_run_tag(
        base=gpu_slug,
        roots=[_MODELS_DIR, _LOGS_DIR, _LOGS_DIR / "tensorboard"],
    )
    run_log_dir = _LOGS_DIR / run_tag
    tb_dir = (_LOGS_DIR / "tensorboard" / run_tag)

    logger = setup_logging(run_log_dir=run_log_dir, run_tag=run_tag)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (_LOGS_DIR / "tensorboard").mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    
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
    logger.info("CUPTI enabled: {}".format(use_cupti))
    logger.info("NVML enabled: {}".format(use_nvml))
    logger.info(f"Run tag: {run_tag}")
    logger.info(f"Device name: {gpu_name}")
    
    # **WARNING: CUPTI during full training is NOT RECOMMENDED on Windows**
    if use_cupti and total_steps > 5000:
        logger.warning("="*80)
        logger.warning("!!! WARNING: CUPTI+full training is EXTREMELY SLOW on Windows !!!")
        logger.warning("Each step calls ncu, which profiles the kernel (5-30 sec). With 50k steps,")
        logger.warning("total time could be 50,000 * 5sec = 250,000 seconds = 70 hours!")
        logger.warning("")
        logger.warning("RECOMMENDATIONS:")
        logger.warning("  1. Use NVML-only mode: --use-nvml (15-30 min for 50k steps)")
        logger.warning("  2. Use CUPTI only for short rollout logging: phase3_rollout_log.py")
        logger.warning("  3. Train with NVML-only, then analyze with CUPTI on small subset")
        logger.warning("="*80)
        time.sleep(3)  # Give user time to read warning
    
    logger.info("Max episode length: {}".format(max_episode_len))
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
            best_model_save_path=str(_MODELS_DIR / f"{run_tag}_best"),
            log_path=str(run_log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=False,
            render=False,
        )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, total_steps // 10),  # ~10 checkpoints over training
        save_path=str(_MODELS_DIR),
        name_prefix=run_tag,
    )
    
    # Garbage collection callback to prevent CUDA memory degradation
    gc_callback = GCCallback(interval=500)  # Clean up every 500 steps
    
    callbacks = [checkpoint_callback, gc_callback]
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
        tensorboard_log=None,
        device=device,
    )

    # Use a stable, GPU-tagged output directory for SB3 logs to avoid PPO_1 / PPO_2 numbering.
    model.set_logger(
        sb3_configure_logger(
            folder=str(tb_dir),
            format_strings=["stdout", "csv", "tensorboard"],
        )
    )
    
    logger.info(f"Starting training for {total_steps} steps...")
    model.learn(
        total_timesteps=total_steps,
        log_interval=1,
        callback=callbacks,
        progress_bar=True,
    )
    
    logger.info(f"Training complete. Saving final model...")
    model.save(str(_MODELS_DIR / run_tag))
    
    # Summary artifact
    summary = {
        "run_tag": run_tag,
        "device_name": gpu_name,
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
        "model_path": str(_MODELS_DIR / f"{run_tag}.zip"),
        "best_model_path": str(_MODELS_DIR / f"{run_tag}_best" / "best_model.zip") if eval_callback else None,
        "log_path": str(run_log_dir),
        "tensorboard_path": str(tb_dir),
    }
    
    summary_path = run_log_dir / "training_summary.json"
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
