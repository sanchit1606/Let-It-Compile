#!/usr/bin/env python3
"""
Phase 7: RL vs Baselines Comparison

Runs three strategies on each kernel × size combination and produces
the main results table:
  1. PTXAS default (no hints) — baseline config
  2. Random search (N random configurations from the action space)
  3. Trained PPO agent (deterministic rollout)

Expected output table:

Strategy        | Kernel    | Size | Mean time (ms) | Best speedup | Samples
----------------|-----------|------|----------------|--------------|--------
PTXAS default   | gemm      | 512  |  X.XX ± 0.XX   |     ---      |   1
Random search   | gemm      | 512  |  Y.YY ± 0.YY   |   Z.Z%       | 100
PPO agent       | gemm      | 512  |  W.WW ± 0.WW   |   V.V%       |  50

Usage:
    python experiments/phase7_rl_vs_baselines.py --model results/models/rtx3050_01.zip
    python experiments/phase7_rl_vs_baselines.py --model results/models/rtx3050_01.zip --kernels gemm reduction --sizes 256 512 1024

Output:
    results/tables/phase7_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from environment.kernel_env import EpisodeConfig, KernelOptimizationEnv

_RESULTS_TABLES = _REPO_ROOT / "results" / "tables"


# ── Strategy runners ─────────────────────────────────────────────────


def run_ptxas_default(
    kernel: str, size: int, *, warmup: int = 1, repeats: int = 10,
) -> Dict[str, object]:
    """Baseline: run with PTXAS default configuration (block_size=256, reg_cap=0)."""
    cfg = EpisodeConfig(
        kernel_name=kernel,
        matrix_size=size,
        max_steps=1,
        warmup=warmup,
        repeats=repeats,
        use_cupti=False,
        use_nvml=False,
    )
    env = KernelOptimizationEnv(cfg)
    obs, info = env.reset()
    baseline_ms = float(info["baseline_ms"])

    return {
        "strategy": "PTXAS default",
        "kernel": kernel,
        "size": size,
        "time_mean_ms": baseline_ms,
        "time_std_ms": 0.0,
        "best_speedup": 1.0,
        "n_samples": 1,
    }


def run_random_search(
    kernel: str, size: int, *, n_samples: int = 100, warmup: int = 1, repeats: int = 5,
) -> Dict[str, object]:
    """Random search: sample N random configurations from the action space."""
    cfg = EpisodeConfig(
        kernel_name=kernel,
        matrix_size=size,
        max_steps=n_samples,
        warmup=warmup,
        repeats=repeats,
        use_cupti=False,
        use_nvml=False,
    )
    env = KernelOptimizationEnv(cfg)
    obs, info = env.reset()
    baseline_ms = float(info["baseline_ms"])

    all_times: List[float] = []
    best_time = baseline_ms
    best_speedup = 1.0

    for _ in range(n_samples):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        step_time = float(info["time_ms"])
        step_speedup = float(info["speedup"])

        all_times.append(step_time)
        if step_time < best_time:
            best_time = step_time
            best_speedup = step_speedup

        if terminated or truncated:
            obs, info = env.reset()

    return {
        "strategy": "Random search",
        "kernel": kernel,
        "size": size,
        "time_mean_ms": float(np.mean(all_times)),
        "time_std_ms": float(np.std(all_times)),
        "best_speedup": best_speedup,
        "n_samples": n_samples,
    }


def run_ppo_agent(
    model_path: str,
    kernel: str,
    size: int,
    *,
    n_episodes: int = 5,
    max_steps: int = 30,
    warmup: int = 1,
    repeats: int = 5,
) -> Dict[str, object]:
    """Evaluate a trained PPO agent on a specific kernel × size."""
    import torch
    from stable_baselines3 import PPO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, device=device)

    all_best_speedups: List[float] = []
    all_best_times: List[float] = []
    all_step_times: List[float] = []

    for ep in range(n_episodes):
        cfg = EpisodeConfig(
            kernel_name=kernel,
            matrix_size=size,
            max_steps=max_steps,
            warmup=warmup,
            repeats=repeats,
            use_cupti=False,
            use_nvml=False,
        )
        env = KernelOptimizationEnv(cfg)
        obs, info = env.reset(seed=ep)
        baseline_ms = float(info["baseline_ms"])

        best_time = baseline_ms
        best_speedup = 1.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            step_time = float(info["time_ms"])
            step_speedup = float(info["speedup"])

            all_step_times.append(step_time)
            if step_time < best_time:
                best_time = step_time
                best_speedup = step_speedup

        all_best_speedups.append(best_speedup)
        all_best_times.append(best_time)

    return {
        "strategy": "PPO agent",
        "kernel": kernel,
        "size": size,
        "time_mean_ms": float(np.mean(all_best_times)),
        "time_std_ms": float(np.std(all_best_times)),
        "best_speedup": float(np.mean(all_best_speedups)),
        "n_samples": max_steps * n_episodes,
    }


# ── Main comparison ──────────────────────────────────────────────────


def run_comparison(
    *,
    model_path: Optional[str],
    kernels: Sequence[str],
    sizes: Sequence[int],
    n_random: int = 100,
    n_ppo_episodes: int = 5,
    ppo_max_steps: int = 30,
) -> List[Dict[str, object]]:
    """Run the full comparison across all kernel × size combinations."""

    results: List[Dict[str, object]] = []
    total_cases = len(kernels) * len(sizes)
    case_idx = 0

    for kernel in kernels:
        for size in sizes:
            case_idx += 1
            print(f"\n[{case_idx}/{total_cases}] Running: {kernel} size={size}")

            # 1. PTXAS default
            print(f"  Strategy: PTXAS default...", end=" ", flush=True)
            t0 = time.time()
            row_default = run_ptxas_default(kernel, size)
            print(f"{time.time()-t0:.1f}s  time={row_default['time_mean_ms']:.3f}ms")
            results.append(row_default)

            # 2. Random search
            print(f"  Strategy: Random search (N={n_random})...", end=" ", flush=True)
            t0 = time.time()
            row_random = run_random_search(kernel, size, n_samples=n_random)
            print(f"{time.time()-t0:.1f}s  best_speedup={row_random['best_speedup']:.3f}x")
            results.append(row_random)

            # 3. PPO agent (if model path provided)
            if model_path is not None:
                print(f"  Strategy: PPO agent ({n_ppo_episodes} episodes)...", end=" ", flush=True)
                t0 = time.time()
                row_ppo = run_ppo_agent(
                    model_path, kernel, size,
                    n_episodes=n_ppo_episodes,
                    max_steps=ppo_max_steps,
                )
                print(f"{time.time()-t0:.1f}s  best_speedup={row_ppo['best_speedup']:.3f}x")
                results.append(row_ppo)

    return results


def print_table(results: List[Dict[str, object]]) -> None:
    """Print a formatted results table."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Phase 7: RL vs Baselines Comparison")
        table.add_column("Strategy", style="cyan")
        table.add_column("Kernel")
        table.add_column("Size", justify="right")
        table.add_column("Time (ms)", justify="right")
        table.add_column("Best Speedup", justify="right", style="green")
        table.add_column("Samples", justify="right")

        for row in results:
            speedup_str = f"{row['best_speedup']:.3f}x"
            if row["strategy"] == "PTXAS default":
                speedup_str = "1.000x (baseline)"

            table.add_row(
                str(row["strategy"]),
                str(row["kernel"]),
                str(row["size"]),
                f"{row['time_mean_ms']:.3f} ± {row['time_std_ms']:.3f}",
                speedup_str,
                str(int(row["n_samples"])),
            )

        console.print(table)

    except ImportError:
        # Fallback without rich
        header = f"{'Strategy':<16s} {'Kernel':<10s} {'Size':>5s} {'Time (ms)':>14s} {'Speedup':>10s} {'Samples':>8s}"
        print(header)
        print("-" * len(header))
        for row in results:
            print(
                f"{row['strategy']:<16s} {row['kernel']:<10s} {row['size']:>5d} "
                f"{row['time_mean_ms']:>7.3f}±{row['time_std_ms']:<5.3f} "
                f"{row['best_speedup']:>9.3f}x {int(row['n_samples']):>8d}"
            )


def save_results(results: List[Dict[str, object]], path: Path) -> None:
    """Save results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)
    print(f"\nResults saved to: {path}")


def print_insights(results: List[Dict[str, object]]) -> None:
    """Print per-kernel best-config analysis."""
    print("\n" + "=" * 60)
    print("Key Insights")
    print("=" * 60)

    kernels = sorted(set(r["kernel"] for r in results))
    for kernel in kernels:
        kernel_rows = [r for r in results if r["kernel"] == kernel]
        best = max(kernel_rows, key=lambda r: r["best_speedup"])
        default_row = [r for r in kernel_rows if r["strategy"] == "PTXAS default"]
        default_time = default_row[0]["time_mean_ms"] if default_row else 0.0

        print(f"\n  {kernel.upper()}:")
        print(f"    Default time:  {default_time:.3f} ms")
        print(f"    Best strategy: {best['strategy']}")
        print(f"    Best speedup:  {best['best_speedup']:.3f}x")

        # Compare strategies
        strategies = {}
        for r in kernel_rows:
            key = r["strategy"]
            if key not in strategies:
                strategies[key] = []
            strategies[key].append(r["best_speedup"])

        for strat, speedups in strategies.items():
            mean_s = np.mean(speedups)
            print(f"    {strat:>16s}: mean speedup = {mean_s:.3f}x")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Phase 7: RL vs Baselines Comparison")
    p.add_argument(
        "--model", type=str, default=None,
        help="Path to trained PPO model .zip (default: auto-detect newest in results/models/)",
    )
    p.add_argument("--kernels", nargs="+", default=["gemm", "reduction", "softmax"])
    p.add_argument("--sizes", nargs="+", type=int, default=[256, 512])
    p.add_argument("--n-random", type=int, default=100, help="Random search samples (default: 100)")
    p.add_argument("--n-ppo-episodes", type=int, default=5, help="PPO evaluation episodes (default: 5)")
    p.add_argument("--ppo-max-steps", type=int, default=30, help="PPO max steps per episode (default: 30)")
    args = p.parse_args()

    # Auto-detect model if not specified
    model_path = args.model
    if model_path is None:
        models_dir = _REPO_ROOT / "results" / "models"
        if models_dir.exists():
            candidates = [
                p for p in models_dir.glob("*.zip")
                if not p.name.endswith("_steps.zip")
                and "checkpoint" not in p.name
                and "phase_detector" not in p.name
            ]
            if candidates:
                model_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
                print(f"Auto-detected model: {model_path}")

    if model_path is None:
        print("WARNING: No PPO model found. Running PTXAS default + Random search only.")

    print("=" * 60)
    print("Phase 7: RL vs Baselines Comparison")
    print("=" * 60)
    print(f"Kernels: {args.kernels}")
    print(f"Sizes:   {args.sizes}")
    print(f"Random search samples: {args.n_random}")
    if model_path:
        print(f"PPO model: {model_path}")
    print("=" * 60)

    results = run_comparison(
        model_path=model_path,
        kernels=args.kernels,
        sizes=args.sizes,
        n_random=args.n_random,
        n_ppo_episodes=args.n_ppo_episodes,
        ppo_max_steps=args.ppo_max_steps,
    )

    # Output
    out_path = _RESULTS_TABLES / "phase7_comparison.csv"
    save_results(results, out_path)
    print_table(results)
    print_insights(results)

    print("\n" + "=" * 60)
    print("Phase 7 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
