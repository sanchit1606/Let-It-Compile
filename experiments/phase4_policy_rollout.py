#!/usr/bin/env python3
"""Phase 4: Evaluate a trained PPO policy on benchmark cases.

This runs the real `KernelOptimizationEnv` and uses the PPO model to pick actions.
It writes GPU-tagged CSV artifacts (step-level + episode-level) into results/tables/.

Example (Windows CMD):
  python experiments\\phase4_policy_rollout.py --model results\\models\\rtx3050_01.zip --use-nvml \
    --kernels gemm reduction softmax --matrix-sizes 256 512 --episodes-per-case 2 --max-steps 30

Notes:
- `--use-cupti` is extremely slow on Windows (each step can invoke ncu).
- For reliable evaluation, prefer NVML-only and larger `repeats`.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from environment.kernel_env import EpisodeConfig, KernelOptimizationEnv


_RESULTS_TABLES = _REPO_ROOT / "results" / "tables"


def _ensure_results_dir() -> None:
    _RESULTS_TABLES.mkdir(parents=True, exist_ok=True)


def _parse_int_list(items: Sequence[str]) -> List[int]:
    out: List[int] = []
    for x in items:
        s = str(x).strip()
        if not s:
            continue
        out.append(int(s))
    return out


def _infer_run_tag_from_model_path(model_path: Path) -> str:
    stem = model_path.stem
    # Common: rtx3050_01.zip
    if re.match(r"^[a-z0-9]+_\d{2}$", stem):
        return stem
    return stem


def _infer_gpu_tag_from_run_tag(run_tag: str) -> str:
    return run_tag.split("_", 1)[0] if "_" in run_tag else run_tag


def _episode_suite(kernels: Sequence[str], matrix_sizes: Sequence[int]) -> List[Tuple[str, int]]:
    cases: List[Tuple[str, int]] = []
    for k in kernels:
        for n in matrix_sizes:
            cases.append((str(k), int(n)))
    return cases


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_policy_rollouts(
    *,
    model_path: Path,
    cases: Sequence[Tuple[str, int]],
    episodes_per_case: int,
    max_steps: int,
    seed: int,
    use_cupti: bool,
    use_nvml: bool,
    warmup: int,
    repeats: int,
    cupti_timeout_s: int,
    out_steps_csv: Path,
    out_episodes_csv: Path,
) -> None:
    _ensure_results_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(str(model_path), device=device)

    step_rows: List[Dict[str, object]] = []
    episode_rows: List[Dict[str, object]] = []

    for case_id, (kernel_name, matrix_size) in enumerate(cases):
        cfg = EpisodeConfig(
            kernel_name=str(kernel_name),
            matrix_size=int(matrix_size),
            max_steps=int(max_steps),
            warmup=int(warmup),
            repeats=int(repeats),
            use_cupti=bool(use_cupti),
            use_nvml=bool(use_nvml),
            cupti_timeout_s=int(cupti_timeout_s),
        )

        env = KernelOptimizationEnv(cfg)

        print(
            f"[phase4] case={case_id} kernel={kernel_name} N={int(matrix_size)} "
            f"episodes={int(episodes_per_case)} cupti={bool(use_cupti)} nvml={bool(use_nvml)}"
        )

        for ep in range(int(episodes_per_case)):
            episode_id = case_id * int(episodes_per_case) + ep
            episode_seed = int(seed + episode_id)

            obs, info0 = env.reset(seed=episode_seed)

            baseline_ms = float(info0["baseline_ms"])
            sum_reward = 0.0
            best_reward = -1e9
            best_speedup = 0.0

            truncated = False
            terminated = False
            step = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                rr_speedup = float(info.get("speedup", np.nan))
                rr_reward = float(reward)

                sum_reward += rr_reward
                if rr_reward > best_reward:
                    best_reward = rr_reward
                if rr_speedup > best_speedup:
                    best_speedup = rr_speedup

                # Try to normalize action format into indices when possible.
                a0 = None
                a1 = None
                try:
                    a_arr = np.asarray(action).reshape(-1)
                    if a_arr.size >= 2:
                        a0 = int(a_arr[0])
                        a1 = int(a_arr[1])
                except Exception:
                    pass

                row: Dict[str, object] = {
                    "episode_id": int(episode_id),
                    "episode_seed": int(episode_seed),
                    "step": int(step),
                    "policy": "ppo",
                    "kernel": str(info.get("kernel")),
                    "matrix_size": int(info.get("matrix_size")),
                    "action_block_idx": a0,
                    "action_regcap_idx": a1,
                    "block_size": int(info.get("block_size")),
                    "reg_cap": int(info.get("reg_cap")),
                    "time_ms": float(info.get("time_ms")),
                    "baseline_ms": float(info.get("baseline_ms")),
                    "speedup": float(info.get("speedup")),
                    "reward": float(rr_reward),
                    "cupti_ok": bool(info.get("cupti_ok")),
                    "cupti_reason": str(info.get("cupti_reason")),
                }

                nvml_vec = info.get("nvml_vec")
                if isinstance(nvml_vec, list) and len(nvml_vec) == 4:
                    row.update(
                        {
                            "nvml_gpu_util": float(nvml_vec[0]),
                            "nvml_mem_util": float(nvml_vec[1]),
                            "nvml_mem_used_frac": float(nvml_vec[2]),
                            "nvml_temp_norm": float(nvml_vec[3]),
                        }
                    )

                step_rows.append(row)
                step += 1

            episode_rows.append(
                {
                    "episode_id": int(episode_id),
                    "episode_seed": int(episode_seed),
                    "policy": "ppo",
                    "kernel": str(kernel_name),
                    "matrix_size": int(matrix_size),
                    "max_steps": int(max_steps),
                    "steps_executed": int(step),
                    "baseline_ms": float(baseline_ms),
                    "sum_reward": float(sum_reward),
                    "mean_reward": float(sum_reward / max(1, step)),
                    "best_reward": float(best_reward),
                    "best_speedup": float(best_speedup),
                    "speedup_from_mean_reward": float((sum_reward / max(1, step)) + 1.0),
                }
            )

    _write_csv(out_steps_csv, step_rows)
    _write_csv(out_episodes_csv, episode_rows)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Phase 4 policy rollout evaluator")

    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to PPO model zip (default: newest results/models/<run_tag>.zip)",
    )

    p.add_argument("--kernels", nargs="+", default=["gemm", "reduction", "softmax"], help="Kernel names")
    p.add_argument("--matrix-sizes", nargs="+", default=[256], help="Matrix sizes (N labels)")

    p.add_argument("--episodes-per-case", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--use-cupti", action="store_true", help="Enable ncu-based metrics (very slow)")
    p.add_argument("--use-nvml", action="store_true", help="Enable NVML telemetry")

    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--cupti-timeout-s", type=int, default=120)

    args = p.parse_args(list(argv) if argv is not None else None)

    models_dir = _REPO_ROOT / "results" / "models"
    if args.model is not None:
        model_path = Path(args.model)
    else:
        candidates = [p for p in models_dir.glob("*.zip") if not p.name.endswith("_steps.zip")]
        if not candidates:
            raise SystemExit(f"No model zips found in {models_dir}")
        model_path = max(candidates, key=lambda p: p.stat().st_mtime)

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    run_tag = _infer_run_tag_from_model_path(model_path)
    gpu_tag = _infer_gpu_tag_from_run_tag(run_tag)

    out_steps = _RESULTS_TABLES / f"phase4_policy_rollout_{gpu_tag}.csv"
    out_eps = _RESULTS_TABLES / f"phase4_policy_episode_summary_{gpu_tag}.csv"

    matrix_sizes = _parse_int_list([str(x) for x in args.matrix_sizes])
    cases = _episode_suite(args.kernels, matrix_sizes)

    run_policy_rollouts(
        model_path=model_path,
        cases=cases,
        episodes_per_case=int(args.episodes_per_case),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        use_cupti=bool(args.use_cupti),
        use_nvml=bool(args.use_nvml),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        cupti_timeout_s=int(args.cupti_timeout_s),
        out_steps_csv=out_steps,
        out_episodes_csv=out_eps,
    )

    print("Phase 4 policy evaluation complete")
    print(f"Model: {model_path}")
    print(f"Step-level CSV: {out_steps}")
    print(f"Episode summary CSV: {out_eps}")


if __name__ == "__main__":
    main()
