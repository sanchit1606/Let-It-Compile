"""Phase 3: Rollout logger (Gym environment)

Purpose
- Produce a reproducible, journal-friendly artifact for Phase 3.
- Runs the Phase 3 Gymnasium environment for a small evaluation suite and logs:
  - step-level data: results/tables/phase3_rollout.csv
  - episode-level summary: results/tables/phase3_episode_summary.csv

This is intentionally lightweight:
- Uses random actions by default (no training required).
- CUPTI (ncu) is optional because it is slow and may require Administrator on Windows.

Example (Windows CMD):
    cd /d "C:\\Users\\HP\\Desktop\\CD PROBLEM STATEMENT\\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && python experiments\\phase3_rollout_log.py

"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from environment.kernel_env import EpisodeConfig, KernelOptimizationEnv


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _REPO_ROOT / "results" / "tables"


def _parse_int_list(values: Sequence[str]) -> List[int]:
    out: List[int] = []
    for v in values:
        v = v.strip()
        if not v:
            continue
        out.append(int(v))
    return out


def _ensure_results_dir() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_metrics(prefix: str, keys: Sequence[str], vec: Optional[Sequence[float]]) -> Dict[str, float]:
    if vec is None:
        return {f"{prefix}_{k}_norm": 0.0 for k in keys}

    arr = np.asarray(vec, dtype=np.float32)
    out: Dict[str, float] = {}
    for i, k in enumerate(keys):
        out[f"{prefix}_{k}_norm"] = float(arr[i]) if i < arr.size else 0.0
    return out


def _episode_suite(kernels: Sequence[str], matrix_sizes: Sequence[int]) -> List[Tuple[str, int]]:
    return [(k, int(n)) for k in kernels for n in matrix_sizes]


def run_rollouts(
    *,
    cases: Iterable[Tuple[str, int]],
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

        # Reuse one environment instance per case to:
        # - avoid repeated NVML initialization spam
        # - amortize Python-side setup
        env = KernelOptimizationEnv(cfg)

        print(f"[phase3] case={case_id} kernel={kernel_name} N={int(matrix_size)} episodes={int(episodes_per_case)}")

        for ep in range(int(episodes_per_case)):
            episode_id = case_id * int(episodes_per_case) + ep
            episode_seed = int(seed + episode_id)

            print(f"[phase3]  episode={episode_id} seed={episode_seed} steps={int(max_steps)} cupti={bool(use_cupti)} nvml={bool(use_nvml)}")

            obs, info0 = env.reset(seed=episode_seed)

            # Metrics schema is driven by the env's obs spec.
            cupti_keys = list(info0.get("cupti_keys") or [])

            baseline_ms = float(info0["baseline_ms"])
            best_speedup = 0.0
            best_reward = -1e9
            sum_reward = 0.0
            sum_time = 0.0

            truncated = False
            terminated = False

            step = 0
            while not (terminated or truncated):
                action = env.action_space.sample()
                obs2, reward, terminated, truncated, info = env.step(action)

                cupti_vec = info.get("cupti_vec")
                nvml_vec = info.get("nvml_vec")

                row: Dict[str, object] = {
                    "episode_id": int(episode_id),
                    "episode_seed": int(episode_seed),
                    "step": int(step),
                    "kernel": str(info.get("kernel")),
                    "matrix_size": int(info.get("matrix_size")),
                    "block_size": int(info.get("block_size")),
                    "reg_cap": int(info.get("reg_cap")),
                    "time_ms": float(info.get("time_ms")),
                    "baseline_ms": float(info.get("baseline_ms")),
                    "speedup": float(info.get("speedup")),
                    "reward": float(reward),
                    "cupti_ok": bool(info.get("cupti_ok")),
                    "cupti_reason": str(info.get("cupti_reason")),
                }

                row.update(_flatten_metrics("cupti", cupti_keys, cupti_vec))

                # NVML has a fixed meaning in profiling/nvml_monitor.py
                # [gpu_util, mem_util, mem_used_frac, temp_norm]
                row.update(
                    {
                        "nvml_gpu_util_norm": float(nvml_vec[0]) if isinstance(nvml_vec, list) and len(nvml_vec) > 0 else 0.0,
                        "nvml_mem_util_norm": float(nvml_vec[1]) if isinstance(nvml_vec, list) and len(nvml_vec) > 1 else 0.0,
                        "nvml_mem_used_frac": float(nvml_vec[2]) if isinstance(nvml_vec, list) and len(nvml_vec) > 2 else 0.0,
                        "nvml_temp_norm": float(nvml_vec[3]) if isinstance(nvml_vec, list) and len(nvml_vec) > 3 else 0.0,
                    }
                )

                step_rows.append(row)

                speedup = float(info.get("speedup"))
                best_speedup = max(best_speedup, speedup)
                best_reward = max(best_reward, float(reward))
                sum_reward += float(reward)
                sum_time += float(info.get("time_ms"))

                obs = obs2
                step += 1

            steps_done = max(1, step)

            episode_rows.append(
                {
                    "episode_id": int(episode_id),
                    "episode_seed": int(episode_seed),
                    "kernel": str(kernel_name),
                    "matrix_size": int(matrix_size),
                    "max_steps": int(max_steps),
                    "use_cupti": bool(use_cupti),
                    "use_nvml": bool(use_nvml),
                    "baseline_ms": float(baseline_ms),
                    "mean_time_ms": float(sum_time / steps_done),
                    "mean_reward": float(sum_reward / steps_done),
                    "best_speedup": float(best_speedup),
                    "best_reward": float(best_reward),
                    "steps": int(steps_done),
                }
            )

            print(f"[phase3]  episode={episode_id} done best_speedup={best_speedup:.3f} mean_reward={float(sum_reward / steps_done):.4f}")

    _write_csv(out_steps_csv, step_rows)
    _write_csv(out_episodes_csv, episode_rows)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write")

    # Stable column order: deterministic based on first row.
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Phase 3 rollout logger")

    p.add_argument("--kernels", nargs="+", default=["gemm", "reduction", "softmax"], help="Kernel names")
    p.add_argument("--matrix-sizes", nargs="+", default=[64], help="Matrix sizes (N labels)")

    p.add_argument("--episodes-per-case", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--use-cupti", action="store_true", help="Enable ncu-based metrics (slow; may need Admin)")
    p.add_argument("--use-nvml", action="store_true", help="Enable NVML telemetry (default off in this script)")

    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--cupti-timeout-s", type=int, default=120)

    p.add_argument(
        "--out-steps",
        type=str,
        default=str(_RESULTS_DIR / "phase3_rollout.csv"),
        help="Step-level CSV output path",
    )
    p.add_argument(
        "--out-episodes",
        type=str,
        default=str(_RESULTS_DIR / "phase3_episode_summary.csv"),
        help="Episode-level CSV output path",
    )

    args = p.parse_args(list(argv) if argv is not None else None)

    matrix_sizes = _parse_int_list([str(x) for x in args.matrix_sizes])
    cases = _episode_suite(args.kernels, matrix_sizes)

    run_rollouts(
        cases=cases,
        episodes_per_case=int(args.episodes_per_case),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        use_cupti=bool(args.use_cupti),
        use_nvml=bool(args.use_nvml),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        cupti_timeout_s=int(args.cupti_timeout_s),
        out_steps_csv=Path(args.out_steps),
        out_episodes_csv=Path(args.out_episodes),
    )

    print("Phase 3 rollout logging complete")
    print(f"Step-level CSV: {Path(args.out_steps)}")
    print(f"Episode summary CSV: {Path(args.out_episodes)}")


if __name__ == "__main__":
    main()
