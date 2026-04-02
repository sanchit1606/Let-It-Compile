"""Phase 1: CUPTI/Nsight Compute counter collection on real kernels.

This phase collects hardware counters (via `ncu`) for the project's *actual*
benchmarks:
- GEMM
- Reduction
- Softmax

It writes a CSV that you can inspect and later feed into the RL environment.

Output:
  results/tables/phase1_result.csv

IMPORTANT (Windows):
- Run from an Administrator Command Prompt for performance counter access.

Example:
    cd /d "C:\\Users\\HP\\Desktop\\CD PROBLEM STATEMENT\\JIT Optimization across GPU stack"
  conda activate gpu-jit-opt
    python experiments\\phase1_collect_counters.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to path (so `import profiling`, `import kernels`, etc. work
# when running as: `python experiments\phase1_collect_counters.py`)
sys.path.insert(0, str(Path(__file__).parent.parent))

from profiling.cupti_collector import CUPTICollector, DEFAULT_NCU_METRICS


# RTX 3050 Ti-friendly defaults (meaningful but not insane for ncu profiling).
KERNELS = ["gemm", "reduction", "softmax"]
MATRIX_SIZES = [512, 1024]
BLOCK_SIZES = [128, 256]
REG_CAPS = [0, 32, 64]  # 0 = default / no cap

# Metrics requested in the Phase 1 objectives.
PHASE1_METRICS: Dict[str, str] = {
    "achieved_occupancy": DEFAULT_NCU_METRICS["achieved_occupancy"],
    "dram_bw_pct": DEFAULT_NCU_METRICS["dram_bw_pct"],
    "l2_hit_rate": DEFAULT_NCU_METRICS["l2_hit_rate"],
    "sm_active_pct": DEFAULT_NCU_METRICS["sm_active_pct"],
    # Optional
    "warp_exec_efficiency_ratio": DEFAULT_NCU_METRICS["warp_exec_efficiency_ratio"],
}

# Optional launch filtering.
# Some Nsight Compute versions don't support these flags (or behave differently),
# which can lead to `ncu` exiting with return code 1.
# Enable explicitly if your `ncu` supports it:
#   set PHASE1_USE_LAUNCH_CONTROL=1
_use_launch_control = os.environ.get("PHASE1_USE_LAUNCH_CONTROL", "0").strip() == "1"

# We do a warmup launch inside the runner; when launch control is enabled we
# profile only the 2nd launch.
NCU_EXTRA_ARGS = ["--launch-skip", "1", "--launch-count", "1"] if _use_launch_control else []

OUTPUT_PATH = Path("results/tables/phase1_result.csv")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_runner_code(kernel_name: str, matrix_size: int, block_size: int, reg_cap: int) -> str:
    """Return a Python script that runs a kernel twice (warmup + measured)."""

    # NOTE: We intentionally do NOT use the warmup built into run_* helpers,
    # because we want exactly two launches so ncu --launch-skip/count is stable.
    # (Numba will compile on the first launch if needed; that's fine.)

    # NOTE: This code is written to a temp file and executed under `ncu`.
    # We must explicitly add the repo root to sys.path so imports like
    # `from kernels.gemm import run_gemm` work regardless of ncu's cwd.
    project_root = str(PROJECT_ROOT)

    return f"""
import sys
from pathlib import Path

sys.path.insert(0, r"{project_root}")

import numpy as np
from numba import cuda

from kernels.gemm import run_gemm
from kernels.reduction import run_reduction
from kernels.softmax import run_softmax

kernel_name = {kernel_name!r}
matrix_size = {int(matrix_size)}
block_size = {int(block_size)}
reg_cap = {int(reg_cap)}

if kernel_name == 'gemm':
    A, B, C, grid, block, kernel_fn = run_gemm(matrix_size, block_size, warmup=0, reg_cap=reg_cap)
    args = (A, B, C, np.int32(matrix_size))

    # warmup
    kernel_fn[grid, block](*args)
    cuda.default_stream().synchronize()

    # measured
    kernel_fn[grid, block](*args)
    cuda.default_stream().synchronize()

elif kernel_name == 'reduction':
    # reduction expects number of elements; Phase 0 uses N*N
    n = matrix_size * matrix_size
    x, out, grid, block, kernel_fn = run_reduction(n, block_size, warmup=0, reg_cap=reg_cap)
    args = (x, out, np.int32(n))
    zero = np.zeros(1, dtype=np.float32)

    # warmup
    out.copy_to_device(zero)
    kernel_fn[grid, block](*args)
    cuda.default_stream().synchronize()

    # measured
    out.copy_to_device(zero)
    kernel_fn[grid, block](*args)
    cuda.default_stream().synchronize()

elif kernel_name == 'softmax':
    x, out, grid, block, kernel_fn = run_softmax(matrix_size, block_size, warmup=0, reg_cap=reg_cap)
    args = (x, out, np.int32(matrix_size), np.int32(matrix_size))

    # warmup
    kernel_fn[grid, block](*args)
    cuda.default_stream().synchronize()

    # measured
    kernel_fn[grid, block](*args)
    cuda.default_stream().synchronize()

else:
    raise ValueError('Unknown kernel')
""".lstrip()


def run_phase1() -> pd.DataFrame:
    """Run Phase 1 sweep and save metrics to CSV."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    collector = CUPTICollector(metrics=PHASE1_METRICS)

    pre = collector.preflight()
    if not pre.ok:
        # Don't crash: write a tiny CSV with the reason so the failure is visible.
        df = pd.DataFrame([
            {
                "ok": False,
                "reason": pre.reason,
                "hint": "Run from an Administrator CMD on Windows, and ensure ncu is installed/on PATH.",
            }
        ])
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"[Phase1] Preflight failed: {pre.reason}")
        print(f"[Phase1] Wrote failure summary to: {OUTPUT_PATH}")
        return df

    rows: List[Dict[str, Any]] = []

    configs = list(product(KERNELS, MATRIX_SIZES, BLOCK_SIZES, REG_CAPS))
    print(f"[Phase1] Total configurations: {len(configs)}")
    print(f"[Phase1] Metrics: {', '.join(PHASE1_METRICS.keys())}")
    print(f"[Phase1] Output: {OUTPUT_PATH}")

    printed_ncu_debug = False

    for kernel_name, matrix_size, block_size, reg_cap in configs:
        code = _make_runner_code(kernel_name, matrix_size, block_size, reg_cap)

        # Try with optional launch control (if enabled), but fall back to a
        # plain run if ncu rejects the flags.
        res = collector.collect_from_python_code(
            code,
            timeout_s=180,
            ncu_extra_args=(NCU_EXTRA_ARGS or None),
        )

        if (not res.ok) and NCU_EXTRA_ARGS:
            # Retry once without extra args.
            res2 = collector.collect_from_python_code(
                code,
                timeout_s=180,
                ncu_extra_args=None,
            )
            if res2.ok:
                res = res2

        if (not res.ok) and (not printed_ncu_debug):
            # Surface the underlying ncu error once; this is usually enough to
            # diagnose missing/unsupported metrics, tool issues, or launch flags.
            stderr_head = (res.stderr or "")[:1200]
            stdout_head = (res.stdout or "")[:800]
            if stderr_head:
                print("[Phase1] ncu stderr (head):")
                print(stderr_head)
            if stdout_head:
                print("[Phase1] ncu stdout (head):")
                print(stdout_head)
            printed_ncu_debug = True

        row: Dict[str, Any] = {
            "kernel": kernel_name,
            "matrix_size": int(matrix_size),
            "block_size": int(block_size),
            "reg_cap": ("default" if not reg_cap else int(reg_cap)),
            "ok": bool(res.ok),
            "reason": res.reason,
        }

        # Raw metrics (as returned by ncu)
        for k in PHASE1_METRICS.keys():
            row[f"{k}_raw"] = res.raw.get(k)
            row[f"{k}_norm"] = res.normalized.get(k)

        rows.append(row)

        status = "OK" if res.ok else "FAIL"
        print(
            f"[Phase1] {status}: {kernel_name} N={matrix_size} block={block_size} regcap={row['reg_cap']} ({res.reason})"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    ok_count = int(df["ok"].sum()) if "ok" in df.columns else 0
    print(f"[Phase1] Done. Rows: {len(df)} (ok={ok_count})")
    print(f"[Phase1] Saved to: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    run_phase1()
