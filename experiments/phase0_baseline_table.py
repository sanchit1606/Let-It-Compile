"""
Phase 0: Foundational Experiment
=================================

Vary --maxrregcount × kernel_type × matrix_size and measure:
  - Theoretical occupancy (computed from register count and hardware limits)
  - Kernel runtime (ms, CUDA event timing)
  - Performance characteristics

Expected to produce a table like:

kernel      | size  | block  | regs/t | theor_occ | time_ms_mean | time_ms_std
------------|-------|--------|--------|-----------|--------------|------------
gemm        | 512   | 256    | 32     | 100%      | 4.2          | 0.1
gemm        | 512   | 256    | 64     | 68%       | 3.8          | 0.1
gemm        | 512   | 256    | 100    | 41%       | 5.1          | 0.2
gemm        | 2048  | 256    | 32     | 100%      | 89.3         | 0.5
...

Run with: python experiments/phase0_baseline_table.py
Results saved to: results/tables/phase0_baseline.csv

This Phase 0 implementation:
- Times kernels using CUDA events (most accurate, no profiling overhead)
- Computes theoretical occupancy from register estimates
- Works on Windows (uses tempfile instead of /tmp)
- Does not require ncu (Nsight Compute)
- Produces the foundational table that validates the research hypothesis
"""

import os
import sys
import shutil
from pathlib import Path
from itertools import product
import subprocess
import tempfile
import csv
import io

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table as RichTable
from rich.progress import track

def _configure_numba_cuda_windows() -> None:
    """Best-effort Windows setup so Numba can find NVVM + libdevice.

    Numba's CUDA JIT needs an NVVM DLL and libdevice bitcode.
    On Windows, we prefer conda's CUDA 12.1 NVVM (just installed),
    then fall back to system CUDA installation.
    """

    if os.name != "nt":
        return

    # IMPORTANT:
    # On some Windows setups, Numba works out-of-the-box by finding a system CUDA
    # toolkit, and forcing an alternate NVVM/libdevice can crash compilation.
    # Therefore we only apply this override when explicitly requested.
    if os.environ.get("PHASE0_FORCE_NUMBA_CUDA_CONFIG", "0") not in ("1", "true", "True"):
        return

    # If user already provided a working NVVM + libdevice path, keep it.
    # Accept both modern NUMBA_CUDA_* and legacy NUMBAPRO_* names.
    existing_nvvm = os.environ.get("NUMBA_CUDA_NVVM") or os.environ.get("NUMBAPRO_NVVM")
    existing_libdevice = os.environ.get("NUMBA_CUDA_LIBDEVICE") or os.environ.get("NUMBAPRO_LIBDEVICE")
    if (
        existing_nvvm
        and existing_libdevice
        and Path(existing_nvvm).exists()
        and Path(existing_libdevice).exists()
    ):
        return

    def _pick_first_existing(paths):
        for p in paths:
            if p and p.exists():
                return p
        return None

    def _find_nvvm_and_libdevice_in_root(root: Path) -> tuple[Path | None, Path | None]:
        """Return (nvvm_dll, libdevice_bc_file) if discoverable under root."""

        if not root.exists():
            return None, None

        # NVVM DLL candidates (conda + system CUDA installs vary here).
        nvvm_candidates = [
            root / "bin" / "nvvm.dll",  # older layouts
            root / "bin" / "nvvm64_40_0.dll",
            root / "nvvm" / "bin" / "nvvm.dll",
            root / "nvvm" / "bin" / "nvvm64_40_0.dll",
            root / "nvvm" / "bin" / "x64" / "nvvm64_40_0.dll",
        ]

        # Also accept any nvvm*.dll we can find in plausible locations.
        for search_dir in [root / "bin", root / "nvvm" / "bin", root / "nvvm" / "bin" / "x64"]:
            if search_dir.exists():
                matches = list(search_dir.glob("nvvm*.dll"))
                nvvm_candidates.extend(matches)

        nvvm_dll = _pick_first_existing(nvvm_candidates)

        # libdevice bitcode candidates.
        libdevice_dirs = [
            root / "nvvm" / "libdevice",
            root / "Library" / "nvvm" / "libdevice",  # if root is CONDA_PREFIX
            root / "share" / "cuda" / "libdevice",
        ]
        libdevice_file = None
        for d in libdevice_dirs:
            if d.exists():
                first_bc = next(d.glob("libdevice*.bc"), None)
                if first_bc:
                    libdevice_file = first_bc
                    break

        # Last resort: search a bit deeper under root for libdevice*.bc.
        if libdevice_file is None:
            try:
                # Limit search area to keep this reasonably fast.
                for base in [root / "nvvm", root / "Library", root]:
                    if base.exists():
                        found = next(base.rglob("libdevice*.bc"), None)
                        if found:
                            libdevice_file = found
                            break
            except Exception:
                libdevice_file = None

        return nvvm_dll, libdevice_file

    # Try conda environment first
    conda_prefix = Path(os.environ.get("CONDA_PREFIX", ""))
    if conda_prefix.exists():
        # In conda, the CUDA root is typically CONDA_PREFIX\Library
        conda_cuda_root = conda_prefix / "Library"
        nvvm_dll, libdevice_file = _find_nvvm_and_libdevice_in_root(conda_cuda_root)
        if nvvm_dll and libdevice_file:
            # Numba 0.59 on Windows may prefer "Conda environment" lookup that searches
            # libdevice beside nvvm.dll (often under Library\bin). If only
            # Library\nvvm\libdevice exists, JIT can incorrectly report "Missing libdevice file".
            # Create a compatibility shim by copying libdevice*.bc next to nvvm.dll.
            try:
                shim_target = nvvm_dll.parent / libdevice_file.name
                if not shim_target.exists():
                    shutil.copy2(libdevice_file, shim_target)
            except Exception:
                # Best effort only - continue with explicit env setup below.
                pass

            # Override CUDA_HOME to a known-good toolkit root.
            os.environ["CUDA_HOME"] = str(conda_cuda_root)
            os.environ["NUMBA_CUDA_NVVM"] = str(nvvm_dll)
            os.environ["NUMBA_CUDA_LIBDEVICE"] = str(libdevice_file)
            os.environ["NUMBAPRO_NVVM"] = str(nvvm_dll)
            os.environ["NUMBAPRO_LIBDEVICE"] = str(libdevice_file)
            try:
                os.add_dll_directory(str(nvvm_dll.parent))
            except Exception:
                pass
            print(f"[Phase0] Numba CUDA configured (conda): NVVM={nvvm_dll}")
            print(f"[Phase0] Numba CUDA configured (conda): LIBDEVICE={libdevice_file}")
            return

    # Fallback: discover system CUDA root
    candidates = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(Path(cuda_home))

    default_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if default_root.exists():
        # Prefer highest version folder (v12.x, v13.x, ...)
        for child in default_root.glob("v*"):
            if child.is_dir():
                candidates.append(child)

    # De-dup while preserving order
    seen = set()
    unique = []
    for c in candidates:
        c = c.resolve()
        if str(c).lower() not in seen:
            seen.add(str(c).lower())
            unique.append(c)

    # Try to locate NVVM DLL + libdevice in system CUDA installs
    for root in sorted(unique, key=lambda p: str(p), reverse=True):
        nvvm_dll, libdevice_file = _find_nvvm_and_libdevice_in_root(root)
        if nvvm_dll and libdevice_file:
            # Override CUDA_HOME to a known-good toolkit root.
            os.environ["CUDA_HOME"] = str(root)
            os.environ["NUMBA_CUDA_NVVM"] = str(nvvm_dll)
            os.environ["NUMBA_CUDA_LIBDEVICE"] = str(libdevice_file)
            os.environ["NUMBAPRO_NVVM"] = str(nvvm_dll)
            os.environ["NUMBAPRO_LIBDEVICE"] = str(libdevice_file)

            # Help Windows DLL loader find dependencies.
            try:
                os.add_dll_directory(str(nvvm_dll.parent))
            except Exception:
                pass
            try:
                os.add_dll_directory(str(root / "bin"))
            except Exception:
                pass
            print(f"[Phase0] Numba CUDA configured (system): CUDA_HOME={root}")
            print(f"[Phase0] Numba CUDA configured (system): NVVM={nvvm_dll}")
            print(f"[Phase0] Numba CUDA configured (system): LIBDEVICE={libdevice_file}")
            return

    # If we get here, we didn't find a usable toolkit layout.
    # Don't crash here; the preflight will emit a clear error.
    print("[Phase0] Warning: Could not auto-locate NVVM + libdevice for Numba CUDA JIT on Windows.")


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure Numba CUDA before importing modules that import `numba.cuda`
# By default, we do NOT override Numba's CUDA toolchain discovery on Windows.
# If you hit NVVM/libdevice discovery errors, re-run with:
#   set PHASE0_FORCE_NUMBA_CUDA_CONFIG=1
_configure_numba_cuda_windows()

from profiling.cuda_timer import time_kernel
from profiling.ncu_utils import ncu_metric_smoke_test
from compiler.ptxas_controller import OccupancyCalculator
from kernels.gemm import run_gemm
from kernels.reduction import run_reduction
from kernels.softmax import run_softmax

console = Console()

# ── Configuration ────────────────────────────────────────────────────
# Register cap levels to test
# For now, skip high register caps that cause context destruction on Windows
# REG_CAPS = [0, 32, 48, 64, 80, 96, 128]
REG_CAPS = [0, 32, 64]  # Reduced set to avoid context issues

# Block sizes to test
BLOCK_SIZES = [64, 128, 256]

# Matrix sizes to test
MATRIX_SIZES = [256, 512, 1024]

# Kernel types
KERNEL_NAMES = ["gemm", "reduction", "softmax"]

# Measurement settings
WARMUP_ITERS = 1
TIMING_REPEATS = 3

# Output path
OUTPUT_PATH = Path("results/tables/phase0_baseline.csv")


# Optional: collect achieved occupancy via Nsight Compute (ncu).
# This is slow (spawns one profiled subprocess per configuration) and may
# require Administrator privileges on Windows.
PHASE0_COLLECT_NCU = os.environ.get("PHASE0_COLLECT_NCU", "0") in ("1", "true", "True")
NCU_METRIC_ACHIEVED_OCC = "sm__warps_active.avg.pct_of_peak_sustained_active"
_NCU_PATH: str | None | bool = None  # None=unknown, False=disabled/unavailable, str=path


def _ensure_ncu_path() -> str | None:
    global _NCU_PATH

    if _NCU_PATH is False:
        return None
    if isinstance(_NCU_PATH, str):
        return _NCU_PATH

    res = ncu_metric_smoke_test(metric=NCU_METRIC_ACHIEVED_OCC)
    if not res.ok:
        _NCU_PATH = False
        if res.reason in ("permission_denied", "permission_denied_not_admin"):
            console.print("[yellow]ncu counters are not accessible (permission). Achieved occupancy will be skipped.[/yellow]")
            if res.reason == "permission_denied_not_admin":
                console.print("[yellow]Hint: run Command Prompt as Administrator to enable GPU counters.[/yellow]")
        return None

    _NCU_PATH = res.ncu_path or False
    return _NCU_PATH if isinstance(_NCU_PATH, str) else None


def _parse_ncu_csv_metric(stdout: str, metric_name: str) -> float | None:
    """Parse `ncu --csv` output and return the metric value as a float."""

    if not stdout:
        return None

    # Find where the CSV header begins (ncu sometimes prints preamble lines).
    start_idx = stdout.find("Metric Name")
    if start_idx == -1:
        return None

    csv_text = stdout[start_idx:]
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            name = (row.get("Metric Name") or row.get("MetricName") or "").strip()
            if name != metric_name:
                continue
            raw = (row.get("Metric Value") or row.get("MetricValue") or row.get("Value") or "").strip()
            if not raw:
                continue
            raw = raw.replace(",", "")
            if raw.endswith("%"):
                raw = raw[:-1]
            return float(raw)
    except Exception:
        return None

    return None


def _measure_achieved_occupancy_via_ncu(kernel_name: str, matrix_size: int, block_size: int, reg_cap: int) -> float | None:
    """Return achieved occupancy fraction [0,1] using ncu, or None if unavailable."""

    ncu_path = _ensure_ncu_path()
    if not ncu_path:
        return None

    # On Windows, PATH often resolves to ncu.BAT; prefer a sibling ncu.exe if present.
    p = Path(ncu_path)
    if os.name == "nt" and p.suffix.lower() in {".cmd", ".bat"}:
        exe_candidate = p.with_suffix(".exe")
        if exe_candidate.exists():
            ncu_path = str(exe_candidate)

    project_root = Path(__file__).parent.parent
    py_exe = sys.executable

    # Run a single configuration in a separate process that ncu can profile.
    # Keep this runner minimal: compile + 1 launch + stream sync.
    runner = f"""
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
    A, B, C, grid, block, kernel_fn = run_gemm(matrix_size, block_size, warmup=1, reg_cap=reg_cap)
    args = (A, B, C, np.int32(matrix_size))
elif kernel_name == 'reduction':
    x, out, grid, block, kernel_fn = run_reduction(matrix_size * matrix_size, block_size, warmup=1, reg_cap=reg_cap)
    args = (x, out, np.int32(matrix_size * matrix_size))
elif kernel_name == 'softmax':
    x, out, grid, block, kernel_fn = run_softmax(matrix_size, block_size, warmup=1, reg_cap=reg_cap)
    args = (x, out, np.int32(matrix_size), np.int32(matrix_size))
else:
    raise ValueError('Unknown kernel')

kernel_fn[grid, block](*args)
cuda.default_stream().synchronize()
""".lstrip()

    with tempfile.TemporaryDirectory(prefix="phase0_ncu_") as td:
        runner_path = Path(td) / "phase0_ncu_runner.py"
        runner_path.write_text(runner, encoding="utf-8")

        cmd = [
            ncu_path,
            "--metrics",
            NCU_METRIC_ACHIEVED_OCC,
            "--csv",
            "--quiet",
            py_exe,
            str(runner_path),
        ]

        try:
            # If ncu is a batch shim, run through cmd.exe parsing.
            if os.name == "nt" and str(ncu_path).lower().endswith((".cmd", ".bat")):
                cmdline = subprocess.list2cmdline(cmd)
                res = subprocess.run(
                    cmdline,
                    cwd=str(project_root),
                    capture_output=True,
                    text=True,
                    timeout=90,
                    shell=True,
                )
            else:
                res = subprocess.run(
                    cmd,
                    cwd=str(project_root),
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
        except Exception:
            return None

    val_pct = _parse_ncu_csv_metric(res.stdout or "", NCU_METRIC_ACHIEVED_OCC)
    if val_pct is None:
        return None
    return float(val_pct) / 100.0


def _threads_per_block(block) -> int:
    """Return total threads per block for 1D/2D/3D block tuples."""

    if isinstance(block, int):
        return int(block)
    total = 1
    for dim in block:
        total *= int(dim)
    return int(total)


def _extract_regs_per_thread(kernel_dispatcher) -> int | None:
    """Extract compiled registers-per-thread from a Numba CUDA dispatcher.

    After the first compilation, Numba stores a `_Kernel` object per signature in
    `dispatcher.overloads`. That `_Kernel` exposes `regs_per_thread`.
    """

    try:
        overloads = getattr(kernel_dispatcher, "overloads", None)
        if not overloads:
            return None
        kernel_obj = next(iter(overloads.values()))
        regs = getattr(kernel_obj, "regs_per_thread", None)
        if regs is None:
            return None
        return int(regs)
    except Exception:
        return None


def run_kernel_test(kernel_name: str,
                    matrix_size: int,
                    block_size: int,
                    reg_cap: int,
                    warmup: int = WARMUP_ITERS,
                    repeats: int = TIMING_REPEATS) -> dict:
    """
    Run a single kernel configuration and measure performance.

    Args:
        kernel_name: "gemm", "reduction", or "softmax"
        matrix_size: Input size (N for N×N matrices, or N² for vectors)
        block_size: Threads per block
        reg_cap: Register cap level (0 = no cap)
        warmup: Warmup iterations before timing
        repeats: Number of timing repetitions

    Returns:
        Dictionary with results
    """
    try:
        # Setup kernel
        if kernel_name == "gemm":
            if block_size == 64:
                # GEMM kernel_8 uses 8x8 tiles, so block must be 64
                A, B, C, grid, block, kernel_fn = run_gemm(matrix_size, block_size, warmup, reg_cap=reg_cap)
            else:
                # GEMM kernel_16 uses 16x16 tiles, block_size 128+ works
                A, B, C, grid, block, kernel_fn = run_gemm(matrix_size, block_size, warmup, reg_cap=reg_cap)
            args = (A, B, C, np.int32(matrix_size))
        elif kernel_name == "reduction":
            x, out, grid, block, kernel_fn = run_reduction(matrix_size * matrix_size, block_size, warmup, reg_cap=reg_cap)
            args = (x, out, np.int32(matrix_size * matrix_size))
        elif kernel_name == "softmax":
            x, out, grid, block, kernel_fn = run_softmax(matrix_size, block_size, warmup, reg_cap=reg_cap)
            args = (x, out, np.int32(matrix_size), np.int32(matrix_size))
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")

        # Time the kernel
        timing = time_kernel(kernel_fn, grid, block, args, warmup=0, repeats=repeats)

        # Register usage
        # - `estimated_regs`: heuristic model (kept for comparison)
        # - `actual_regs`: true regs/thread from the compiled kernel (PTXAS result)
        estimated_regs = OccupancyCalculator.estimate_register_count(kernel_name, block_size)
        actual_regs = _extract_regs_per_thread(kernel_fn)
        if actual_regs is None:
            # Fallback: if we cannot read PTXAS result, keep Phase 0 running.
            actual_regs = estimated_regs

        threads_per_block = _threads_per_block(block)

        # Compute theoretical occupancy
        theoret_occ = OccupancyCalculator.compute_occupancy(actual_regs, threads_per_block, shared_mem_bytes=0)

        result = {
            "kernel": kernel_name,
            "matrix_size": matrix_size,
            "block_size": block_size,
            "threads_per_block": threads_per_block,
            "reg_cap": reg_cap if reg_cap > 0 else "default",
            "est_regs": estimated_regs,
            "actual_regs": actual_regs,
            "theor_occ": theoret_occ,
            "time_ms_mean": timing.mean_ms,
            "time_ms_std": timing.std_ms,
            "time_ms_min": timing.min_ms,
            "time_ms_max": timing.max_ms,
        }

        if PHASE0_COLLECT_NCU:
            achieved = _measure_achieved_occupancy_via_ncu(kernel_name, matrix_size, block_size, reg_cap)
            result["achieved_occ"] = achieved

        # Best-effort cleanup of device buffers (avoid UnboundLocalError)
        if kernel_name == "gemm":
            del A, B, C
        else:
            del x, out
        del grid, block, kernel_fn, args

        return result

    except Exception as e:
        error_msg = str(e)
        # Only print brief errors to reduce noise
        if "CONTEXT_IS_DESTROYED" not in error_msg and "context" not in error_msg.lower():
            console.print(f"[red]Error: {kernel_name} size={matrix_size} block={block_size} regcap={reg_cap}: {e}[/red]")
        return None


def run_phase0():
    """Run the complete Phase 0 baseline experiment."""
    console.print("[cyan bold]Phase 0: Foundational Experiment[/cyan bold]")
    console.print("[cyan]GPU: RTX 3050 Ti (Ampere, SM 8.6)[/cyan]")
    console.print()

    # Preflight: sanity-check CUDA JIT with a conservative GEMM launch.
    try:
        _ = run_kernel_test("gemm", 256, 64, 0, warmup=1, repeats=1)
        if _ is None:
            raise RuntimeError("CUDA JIT preflight failed")
    except Exception as e:
        msg = str(e)
        if "libNVVM" in msg or "nvvm.dll" in msg or "NVVM" in msg:
            console.print("[bold red]Numba CUDA JIT cannot find NVVM (nvvm.dll).[/bold red]")
            console.print("[yellow]Fix: ensure system CUDA NVVM is available, then re-run Phase 0.[/yellow]")
            console.print("[yellow]Detected system NVVM is usually at:[/yellow]")
            console.print(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\nvvm\bin\x64\nvvm64_40_0.dll")
            console.print("[yellow]If this script still fails, set NUMBA_CUDA_NVVM and NUMBA_CUDA_LIBDEVICE to your CUDA paths.[/yellow]")
        else:
            console.print(f"[bold red]CUDA JIT preflight failed:[/bold red] {msg}")
        console.print("[yellow]Proceeding with full sweep; failing configs will be skipped.[/yellow]")

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Generate all configurations
    configs = list(product(KERNEL_NAMES, MATRIX_SIZES, BLOCK_SIZES, REG_CAPS))
    console.print(f"[cyan]Total configurations: {len(configs)}[/cyan]")
    console.print(f"[cyan]Estimated time: {len(configs) * TIMING_REPEATS * 0.01:.0f}s[/cyan]")
    console.print()

    results = []
    failed = 0

    for kernel, size, block, regcap in track(configs, description="Running Phase 0"):
        result = run_kernel_test(kernel, size, block, regcap)
        if result:
            results.append(result)
        else:
            failed += 1

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(OUTPUT_PATH, index=False)
    console.print(f"\n[bold green]Saved {len(results)} results to {OUTPUT_PATH}[/bold green]")
    if failed > 0:
        console.print(f"[yellow]⚠ {failed} configurations failed[/yellow]")

    # Print summary table
    print_summary_table(df)

    return df


def print_summary_table(df: pd.DataFrame):
    """Print a formatted summary table of key results."""
    console.print()
    console.print("[bold]Phase 0 Summary Table[/bold]")

    if df is None or df.empty or "kernel" not in df.columns:
        console.print("[bold red]No successful configurations were measured.[/bold red]")
        console.print("[yellow]This usually means Numba CUDA JIT could not compile (NVVM/libdevice missing).[/yellow]")
        console.print("[yellow]Fix NVVM first, then re-run: python experiments/phase0_baseline_table.py[/yellow]")
        return

    # Create a nicely formatted table
    table = RichTable(title="Register Cap vs Occupancy vs Runtime")
    table.add_column("Kernel", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Block", justify="right")
    table.add_column("RegCap", justify="right")
    table.add_column("Regs", justify="right")
    table.add_column("Occ %", justify="right", style="green")
    show_ach = "achieved_occ" in df.columns and df["achieved_occ"].notna().any()
    if show_ach:
        table.add_column("AchOcc %", justify="right", style="green")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Std", justify="right", style="dim")

    # Show key rows
    for _, row in df.iterrows():
        base_cells = [
            row["kernel"],
            str(row["matrix_size"]),
            str(row["block_size"]),
            str(row["reg_cap"]),
            str(row["actual_regs"]),
            f"{row['theor_occ']*100:.0f}%",
        ]
        if show_ach:
            v = row.get("achieved_occ")
            base_cells.append("" if pd.isna(v) else f"{float(v)*100:.0f}%")
        base_cells.extend([
            f"{row['time_ms_mean']:.2f}",
            f"+/-{row['time_ms_std']:.2f}",
        ])
        table.add_row(*base_cells)

    console.print(table)

    # Print key insights
    console.print()
    console.print("[bold yellow]Key Insights:[/bold yellow]")

    # Find best performing configs per kernel
    for kernel in df["kernel"].unique():
        kernel_df = df[df["kernel"] == kernel]
        best_idx = kernel_df["time_ms_mean"].idxmin()
        best_row = df.loc[best_idx]
        console.print(
            f"  {kernel.upper()}: Best time = {best_row['time_ms_mean']:.2f}ms at "
            f"block_size={best_row['block_size']}, reg_cap={best_row['reg_cap']}, "
            f"occupancy={best_row['theor_occ']*100:.0f}%"
        )

    # Show occupancy trends
    console.print()
    console.print("[bold yellow]Occupancy vs Register Cap Relationship:[/bold yellow]")
    for kernel in df["kernel"].unique():
        kernel_df = df[df["kernel"] == kernel].sort_values("actual_regs")
        sample = kernel_df.iloc[::max(1, len(kernel_df)//3)]  # Show ~3 samples
        for _, row in sample.iterrows():
            console.print(
                f"  {kernel}: {row['actual_regs']}regs -> {row['theor_occ']*100:.0f}% occupancy"
            )


if __name__ == "__main__":
    df = run_phase0()
    console.print()
    console.print("[bold green]Phase 0 complete![/bold green]")
    console.print(f"[cyan]Data saved to: {OUTPUT_PATH.absolute()}[/cyan]")
    console.print(f"[cyan]Rows: {len(df)}[/cyan]")
    console.print(f"[cyan]Columns: {', '.join(df.columns)}[/cyan]")
