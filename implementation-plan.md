# Adaptive ML-Driven GPU Compilation Optimization
## Implementation Plan — RTX 3050 Ti Prototype

> **Purpose:** Step-by-step implementation guide for building the prototype of the
> adaptive JIT optimization system. Pass this file to GitHub Copilot / Antigravity
> as context before starting any module. Every section has exact file paths,
> commands, code skeletons, and expected outputs.
>
> **Hardware target:** NVIDIA RTX 3050 Ti Laptop GPU (sm_86, 20 SMs, 2048 CUDA cores,
> 4 GB GDDR6, Ampere architecture)
>
> **Research goal:** Demonstrate that an RL agent conditioned on CUPTI hardware
> counter signals can adaptively select `--maxrregcount` (PTXAS register cap),
> block size, and shared memory allocation, outperforming PTXAS static defaults
> on GEMM, reduction, and convolution kernels.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Project Structure](#2-project-structure)
3. [Phase 0 — Foundational Experiment (The Baseline Table)](#3-phase-0--foundational-experiment)
4. [Phase 1 — CUPTI Instrumentation Layer](#4-phase-1--cupti-instrumentation-layer)
5. [Phase 2 — Benchmark Kernels](#5-phase-2--benchmark-kernels)
6. [Phase 3 — RL Environment (Gym Interface)](#6-phase-3--rl-environment)
7. [Phase 4 — PPO Agent Training](#7-phase-4--ppo-agent-training)
8. [Phase 5 — BiLSTM Phase Detector](#8-phase-5--bilstm-phase-detector)
9. [Phase 6 — GNN IR Encoder (Optional DL Upgrade)](#9-phase-6--gnn-ir-encoder)
10. [Phase 7 — Evaluation & Results Tables](#10-phase-7--evaluation--results-tables)
11. [Copilot Prompting Guide](#11-copilot-prompting-guide)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Environment Setup

### 1.1 System Requirements Check

Run these first to confirm your environment before touching any code:

```bash
# Check CUDA version
nvcc --version

# Check GPU and driver
nvidia-smi

# Check compute capability (should show 8.6 for RTX 3050 Ti)
python3 -c "import torch; print(torch.cuda.get_device_capability())"

# Check if CUPTI is available
ls /usr/local/cuda/extras/CUPTI/lib64/
# Expected: libcupti.so, libcupti.so.12 (or similar)

# Check Nsight Compute CLI is available (needed for counter collection)
ncu --version
```

**Expected output for RTX 3050 Ti:**
```
CUDA Version: 12.x
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
Compute Capability: (8, 6)   ← sm_86
```

---

### 1.2 Conda Environment

```bash
# Create isolated environment
conda create -n gpu-jit-opt python=3.10 -y
conda activate gpu-jit-opt

# Core deep learning stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Numba with CUDA support
conda install numba cudatoolkit=12.1 -c conda-forge -y

# RL library (Stable-Baselines3 — do NOT implement PPO from scratch)
pip install stable-baselines3[extra]==2.3.2
pip install gymnasium==0.29.1

# GNN library (for Phase 6)
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Profiling and instrumentation
pip install pynvml           # Python bindings for NVML (occupancy, power, etc.)
pip install cuda-python      # NVIDIA's official Python CUDA bindings (includes CUPTI)
pip install nvtx             # NVTX markers for Nsight tracing

# Utilities
pip install numpy pandas matplotlib seaborn tqdm rich
pip install jupyter ipykernel
pip install pytest black isort

# Verify everything
python3 -c "
import torch, numba, stable_baselines3, gymnasium, torch_geometric
print('PyTorch CUDA:', torch.cuda.is_available())
print('Numba CUDA:', numba.cuda.is_available())
print('All imports OK')
"
```

---

### 1.3 CUPTI Access (Important — read this carefully)

CUPTI requires either:
- Root access, OR
- Setting `paranoid_level` to allow user-space profiling

```bash
# Check current paranoid level
cat /proc/sys/kernel/perf_event_paranoid

# If it returns 4 or above, run this (survives until reboot):
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'

# To make it permanent:
echo 'kernel.perf_event_paranoid=1' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Test CUPTI access using Nsight Compute (most reliable method)
# This collects a real metric — if this works, all CUPTI collection will work
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    python3 -c "
import numba.cuda as cuda
import numpy as np

@cuda.jit
def add(a, b, c):
    i = cuda.grid(1)
    if i < a.shape[0]:
        c[i] = a[i] + b[i]

n = 1024
a = cuda.to_device(np.ones(n, dtype=np.float32))
b = cuda.to_device(np.ones(n, dtype=np.float32))
c = cuda.device_array(n, dtype=np.float32)
add[32, 32](a, b, c)
"
```

---

## 2. Project Structure

Create this exact directory layout before writing any code:

```
gpu-jit-opt/
│
├── README.md
├── implementation-plan.md          ← this file
├── requirements.txt
├── pyproject.toml
│
├── kernels/                        ← CUDA kernel definitions
│   ├── __init__.py
│   ├── gemm.py                     ← Matrix multiplication (Numba CUDA)
│   ├── reduction.py                ← Parallel reduction
│   ├── conv2d.py                   ← 2D convolution
│   └── softmax.py                  ← Row-wise softmax
│
├── profiling/                      ← Hardware counter collection
│   ├── __init__.py
│   ├── cupti_collector.py          ← CUPTI counter collection via ncu subprocess
│   ├── nvml_monitor.py             ← Real-time GPU state via pynvml
│   ├── cuda_timer.py               ← Precise kernel timing via CUDA events
│   └── metrics.py                  ← Metric definitions and normalization
│
├── compiler/                       ← Compilation control layer
│   ├── __init__.py
│   ├── ptxas_controller.py         ← --maxrregcount and compilation flag control
│   ├── numba_compiler.py           ← Numba JIT compilation with configurable params
│   └── ir_extractor.py             ← Extract LLVM IR / PTX from compiled kernels
│
├── environment/                    ← RL Gymnasium environment
│   ├── __init__.py
│   ├── kernel_env.py               ← Main Gym environment
│   ├── action_space.py             ← Action space definitions
│   ├── state_space.py              ← State space + normalization
│   └── reward.py                   ← Reward function
│
├── models/                         ← ML model definitions
│   ├── __init__.py
│   ├── phase_detector.py           ← BiLSTM phase detector
│   ├── gnn_encoder.py              ← GNN over IR graph (Phase 6)
│   └── policy.py                   ← Custom policy network for SB3
│
├── training/                       ← Training scripts
│   ├── train_rl.py                 ← PPO training entry point
│   ├── train_phase_detector.py     ← BiLSTM training
│   └── config.py                   ← All hyperparameters in one place
│
├── experiments/                    ← Experiment runners
│   ├── phase0_baseline_table.py    ← THE FOUNDATIONAL EXPERIMENT (run first)
│   ├── phase1_cupti_validation.py  ← Validate counter collection
│   ├── phase2_rl_baseline.py       ← RL vs random search vs default
│   └── phase3_phase_detection.py   ← Phase detector evaluation
│
├── results/                        ← Auto-generated (gitignored)
│   ├── tables/
│   ├── plots/
│   └── checkpoints/
│
└── tests/
    ├── test_kernels.py
    ├── test_cupti.py
    └── test_environment.py
```

```bash
# Create the full structure
mkdir -p gpu-jit-opt/{kernels,profiling,compiler,environment,models,training,experiments,results/{tables,plots,checkpoints},tests}
cd gpu-jit-opt
touch {README.md,requirements.txt,pyproject.toml}
touch kernels/{__init__,gemm,reduction,conv2d,softmax}.py
touch profiling/{__init__,cupti_collector,nvml_monitor,cuda_timer,metrics}.py
touch compiler/{__init__,ptxas_controller,numba_compiler,ir_extractor}.py
touch environment/{__init__,kernel_env,action_space,state_space,reward}.py
touch models/{__init__,phase_detector,gnn_encoder,policy}.py
touch training/{train_rl,train_phase_detector,config}.py
touch experiments/{phase0_baseline_table,phase1_cupti_validation,phase2_rl_baseline,phase3_phase_detection}.py
touch tests/{test_kernels,test_cupti,test_environment}.py
```

---

## 3. Phase 0 — Foundational Experiment

> **This is the first thing you implement and run. produces
> the core table that validates your entire research hypothesis.**
>
> **Goal:** Show that `--maxrregcount` significantly affects warp occupancy and
> kernel runtime, and that the optimal value differs between compute-bound and
> memory-bound kernels. This is the table that has never been published.

### 3.1 `experiments/phase0_baseline_table.py`

```python
"""
Phase 0: Foundational Experiment
=================================
Vary --maxrregcount × kernel_type × matrix_size and measure:
  - Theoretical occupancy (computed from register count)
  - Achieved occupancy (measured via ncu CUPTI counter)
  - Kernel runtime (ms, CUDA event timing)

Expected to produce a table like:

kernel      | size  | regcap | regs/thread | theor_occ | achiev_occ | time_ms
------------|-------|--------|-------------|-----------|------------|--------
gemm        | 512   | 32     | 32          | 100%      | 78%        | 4.2
gemm        | 512   | 64     | 61          | 68%       | 62%        | 3.8
gemm        | 512   | 128    | 98          | 41%       | 39%        | 5.1
gemm        | 2048  | 32     | 32          | 100%      | 71%        | 89.3
...

Run with: python experiments/phase0_baseline_table.py
Results saved to: results/tables/phase0_baseline.csv
"""

import subprocess
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from rich.console import Console
from rich.table import Table as RichTable
from rich.progress import track

console = Console()

# ── Configuration ────────────────────────────────────────────────────
REG_CAPS    = [16, 24, 32, 40, 48, 64, 80, 96, 128, 0]  # 0 = no cap (PTXAS default)
BLOCK_SIZES = [64, 128, 256, 512]
MATRIX_SIZES = [256, 512, 1024, 2048]
KERNEL_NAMES = ["gemm", "reduction", "softmax"]
REPEATS = 5   # Average over N runs for stable timing
OUTPUT_PATH = Path("results/tables/phase0_baseline.csv")

# ── CUPTI counter names for ncu ──────────────────────────────────────
# These are verified counter names for sm_86 (Ampere)
COUNTERS = [
    "sm__warps_active.avg.pct_of_peak_sustained_active",   # achieved occupancy
    "sm__maximum_warps_per_active_cycle_pct",              # theoretical occupancy
    "dram__bytes_read.sum",                                 # memory reads
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",   # FP add ops (compute proxy)
    "l2_global_hit_rate",                                  # L2 cache hit rate
]


def compile_and_run_kernel(kernel_name: str,
                            matrix_size: int,
                            block_size: int,
                            reg_cap: int,
                            repeats: int = 5) -> dict:
    """
    Compile a kernel with specific --maxrregcount and measure performance.

    Strategy: We use a subprocess call to run a small Python script that
    compiles and executes the kernel. This isolates the compilation per
    configuration. ncu wraps the subprocess to collect CUPTI counters.
    """
    # Build the inner script that will be profiled
    runner_script = f"""
import sys
sys.path.insert(0, '.')
import numpy as np
import numba.cuda as cuda
import time

# Import the kernel factory
from kernels.gemm import make_gemm_kernel
from kernels.reduction import make_reduction_kernel
from kernels.softmax import make_softmax_kernel

kernel_factories = {{
    'gemm': make_gemm_kernel,
    'reduction': make_reduction_kernel,
    'softmax': make_softmax_kernel,
}}

# Create kernel with specified block size and reg cap
kernel_fn = kernel_factories['{kernel_name}'](
    block_size={block_size},
    reg_cap={reg_cap if reg_cap > 0 else 'None'}
)

# Setup inputs
n = {matrix_size}
if '{kernel_name}' == 'gemm':
    A = cuda.to_device(np.random.randn(n, n).astype(np.float32))
    B = cuda.to_device(np.random.randn(n, n).astype(np.float32))
    C = cuda.device_array((n, n), dtype=np.float32)
    grid = ((n + {block_size} - 1) // {block_size},
            (n + {block_size} - 1) // {block_size})
    block = ({block_size}, 1, 1)
    kernel_fn[grid, block](A, B, C, n)
elif '{kernel_name}' == 'reduction':
    x = cuda.to_device(np.random.randn(n * n).astype(np.float32))
    out = cuda.device_array(1, dtype=np.float32)
    threads = {block_size}
    blocks = (n * n + threads - 1) // threads
    kernel_fn[blocks, threads](x, out)
elif '{kernel_name}' == 'softmax':
    x = cuda.to_device(np.random.randn(n, n).astype(np.float32))
    out = cuda.device_array((n, n), dtype=np.float32)
    kernel_fn[n, {block_size}](x, out, n)

cuda.synchronize()
"""

    # Write runner to temp file
    runner_path = Path("/tmp/kernel_runner.py")
    runner_path.write_text(runner_script)

    # Build ncu command
    counter_str = ",".join(COUNTERS)
    ncu_cmd = [
        "ncu",
        "--metrics", counter_str,
        "--csv",
        "--quiet",
        "python3", str(runner_path)
    ]

    try:
        result = subprocess.run(
            ncu_cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        counters_out = parse_ncu_csv(result.stdout)
    except subprocess.TimeoutExpired:
        console.print(f"[red]Timeout: {kernel_name} size={matrix_size} regcap={reg_cap}[/red]")
        counters_out = {}
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        counters_out = {}

    # Timing (separate from ncu to avoid profiling overhead)
    timing_ms = measure_kernel_time(kernel_name, matrix_size, block_size, reg_cap, repeats)

    return {
        "kernel": kernel_name,
        "matrix_size": matrix_size,
        "block_size": block_size,
        "reg_cap": reg_cap if reg_cap > 0 else "default",
        "time_ms_mean": np.mean(timing_ms),
        "time_ms_std": np.std(timing_ms),
        **counters_out
    }


def parse_ncu_csv(csv_output: str) -> dict:
    """Parse ncu --csv output and extract relevant metrics."""
    import csv
    import io
    metrics = {}
    try:
        reader = csv.DictReader(io.StringIO(csv_output))
        for row in reader:
            metric_name = row.get("Metric Name", "").strip()
            metric_value = row.get("Metric Value", "0").strip()
            if metric_name in COUNTERS:
                try:
                    metrics[metric_name.split(".")[-1]] = float(
                        metric_value.replace(",", "").replace("%", "")
                    )
                except ValueError:
                    metrics[metric_name.split(".")[-1]] = 0.0
    except Exception:
        pass
    return metrics


def measure_kernel_time(kernel_name, matrix_size, block_size, reg_cap, repeats) -> list:
    """Measure kernel execution time using CUDA events (no profiling overhead)."""
    import numba.cuda as cuda
    import numpy as np
    # This is a simplified version — actual kernel calls go here
    # Returns list of timing measurements in milliseconds
    # TODO: Import actual kernel from kernels/ and time it here
    times = []
    for _ in range(repeats):
        start = cuda.event()
        end = cuda.event()
        start.record()
        # kernel_fn[grid, block](args...)
        end.record()
        end.synchronize()
        times.append(cuda.event_elapsed_time(start, end))
    return times if times else [0.0]


def run_phase0():
    """Run the full baseline experiment grid."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []

    configs = list(product(KERNEL_NAMES, MATRIX_SIZES, BLOCK_SIZES, REG_CAPS))
    console.print(f"[cyan]Running {len(configs)} configurations...[/cyan]")

    for kernel, size, block, regcap in track(configs, description="Sweeping configs"):
        row = compile_and_run_kernel(kernel, size, block, regcap)
        results.append(row)
        console.print(
            f"[green]{kernel}[/green] size={size} block={block} "
            f"regcap={regcap} → {row.get('time_ms_mean', '?'):.2f}ms"
        )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)
    console.print(f"\n[bold green]Saved to {OUTPUT_PATH}[/bold green]")

    # Print summary table
    print_summary_table(df)
    return df


def print_summary_table(df: pd.DataFrame):
    """Print a rich formatted summary of the results."""
    table = RichTable(title="Phase 0: Register Cap vs Occupancy vs Runtime")
    table.add_column("Kernel", style="cyan")
    table.add_column("Size")
    table.add_column("Block")
    table.add_column("RegCap")
    table.add_column("Occ %", justify="right")
    table.add_column("Time (ms)", justify="right")

    for _, row in df.iterrows():
        table.add_row(
            str(row["kernel"]),
            str(row["matrix_size"]),
            str(row["block_size"]),
            str(row["reg_cap"]),
            f"{row.get('avg', 0):.1f}%",
            f"{row.get('time_ms_mean', 0):.2f} ± {row.get('time_ms_std', 0):.2f}"
        )

    console.print(table)


if __name__ == "__main__":
    run_phase0()
```

---

## 4. Phase 1 — CUPTI Instrumentation Layer

### Phase 1 goals (what we must achieve)

Phase 1 exists to make GPU execution **measurable** and **RL-ready**.

Goals:
- Collect a small, stable set of GPU performance counters per kernel run (CUPTI-derived) using a **reliable Windows-friendly path**.
- Provide a single Python API that returns a **numeric state vector** (raw + normalized), suitable for an RL environment.
- Detect and handle common failure modes gracefully (especially Windows WDDM counter permission errors like `ERR_NVGPUCTRPERM`).

Definition of Done (Phase 1 is complete when):
- We can run a smoke test that confirms counters are accessible (or returns a clear reason + fix instructions).
- We can collect at least these metrics for a single kernel launch via Nsight Compute (`ncu`) when available:
    - achieved occupancy
    - DRAM throughput (% of peak)
    - L2 hit-rate
    - SM active (% of peak)
    - (optional) warp execution efficiency
- The collector API:
    - works on Windows without requiring manual shell quoting hacks,
    - returns `None` / skips cleanly if `ncu` is missing or counters are blocked,
    - never crashes the whole experiment suite if profiling is unavailable.

Primary deliverables:
- `profiling/ncu_utils.py`: smoke test + permission diagnostics
- `profiling/cupti_collector.py`: metric collection wrapper around `ncu --csv`
- Unit smoke test in `tests/` that imports the collector and (optionally) runs the smoke test

### 4.1 `profiling/cuda_timer.py`

```python
"""
Precise CUDA kernel timing using CUDA events.
CUDA events are synchronized with the GPU command stream,
giving accurate per-kernel timing without CPU-side overhead.
"""

import numba.cuda as cuda
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List


@dataclass
class TimingResult:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    all_ms: List[float]


@contextmanager
def cuda_timer():
    """Context manager for single kernel timing."""
    start = cuda.event()
    end = cuda.event()
    start.record()
    yield
    end.record()
    end.synchronize()
    elapsed = cuda.event_elapsed_time(start, end)
    return elapsed


def time_kernel(kernel_fn, grid, block, args: tuple, warmup: int = 3, repeats: int = 10) -> TimingResult:
    """
    Time a CUDA kernel with warmup iterations.

    Args:
        kernel_fn: Compiled Numba CUDA kernel (result of @cuda.jit)
        grid: Grid dimensions tuple e.g. (16, 16) or integer
        block: Block dimensions tuple e.g. (16, 16) or integer
        args: Kernel arguments tuple
        warmup: Number of warmup iterations (not measured)
        repeats: Number of measured iterations

    Returns:
        TimingResult with statistics in milliseconds
    """
    # Warmup — JIT compilation happens here on first call
    for _ in range(warmup):
        kernel_fn[grid, block](*args)
    cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeats):
        start_evt = cuda.event()
        end_evt = cuda.event()
        start_evt.record()
        kernel_fn[grid, block](*args)
        end_evt.record()
        end_evt.synchronize()
        times.append(cuda.event_elapsed_time(start_evt, end_evt))

    times_arr = np.array(times)
    return TimingResult(
        mean_ms=float(np.mean(times_arr)),
        std_ms=float(np.std(times_arr)),
        min_ms=float(np.min(times_arr)),
        max_ms=float(np.max(times_arr)),
        all_ms=times
    )
```

---

### 4.2 `profiling/cupti_collector.py`

```python
"""
CUPTI hardware counter collection via Nsight Compute CLI (ncu).

Strategy: Use ncu as a subprocess to collect hardware counters.
This is the most reliable approach for RTX 3050 Ti without
requiring raw CUPTI API calls (which require root and kernel modules).

The collected counters form the state vector for the RL agent:
    x_t = [occupancy, l2_hit_rate, dram_bw_pct, warp_eff, sm_active_pct]

These are the GPU-side equivalents of the CPU PMU vector from the JIT paper.
"""

import subprocess
import csv
import io
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np


# ── Verified counter names for sm_86 (Ampere, RTX 3050 Ti) ──────────
# Source: Nsight Compute Kernel Profiling Guide for Ampere
# https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
CUPTI_COUNTERS = {
    # Occupancy — most important for register allocation decisions
    "achieved_occupancy":    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occupancy": "sm__maximum_warps_per_active_cycle_pct",

    # Memory hierarchy
    "l2_hit_rate":           "lts__t_sector_hit_rate.pct",
    "l1_hit_rate":           "l1tex__t_sector_hit_rate.pct",
    "dram_bw_pct":           "dram__throughput.avg.pct_of_peak_sustained_elapsed",

    # Compute utilization
    "sm_active_pct":         "sm__active_cycles.avg.pct_of_peak_sustained_elapsed",

    # Warp efficiency (penalizes divergence)
    "warp_exec_efficiency":  "smsp__thread_inst_executed_per_inst_executed.ratio",

    # Stall reasons (for phase classification)
    "stall_long_sb":         "smsp__average_warp_latency_per_inst_issued.ratio",
}


class CUPTICollector:
    """
    Collects GPU hardware counters for a given kernel execution.

    Usage:
        collector = CUPTICollector()
        state = collector.collect(script_path="/tmp/runner.py")
        # state is a normalized numpy array of shape (5,)
    """

    def __init__(self, ncu_path: str = "ncu"):
        self.ncu_path = ncu_path
        self._verify_ncu()

    def _verify_ncu(self):
        """Check ncu is available and has GPU access."""
        result = subprocess.run(
            [self.ncu_path, "--version"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Nsight Compute (ncu) not found. "
                "Install CUDA toolkit or add to PATH. "
                "On Ubuntu: sudo apt install nsight-compute"
            )

    def collect(self, script_path: str, timeout: int = 120) -> np.ndarray:
        """
        Run a Python script under ncu and collect hardware counters.

        Args:
            script_path: Path to the Python script that runs the kernel
            timeout: Maximum seconds to wait

        Returns:
            Normalized state vector of shape (5,):
            [achieved_occ, l2_hit_rate, dram_bw_pct, warp_eff, sm_active_pct]
            All values in [0, 1]
        """
        counter_str = ",".join(CUPTI_COUNTERS.values())

        cmd = [
            self.ncu_path,
            "--metrics", counter_str,
            "--csv",
            "--quiet",
            "--target-processes", "all",
            "python3", script_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            raw = self._parse_csv(result.stdout)
        except subprocess.TimeoutExpired:
            print(f"[CUPTI] Timeout collecting from {script_path}")
            return self._zero_state()
        except Exception as e:
            print(f"[CUPTI] Error: {e}")
            return self._zero_state()

        return self._to_normalized_vector(raw)

    def _parse_csv(self, csv_str: str) -> Dict[str, float]:
        """Parse ncu CSV output into a dict of metric_name → value."""
        metrics = {}
        # ncu CSV format has many columns; we extract Metric Name and Metric Value
        reader = csv.DictReader(io.StringIO(csv_str))
        for row in reader:
            name = row.get("Metric Name", "").strip()
            value_str = row.get("Metric Value", "0").strip()
            # Reverse lookup: find our short name for this counter
            for short_name, full_name in CUPTI_COUNTERS.items():
                if full_name == name:
                    try:
                        # Remove % signs, commas
                        clean = value_str.replace(",", "").replace("%", "").strip()
                        metrics[short_name] = float(clean)
                    except ValueError:
                        metrics[short_name] = 0.0
        return metrics

    def _to_normalized_vector(self, raw: Dict[str, float]) -> np.ndarray:
        """
        Convert raw counter dict to normalized state vector.
        All values divided by 100 since counters are in percentage (0-100%).
        """
        vec = np.array([
            raw.get("achieved_occupancy",  0.0) / 100.0,
            raw.get("l2_hit_rate",         0.0) / 100.0,
            raw.get("dram_bw_pct",         0.0) / 100.0,
            raw.get("warp_exec_efficiency", 0.0) / 100.0,
            raw.get("sm_active_pct",       0.0) / 100.0,
        ], dtype=np.float32)
        return np.clip(vec, 0.0, 1.0)

    def _zero_state(self) -> np.ndarray:
        """Return zero state vector on failure."""
        return np.zeros(5, dtype=np.float32)
```

---

### 4.3 `profiling/nvml_monitor.py`

```python
"""
Real-time GPU monitoring via pynvml (NVIDIA Management Library).
Used for lightweight state collection BETWEEN kernel runs.
Does not require profiling privileges unlike CUPTI.
"""

import pynvml
import numpy as np
from dataclasses import dataclass


@dataclass
class GPUState:
    gpu_util_pct: float       # % of time SM was executing at least one warp
    mem_util_pct: float       # % of time memory interface was active
    mem_used_mb: float        # Used VRAM in MB
    mem_total_mb: float       # Total VRAM in MB
    temperature_c: float      # GPU temperature
    power_w: float            # Current power draw in Watts
    clock_mhz: float          # Current graphics clock in MHz


class NVMLMonitor:
    """Lightweight real-time GPU monitor using NVML."""

    def __init__(self, device_index: int = 0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(self.handle)
        print(f"[NVML] Monitoring: {name}")

    def get_state(self) -> GPUState:
        """Sample current GPU state. Fast (<1ms)."""
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW → W
        except pynvml.NVMLError:
            power = 0.0
        clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)

        return GPUState(
            gpu_util_pct=float(util.gpu),
            mem_util_pct=float(util.memory),
            mem_used_mb=mem.used / 1024**2,
            mem_total_mb=mem.total / 1024**2,
            temperature_c=float(temp),
            power_w=float(power),
            clock_mhz=float(clock),
        )

    def to_vector(self) -> np.ndarray:
        """Return normalized state as numpy vector for RL state space."""
        state = self.get_state()
        return np.array([
            state.gpu_util_pct / 100.0,
            state.mem_util_pct / 100.0,
            state.mem_used_mb / state.mem_total_mb,
            state.temperature_c / 100.0,   # Normalize to ~[0,1]
        ], dtype=np.float32)

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
```

---

## 5. Phase 2 — Benchmark Kernels

### 5.1 `kernels/gemm.py`

```python
"""
Tiled Matrix Multiplication (GEMM) kernel.

Parameterized by:
  - block_size: threads per block (square tile dimension)
  - reg_cap: maximum registers per thread hint (--maxrregcount equivalent)

Note on --maxrregcount with Numba:
  Numba does not directly expose --maxrregcount. We control it by
  setting the CUDA_LAUNCH_BLOCKING environment variable and using
  numba.cuda.compiler options. For the prototype, we use ptxas
  flags via NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM and
  compile_ptx() with custom flags.

  The practical control method for Numba:
    numba.cuda.compiler.compile_ptx(func, sig, cc=(8,6),
                                    device=True,
                                    fastmath=False)
  then compile PTX with: ptxas --maxrregcount=N input.ptx -o output.cubin
"""

import numba.cuda as cuda
import numba as nb
import numpy as np
from numba import float32


# ── Fixed tile size for the Numba kernel ────────────────────────────
# We parameterize block_size externally but tile size must be a compile-time constant
TILE_SIZE = 16  # Change to 32 for larger block experiments


@cuda.jit
def gemm_kernel_16(A, B, C, N):
    """Tiled GEMM with 16x16 tile. Best for block_size=256 (16x16 threads)."""
    TILE = 16
    # Shared memory tiles
    sA = cuda.shared.array(shape=(TILE, TILE), dtype=float32)
    sB = cuda.shared.array(shape=(TILE, TILE), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx

    acc = float32(0.0)

    for k in range((N + TILE - 1) // TILE):
        # Load tile from A
        if row < N and k * TILE + tx < N:
            sA[ty, tx] = A[row, k * TILE + tx]
        else:
            sA[ty, tx] = float32(0.0)

        # Load tile from B
        if k * TILE + ty < N and col < N:
            sB[ty, tx] = B[k * TILE + ty, col]
        else:
            sB[ty, tx] = float32(0.0)

        cuda.syncthreads()

        # Compute partial dot product
        for i in range(TILE):
            acc += sA[ty, i] * sB[i, tx]

        cuda.syncthreads()

    if row < N and col < N:
        C[row, col] = acc


def make_gemm_kernel(block_size: int = 256, reg_cap: int = None):
    """
    Factory function that returns a kernel callable configured
    for the given block_size.

    For the prototype, reg_cap is noted but the actual PTXAS
    --maxrregcount control is done via the ptxas_controller module.
    """
    # Numba requires TILE to match threadIdx dimensions
    # For prototype: always use TILE=16, block=(16,16)
    return gemm_kernel_16


def run_gemm(N: int, block_size: int = 256, warmup: int = 3) -> tuple:
    """
    Run GEMM and return (A_dev, B_dev, C_dev, grid, block).
    Caller handles timing.
    """
    tile = 16  # Must match kernel
    A = cuda.to_device(np.random.randn(N, N).astype(np.float32))
    B = cuda.to_device(np.random.randn(N, N).astype(np.float32))
    C = cuda.device_array((N, N), dtype=np.float32)

    grid  = ((N + tile - 1) // tile, (N + tile - 1) // tile)
    block = (tile, tile)

    # Warmup (triggers JIT compilation)
    for _ in range(warmup):
        gemm_kernel_16[grid, block](A, B, C, N)
    cuda.synchronize()

    return A, B, C, grid, block
```

---

### 5.2 `kernels/reduction.py`

```python
"""
Parallel Reduction kernel (sum).
Classic memory-bound benchmark — good contrast to GEMM.
"""

import numba.cuda as cuda
import numba as nb
from numba import float32
import numpy as np


@cuda.jit
def reduction_kernel(input_arr, output, n):
    """
    Two-pass parallel reduction using shared memory.
    Each block reduces its chunk and writes partial sum to output.
    Second pass reduces the partial sums (done on CPU for prototype).
    """
    BLOCK = cuda.blockDim.x
    tid   = cuda.threadIdx.x
    bid   = cuda.blockIdx.x
    gid   = bid * BLOCK + tid

    # Shared memory — size must match block size
    # Using 1024 as max — actual usage depends on block_size at launch
    smem = cuda.shared.array(1024, dtype=float32)

    # Load from global memory
    smem[tid] = input_arr[gid] if gid < n else float32(0.0)
    cuda.syncthreads()

    # Reduction in shared memory (tree reduction)
    s = BLOCK // 2
    while s > 0:
        if tid < s:
            smem[tid] += smem[tid + s]
        cuda.syncthreads()
        s //= 2

    # Write block result
    if tid == 0:
        cuda.atomic.add(output, 0, smem[0])


def run_reduction(N: int, block_size: int = 256, warmup: int = 3) -> tuple:
    """Setup and return reduction inputs."""
    x = cuda.to_device(np.random.randn(N).astype(np.float32))
    out = cuda.device_array(1, dtype=np.float32)
    out[0] = 0.0

    blocks = (N + block_size - 1) // block_size
    grid   = (blocks,)
    block  = (block_size,)

    for _ in range(warmup):
        reduction_kernel[grid, block](x, out, N)
    cuda.synchronize()

    return x, out, grid, block
```

---

### 5.3 `compiler/ptxas_controller.py`

```python
"""
Control PTXAS compilation parameters for Numba kernels.

Key challenge: Numba does not natively expose --maxrregcount.

Approach for prototype:
  1. Compile kernel to PTX using numba.cuda.compiler
  2. Write PTX to temp file
  3. Call ptxas directly with --maxrregcount=N
  4. Load the compiled cubin back via CUDA driver API

This gives us direct PTXAS parameter control.
"""

import subprocess
import tempfile
import os
from pathlib import Path
import numpy as np
import numba
import numba.cuda as cuda
from numba.cuda import compiler as cuda_compiler


class PTXASController:
    """
    Controls PTXAS compilation flags for Numba kernels.

    Main capability: compile a Numba kernel to PTX, then
    recompile that PTX to CUBIN with custom --maxrregcount.
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.cc = cuda.get_current_device().compute_capability
        self.sm = f"sm_{self.cc[0]}{self.cc[1]}"
        print(f"[PTXAS] Target: {self.sm} (compute capability {self.cc})")
        self._verify_ptxas()

    def _verify_ptxas(self):
        """Check ptxas is available."""
        result = subprocess.run(["ptxas", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("ptxas not found. Make sure CUDA toolkit bin is in PATH.")
        print(f"[PTXAS] {result.stdout.strip().splitlines()[0]}")

    def get_register_usage(self, ptx_path: str, reg_cap: int = 0) -> dict:
        """
        Compile PTX with ptxas and extract register usage from verbose output.

        Args:
            ptx_path: Path to PTX file
            reg_cap: --maxrregcount value (0 = no cap)

        Returns:
            dict with 'registers_per_thread', 'shared_mem_bytes', 'local_mem_bytes'
        """
        cmd = [
            "ptxas",
            f"--gpu-name={self.sm}",
            "--verbose",  # This gives us register counts
            ptx_path,
            "-o", "/dev/null"  # Discard output, just want the verbose info
        ]
        if reg_cap > 0:
            cmd.extend(["--maxrregcount", str(reg_cap)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse output: look for lines like:
        # ptxas info    : Used 32 registers, 1024 bytes smem, 0 bytes lmem
        info = {
            "registers_per_thread": 0,
            "shared_mem_bytes": 0,
            "local_mem_bytes": 0,
            "reg_cap": reg_cap
        }

        for line in (result.stderr + result.stdout).splitlines():
            if "Used" in line and "registers" in line:
                parts = line.split()
                try:
                    reg_idx = parts.index("registers,") - 1
                    info["registers_per_thread"] = int(parts[reg_idx])
                except (ValueError, IndexError):
                    pass
                try:
                    smem_idx = parts.index("bytes") - 1
                    info["shared_mem_bytes"] = int(parts[smem_idx])
                except (ValueError, IndexError):
                    pass

        return info

    def compile_with_regcap(self, ptx_path: str, reg_cap: int, output_path: str) -> bool:
        """
        Compile PTX to CUBIN with --maxrregcount constraint.

        Returns True if compilation succeeded.
        """
        cmd = [
            "ptxas",
            f"--gpu-name={self.sm}",
            ptx_path,
            "-o", output_path,
        ]
        if reg_cap > 0:
            cmd.extend(["--maxrregcount", str(reg_cap)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[PTXAS] Compilation failed:\n{result.stderr}")
            return False
        return True

    def compute_theoretical_occupancy(self, registers_per_thread: int,
                                       block_size: int,
                                       shared_mem_bytes: int = 0) -> float:
        """
        Compute theoretical occupancy for sm_86 (RTX 3050 Ti Ampere).

        Hardware specs for sm_86:
          - Max warps per SM: 48
          - Max threads per SM: 1536
          - Max registers per SM: 65536
          - Max shared memory per SM: 100 KB (default), up to 164 KB (opt-in)
          - Max blocks per SM: 16
          - Max threads per block: 1024

        Formula: occupancy = min(
            max_warps,
            floor(max_regs_per_sm / (registers_per_thread * 32)),   # register limit
            floor(max_threads_per_sm / block_size) * (block_size/32)  # thread limit
        ) / max_warps

        Source: CUDA Occupancy Calculator
        """
        # sm_86 hardware limits
        MAX_WARPS_PER_SM      = 48
        MAX_REGISTERS_PER_SM  = 65536
        MAX_THREADS_PER_SM    = 1536
        MAX_BLOCKS_PER_SM     = 16
        WARP_SIZE             = 32

        warps_per_block = (block_size + WARP_SIZE - 1) // WARP_SIZE

        # Limit 1: Register constraint
        if registers_per_thread > 0:
            warps_reg_limit = MAX_REGISTERS_PER_SM // (registers_per_thread * WARP_SIZE)
        else:
            warps_reg_limit = MAX_WARPS_PER_SM

        # Limit 2: Thread constraint
        blocks_thread_limit = MAX_THREADS_PER_SM // block_size
        warps_thread_limit  = blocks_thread_limit * warps_per_block

        # Limit 3: Block count constraint
        warps_block_limit = MAX_BLOCKS_PER_SM * warps_per_block

        active_warps = min(warps_reg_limit, warps_thread_limit,
                          warps_block_limit, MAX_WARPS_PER_SM)

        return active_warps / MAX_WARPS_PER_SM
```

---

## 6. Phase 3 — RL Environment

### 6.1 `environment/kernel_env.py`

```python
"""
Gymnasium environment for CUDA kernel optimization.

The agent takes actions (block_size, reg_cap, shared_mem)
and receives rewards based on measured kernel speedup relative to baseline.

State:
    [cupti_vector(5), nvml_vector(4), kernel_type_onehot(3), prev_action(4)]
    Total: 16 dimensions

Action:
    Discrete MultiDiscrete:
    [block_size_idx, reg_cap_idx, shared_mem_idx, unroll_idx]

    block_size options:  [64, 128, 256, 512]               (4 choices)
    reg_cap options:     [0, 16, 24, 32, 40, 48, 64, 128]  (8 choices, 0=default)
    shared_mem options:  [0, 8192, 16384, 32768, 49152]    (5 choices, bytes)
    unroll options:      [1, 2, 4, 8]                       (4 choices)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

from profiling.cupti_collector import CUPTICollector
from profiling.nvml_monitor import NVMLMonitor
from profiling.cuda_timer import time_kernel
from compiler.ptxas_controller import PTXASController
from kernels.gemm import run_gemm
from kernels.reduction import run_reduction


# ── Action space configuration ───────────────────────────────────────
BLOCK_SIZES  = [64, 128, 256, 512]
REG_CAPS     = [0, 16, 24, 32, 40, 48, 64, 128]  # 0 = PTXAS default (no cap)
SHARED_MEMS  = [0, 8192, 16384, 32768, 49152]     # bytes
UNROLL_FACTS = [1, 2, 4, 8]

KERNEL_NAMES = ["gemm", "reduction", "softmax"]

# State dimension
STATE_DIM = 5 + 4 + len(KERNEL_NAMES) + 4  # cupti + nvml + onehot + prev_action
ACTION_DIM = [len(BLOCK_SIZES), len(REG_CAPS), len(SHARED_MEMS), len(UNROLL_FACTS)]


@dataclass
class EpisodeConfig:
    kernel_name: str = "gemm"
    matrix_size: int = 512
    max_steps: int = 50


class KernelOptimizationEnv(gym.Env):
    """
    Gymnasium environment for RL-guided CUDA kernel optimization.

    Each episode optimizes one kernel instance (fixed kernel_name + matrix_size).
    The agent tries different configurations (actions) and receives reward
    proportional to speedup over the PTXAS default configuration.

    Episode structure:
      1. Compile and run kernel with PTXAS default → baseline_time
      2. For each step:
         a. Agent selects action (block_size_idx, reg_cap_idx, ...)
         b. Kernel compiled+run with that config → step_time
         c. Reward = (baseline_time - step_time) / baseline_time  (speedup fraction)
         d. State updated with new CUPTI counters
      3. Episode ends after max_steps
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EpisodeConfig = None, render_mode=None):
        super().__init__()
        self.config = config or EpisodeConfig()

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(STATE_DIM,),
            dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(ACTION_DIM)

        # Components
        self.cupti    = CUPTICollector()
        self.nvml     = NVMLMonitor()
        self.ptxas    = PTXASController()

        # Episode state
        self._baseline_time: Optional[float] = None
        self._step_count: int = 0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._best_speedup: float = 0.0

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._best_speedup = 0.0

        # Measure baseline: PTXAS defaults, block_size=256
        self._baseline_time = self._measure_kernel(
            block_size=256, reg_cap=0, shared_mem=0, unroll=1
        ).mean_ms

        obs = self._build_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step_count += 1

        # Decode action
        block_size  = BLOCK_SIZES[action[0]]
        reg_cap     = REG_CAPS[action[1]]
        shared_mem  = SHARED_MEMS[action[2]]
        unroll      = UNROLL_FACTS[action[3]]

        # Run kernel with this configuration
        timing = self._measure_kernel(block_size, reg_cap, shared_mem, unroll)
        step_time = timing.mean_ms

        # Compute reward
        # Positive = improvement over baseline, Negative = regression
        if self._baseline_time > 0:
            speedup = (self._baseline_time - step_time) / self._baseline_time
        else:
            speedup = 0.0

        # Scale reward and clip to [-1, 1]
        # A 26% speedup (CuAsmRL max) would give reward ≈ 0.26
        reward = float(np.clip(speedup, -1.0, 1.0))

        if speedup > self._best_speedup:
            self._best_speedup = speedup

        # Normalize action for state
        self._prev_action = np.array([
            block_size / 512.0,
            reg_cap / 128.0,
            shared_mem / 49152.0,
            unroll / 8.0
        ], dtype=np.float32)

        obs = self._build_obs()

        terminated = False  # Never terminate early
        truncated  = (self._step_count >= self.config.max_steps)

        info = {
            "step_time_ms": step_time,
            "baseline_time_ms": self._baseline_time,
            "speedup": speedup,
            "best_speedup": self._best_speedup,
            "block_size": block_size,
            "reg_cap": reg_cap,
        }

        return obs, reward, terminated, truncated, info

    def _measure_kernel(self, block_size: int, reg_cap: int,
                         shared_mem: int, unroll: int):
        """Run the configured kernel and return timing."""
        import numba.cuda as cuda
        from profiling.cuda_timer import time_kernel

        kernel_name = self.config.kernel_name
        N = self.config.matrix_size

        if kernel_name == "gemm":
            from kernels.gemm import gemm_kernel_16, run_gemm
            A, B, C, grid, block = run_gemm(N, block_size, warmup=2)
            return time_kernel(gemm_kernel_16, grid, block, (A, B, C, N), warmup=2, repeats=5)
        elif kernel_name == "reduction":
            from kernels.reduction import reduction_kernel, run_reduction
            x, out, grid, block = run_reduction(N * N, block_size, warmup=2)
            return time_kernel(reduction_kernel, grid, block, (x, out, N*N), warmup=2, repeats=5)
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")

    def _collect_cupti_state(self) -> np.ndarray:
        """
        Collect CUPTI counters for current kernel.
        Returns normalized 5-dim vector.

        Note: For prototype, we use NVML as a faster approximation
        when ncu is too slow (ncu adds ~2s overhead per collection).
        In production runs, use self.cupti.collect() instead.
        """
        # Fast approximation via NVML (for training speed)
        nvml_vec = self.nvml.to_vector()  # [gpu_util, mem_util, mem_used_ratio, temp]
        # Pad/map to CUPTI-like format
        cupti_approx = np.array([
            nvml_vec[0],  # sm_util → achieved_occupancy proxy
            0.5,          # l2_hit_rate (unknown from NVML)
            nvml_vec[1],  # mem_util → dram_bw proxy
            nvml_vec[0],  # warp_eff proxy
            nvml_vec[0],  # sm_active proxy
        ], dtype=np.float32)
        return cupti_approx

    def _build_obs(self) -> np.ndarray:
        """Build the full observation vector."""
        cupti_vec  = self._collect_cupti_state()         # (5,)
        nvml_vec   = self.nvml.to_vector()               # (4,)
        kernel_ohe = self._kernel_onehot()               # (3,)
        prev_act   = self._prev_action                   # (4,)

        obs = np.concatenate([cupti_vec, nvml_vec, kernel_ohe, prev_act])
        return obs.astype(np.float32)

    def _kernel_onehot(self) -> np.ndarray:
        idx = KERNEL_NAMES.index(self.config.kernel_name)
        ohe = np.zeros(len(KERNEL_NAMES), dtype=np.float32)
        ohe[idx] = 1.0
        return ohe
```

---

## 7. Phase 4 — PPO Agent Training

### 7.1 `training/config.py`

```python
"""
All hyperparameters in one place.
Modify this file first when tuning — do not scatter magic numbers.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOConfig:
    # Network architecture
    policy_type: str   = "MlpPolicy"       # SB3 policy type
    net_arch: List[int] = field(default_factory=lambda: [128, 128, 64])  # hidden layers

    # PPO hyperparameters
    learning_rate: float  = 3e-4
    n_steps: int          = 512     # Steps per rollout per env
    batch_size: int       = 64
    n_epochs: int         = 10
    gamma: float          = 0.95    # Lower than typical (0.99) because horizon is short
    gae_lambda: float     = 0.95
    clip_range: float     = 0.2     # PPO epsilon
    ent_coef: float       = 0.01    # Entropy coefficient (encourages exploration)
    vf_coef: float        = 0.5
    max_grad_norm: float  = 0.5

    # Training duration
    total_timesteps: int  = 50_000  # Enough for prototype demonstration

    # Evaluation
    eval_freq: int        = 2_000   # Evaluate every N steps
    n_eval_episodes: int  = 5

    # Saving
    save_freq: int        = 5_000
    checkpoint_dir: str   = "results/checkpoints"
    log_dir: str          = "results/logs"


@dataclass
class EnvConfig:
    kernel_names: List[str]  = field(default_factory=lambda: ["gemm", "reduction"])
    matrix_sizes: List[int]  = field(default_factory=lambda: [256, 512, 1024])
    max_steps_per_episode: int = 50   # How many config tries per episode


@dataclass
class ExperimentConfig:
    ppo:  PPOConfig = field(default_factory=PPOConfig)
    env:  EnvConfig = field(default_factory=EnvConfig)
    seed: int       = 42
    device: str     = "cuda"   # or "cpu" for policy network
```

---

### 7.2 `training/train_rl.py`

```python
"""
Main PPO training entry point.

Usage:
    python training/train_rl.py

This trains the PPO agent on the KernelOptimizationEnv
and saves checkpoints + tensorboard logs.
"""

import sys
sys.path.insert(0, ".")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
from rich.console import Console

from environment.kernel_env import KernelOptimizationEnv, EpisodeConfig
from training.config import ExperimentConfig

console = Console()


def make_env(kernel_name: str = "gemm", matrix_size: int = 512):
    """Environment factory for SB3 vectorized environments."""
    def _init():
        config = EpisodeConfig(
            kernel_name=kernel_name,
            matrix_size=matrix_size,
            max_steps=50
        )
        env = KernelOptimizationEnv(config=config)
        env = Monitor(env)
        return env
    return _init


def train(cfg: ExperimentConfig = None):
    cfg = cfg or ExperimentConfig()

    # Paths
    Path(cfg.ppo.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.ppo.log_dir).mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Setting up environments...[/cyan]")

    # Training environment — GEMM kernel, 512x512
    train_env = make_vec_env(
        make_env("gemm", 512),
        n_envs=1,    # Single env for GPU (can't parallelize GPU)
        seed=cfg.seed
    )

    # Eval environment — different size to test generalization
    eval_env = make_vec_env(
        make_env("gemm", 1024),
        n_envs=1,
        seed=cfg.seed + 1
    )

    console.print("[cyan]Building PPO agent...[/cyan]")

    # Custom policy network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=cfg.ppo.net_arch,   # Actor network
            vf=cfg.ppo.net_arch    # Critic network
        ),
        activation_fn=torch.nn.ReLU
    )

    model = PPO(
        policy=cfg.ppo.policy_type,
        env=train_env,
        learning_rate=cfg.ppo.learning_rate,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=cfg.ppo.log_dir,
        verbose=1,
        seed=cfg.seed,
        device="cpu"   # Policy network on CPU; GPU used for kernel execution
    )

    console.print(f"[green]Model: {sum(p.numel() for p in model.policy.parameters())} parameters[/green]")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg.ppo.checkpoint_dir,
        log_path=cfg.ppo.log_dir,
        eval_freq=cfg.ppo.eval_freq,
        n_eval_episodes=cfg.ppo.n_eval_episodes,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.ppo.save_freq,
        save_path=cfg.ppo.checkpoint_dir,
        name_prefix="ppo_kernel"
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    console.print(f"[cyan]Starting training for {cfg.ppo.total_timesteps} timesteps...[/cyan]")
    console.print("[yellow]Tensorboard: tensorboard --logdir results/logs[/yellow]")

    model.learn(
        total_timesteps=cfg.ppo.total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    # Save final model
    model.save(f"{cfg.ppo.checkpoint_dir}/ppo_kernel_final")
    console.print("[bold green]Training complete![/bold green]")

    return model


def evaluate(model_path: str, kernel_name: str = "gemm",
             matrix_size: int = 512, n_episodes: int = 10):
    """Load and evaluate a trained model."""
    model = PPO.load(model_path)

    config = EpisodeConfig(kernel_name=kernel_name,
                           matrix_size=matrix_size, max_steps=50)
    env = KernelOptimizationEnv(config=config)

    speedups = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        best_speedup = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info["speedup"] > best_speedup:
                best_speedup = info["speedup"]
        speedups.append(best_speedup)
        console.print(f"Episode {ep+1}: best speedup = {best_speedup:.1%}")

    console.print(f"\n[bold]Mean best speedup: {np.mean(speedups):.1%} ± {np.std(speedups):.1%}[/bold]")
    return speedups


if __name__ == "__main__":
    model = train()
```

---

## 8. Phase 5 — BiLSTM Phase Detector

### 8.1 `models/phase_detector.py`

```python
"""
BiLSTM-based temporal phase detector.

Input:  Sliding window of T=20 CUPTI counter vectors, shape (T, 5)
Output: Phase probability distribution over {compute, memory, latency, mixed}
        + uncertainty scalar

Phase labels (ground truth from roofline):
  0 = compute-bound  (high IPC, low DRAM BW, kernel on compute ceiling)
  1 = memory-bound   (low IPC, high DRAM BW, kernel on memory BW slope)
  2 = latency-bound  (low IPC, low DRAM BW, warp stalls dominate)
  3 = mixed          (transitions, unclear regime)

Training data:
  - Collect CUPTI traces for GEMM at various sizes
  - Label by arithmetic intensity vs roofline ridge point:
    RTX 3050 Ti ridge ≈ 7.8 TFLOP/s ÷ 192 GB/s ≈ 40.6 FLOP/byte
  - Small matrices (N<256): latency-bound
  - Medium matrices (256<N<1024): memory-bound
  - Large matrices (N>1024): compute-bound (for GEMM)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class PhaseDetector(nn.Module):
    """
    Bidirectional LSTM for GPU kernel phase detection.

    Architecture:
      Input:  (batch, T, 5)         ← T timesteps of 5 CUPTI counters
      BiLSTM: (batch, T, 2*hidden)  ← bidirectional
      MLP:    (batch, 4)            ← phase probabilities
              (batch, 1)            ← uncertainty [0,1]
    """

    WINDOW_SIZE = 20
    INPUT_DIM   = 5    # CUPTI counter dimensions
    NUM_PHASES  = 4    # compute, memory, latency, mixed

    PHASE_NAMES = {
        0: "compute-bound",
        1: "memory-bound",
        2: "latency-bound",
        3: "mixed"
    }

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.INPUT_DIM,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Phase classification head
        self.phase_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, self.NUM_PHASES)
        )

        # Uncertainty head (epistemic — higher = less confident)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, T, 5)
               T = window size (20 timesteps)
               5 = CUPTI counter dimensions

        Returns:
            phase_probs:  (batch, 4) — softmax probabilities
            uncertainty:  (batch, 1) — scalar uncertainty [0, 1]
        """
        # BiLSTM: output shape (batch, T, 2*hidden)
        lstm_out, _ = self.lstm(x)

        # Use last forward + last backward hidden states
        # Forward: lstm_out[:, -1, :hidden_dim]
        # Backward: lstm_out[:, 0, hidden_dim:]
        fwd = lstm_out[:, -1, :self.hidden_dim]
        bwd = lstm_out[:, 0,  self.hidden_dim:]
        combined = torch.cat([fwd, bwd], dim=-1)   # (batch, 2*hidden)

        phase_logits = self.phase_head(combined)
        phase_probs  = torch.softmax(phase_logits, dim=-1)
        uncertainty  = self.uncertainty_head(combined)

        return phase_probs, uncertainty

    def predict(self, counter_window: np.ndarray) -> Tuple[int, float, float]:
        """
        Single-window inference.

        Args:
            counter_window: np.ndarray of shape (T, 5) or (5,) for single step

        Returns:
            phase_label:  int in {0,1,2,3}
            confidence:   float in [0,1]
            uncertainty:  float in [0,1]
        """
        self.eval()
        with torch.no_grad():
            if counter_window.ndim == 1:
                # Single timestep — pad to window
                window = np.zeros((self.WINDOW_SIZE, self.INPUT_DIM), dtype=np.float32)
                window[-1] = counter_window
            else:
                window = counter_window[-self.WINDOW_SIZE:]  # Take last T steps

            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, T, 5)
            probs, unc = self.forward(x)

            phase_label = int(probs.argmax(dim=-1).item())
            confidence  = float(probs.max().item())
            uncertainty = float(unc.item())

        return phase_label, confidence, uncertainty

    def phase_name(self, label: int) -> str:
        return self.PHASE_NAMES.get(label, "unknown")


def create_training_labels(cupti_traces: np.ndarray,
                            arithmetic_intensities: np.ndarray,
                            ridge_point: float = 40.6) -> np.ndarray:
    """
    Generate phase labels for training data based on roofline model.

    RTX 3050 Ti roofline ridge point:
      Peak compute: ~7.8 TFLOP/s (FP32)
      Peak memory BW: ~192 GB/s
      Ridge point = 7.8e12 / 192e9 ≈ 40.6 FLOP/byte

    Args:
        cupti_traces: (N, 5) array of normalized CUPTI counter measurements
        arithmetic_intensities: (N,) array of FLOP/byte for each measurement
        ridge_point: FLOP/byte at compute/memory boundary

    Returns:
        labels: (N,) int array with values in {0,1,2,3}
    """
    labels = np.zeros(len(arithmetic_intensities), dtype=int)

    for i, ai in enumerate(arithmetic_intensities):
        achieved_occ = cupti_traces[i, 0]
        dram_bw_pct  = cupti_traces[i, 2]

        if ai > ridge_point and achieved_occ > 0.5:
            labels[i] = 0  # compute-bound
        elif ai < ridge_point and dram_bw_pct > 0.5:
            labels[i] = 1  # memory-bound
        elif achieved_occ < 0.3 and dram_bw_pct < 0.3:
            labels[i] = 2  # latency-bound
        else:
            labels[i] = 3  # mixed

    return labels
```

---

## 9. Phase 6 — GNN IR Encoder

> **This is optional — implement only after Phase 4 is working and producing results.**
> The GNN upgrades the state representation from PMU-counters-only to
> PMU-counters + kernel-structure, which is the key DL contribution.

### 9.1 `compiler/ir_extractor.py`

```python
"""
Extract LLVM IR and PTX from compiled Numba CUDA kernels.
These are used to build the graph for the GNN encoder.
"""

import numba.cuda as cuda
from numba.cuda import compiler as cuda_compiler
import numpy as np
from pathlib import Path
import re


def extract_ptx(kernel_fn, arg_types: tuple, cc: tuple = (8, 6)) -> str:
    """
    Extract PTX source from a Numba CUDA kernel.

    Args:
        kernel_fn: Numba CUDA kernel function (decorated with @cuda.jit)
        arg_types: Tuple of Numba types matching kernel signature
        cc: Compute capability tuple, default (8,6) for RTX 3050 Ti

    Returns:
        PTX source code as string
    """
    ptx, _ = cuda_compiler.compile_ptx(
        pyfunc=kernel_fn.py_func,
        sig=arg_types,
        cc=cc,
        device=False,
        fastmath=False
    )
    return ptx


def ptx_to_graph_features(ptx_source: str) -> dict:
    """
    Extract structural features from PTX source for GNN input.

    Returns a dict suitable for building a PyTorch Geometric Data object.
    This is a simplified version for the prototype — the full GNN
    implementation is in models/gnn_encoder.py.
    """
    lines = ptx_source.split('\n')

    # Count instruction types
    features = {
        "n_instructions": 0,
        "n_loads":        0,
        "n_stores":       0,
        "n_fma":          0,     # Fused multiply-add (compute intensity)
        "n_branches":     0,
        "n_sync":         0,     # syncthreads calls
        "n_registers":    0,
        "shared_mem_bytes": 0,
        "n_loop_bodies":  0,     # Estimated from branch patterns
    }

    for line in lines:
        line = line.strip()
        if not line or line.startswith('//') or line.startswith('.'):
            continue
        features["n_instructions"] += 1
        if line.startswith('ld.') or 'ld.' in line:
            features["n_loads"] += 1
        if line.startswith('st.') or 'st.' in line:
            features["n_stores"] += 1
        if 'fma' in line or 'mad' in line:
            features["n_fma"] += 1
        if line.startswith('bra') or line.startswith('setp'):
            features["n_branches"] += 1
        if 'bar.sync' in line:
            features["n_sync"] += 1

    # Extract register count from PTX header
    reg_match = re.search(r'\.reg\s+\.f32\s+%f<(\d+)>', ptx_source)
    if reg_match:
        features["n_registers"] = int(reg_match.group(1))

    # Extract shared memory
    smem_match = re.search(r'\.shared\s+\.align\s+\d+\s+\.b8\s+\w+\[(\d+)\]', ptx_source)
    if smem_match:
        features["shared_mem_bytes"] = int(smem_match.group(1))

    # Compute arithmetic intensity proxy
    total_mem_ops = features["n_loads"] + features["n_stores"]
    if total_mem_ops > 0:
        features["arithmetic_intensity_proxy"] = features["n_fma"] / total_mem_ops
    else:
        features["arithmetic_intensity_proxy"] = 0.0

    return features
```

---

## 10. Phase 7 — Evaluation & Results Tables

### 10.1 `experiments/phase2_rl_baseline.py`

```python
"""
Phase 2: RL vs Baselines Comparison

Runs three strategies on each kernel and produces the main results table:
  1. PTXAS default (no hints)
  2. Random search (N=100 random configurations)
  3. Trained PPO agent (deterministic)

Expected output table:

Strategy        | Kernel    | Size | Mean time (ms) | Best speedup | Samples
----------------|-----------|------|----------------|--------------|--------
PTXAS default   | gemm      | 512  |  X.XX ± 0.XX   |     ---      |   1
Random search   | gemm      | 512  |  Y.YY ± 0.YY   |   Z.Z%       | 100
PPO agent       | gemm      | 512  |  W.WW ± 0.WW   |   V.V%       |  50

Run after training: python experiments/phase2_rl_baseline.py --model results/checkpoints/best_model
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from rich.console import Console
from rich.table import Table

from environment.kernel_env import KernelOptimizationEnv, EpisodeConfig, BLOCK_SIZES, REG_CAPS

console = Console()


def run_ptxas_default(kernel: str, size: int, repeats: int = 10) -> dict:
    """Baseline: run with PTXAS defaults."""
    env = KernelOptimizationEnv(EpisodeConfig(kernel, size))
    env.reset()
    timing = env._measure_kernel(block_size=256, reg_cap=0, shared_mem=0, unroll=1)
    return {
        "strategy": "PTXAS default",
        "kernel": kernel, "size": size,
        "time_mean": timing.mean_ms, "time_std": timing.std_ms,
        "speedup": 0.0, "n_samples": 1
    }


def run_random_search(kernel: str, size: int, n_samples: int = 100) -> dict:
    """Baseline: random search over configuration space."""
    env = KernelOptimizationEnv(EpisodeConfig(kernel, size))
    obs, _ = env.reset()
    baseline = env._baseline_time

    best_time = baseline
    best_speedup = 0.0

    for _ in range(n_samples):
        action = env.action_space.sample()
        obs, reward, _, _, info = env.step(action)
        if info["step_time_ms"] < best_time:
            best_time = info["step_time_ms"]
            best_speedup = info["speedup"]

    return {
        "strategy": "Random search",
        "kernel": kernel, "size": size,
        "time_mean": best_time, "time_std": 0.0,
        "speedup": best_speedup, "n_samples": n_samples
    }


def run_ppo_agent(model_path: str, kernel: str, size: int, n_episodes: int = 5) -> dict:
    """Evaluate trained PPO agent."""
    model = PPO.load(model_path)
    env = KernelOptimizationEnv(EpisodeConfig(kernel, size, max_steps=50))

    all_best_speedups = []
    all_best_times = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        baseline = env._baseline_time
        best_time = baseline
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info["step_time_ms"] < best_time:
                best_time = info["step_time_ms"]

        speedup = (baseline - best_time) / baseline
        all_best_speedups.append(speedup)
        all_best_times.append(best_time)

    return {
        "strategy": "PPO agent",
        "kernel": kernel, "size": size,
        "time_mean": np.mean(all_best_times),
        "time_std": np.std(all_best_times),
        "speedup": np.mean(all_best_speedups),
        "n_samples": 50 * n_episodes
    }


def run_comparison(model_path: str):
    """Run full comparison and produce results table."""
    kernels = ["gemm", "reduction"]
    sizes   = [256, 512, 1024]
    results = []

    for kernel in kernels:
        for size in sizes:
            console.print(f"\n[cyan]Running: {kernel} size={size}[/cyan]")
            results.append(run_ptxas_default(kernel, size))
            results.append(run_random_search(kernel, size, n_samples=100))
            if model_path:
                results.append(run_ppo_agent(model_path, kernel, size))

    df = pd.DataFrame(results)
    df.to_csv("results/tables/phase2_comparison.csv", index=False)

    # Pretty print
    table = Table(title="RL vs Baselines Comparison")
    for col in ["strategy", "kernel", "size", "time_mean", "speedup", "n_samples"]:
        table.add_column(col)

    for _, row in df.iterrows():
        table.add_row(
            row["strategy"],
            row["kernel"],
            str(row["size"]),
            f"{row['time_mean']:.2f}ms",
            f"{row['speedup']:.1%}",
            str(int(row["n_samples"]))
        )

    console.print(table)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to trained PPO model")
    args = parser.parse_args()
    run_comparison(args.model)
```

---

## 11. Copilot Prompting Guide

> Use these prompts when working with GitHub Copilot or Antigravity.
> Always provide the relevant module file as context first.

### 11.1 For implementing a new kernel

```
Context: kernels/gemm.py is open in editor.

Prompt: "Implement a softmax kernel in kernels/softmax.py following
the same pattern as the gemm kernel. It should:
- Be parameterized by block_size and take a 2D float32 input
- Compute row-wise softmax in-place using shared memory
- Have a corresponding run_softmax() factory function
- Follow Numba @cuda.jit conventions matching gemm.py"
```

### 11.2 For CUPTI collection debugging

```
Context: profiling/cupti_collector.py is open.

Prompt: "The ncu subprocess in CUPTICollector.collect() is returning
empty CSV output. Add a debug mode that:
- Saves the raw ncu stdout/stderr to a debug log file
- Tries a fallback with fewer metrics if the full set fails
- Reports which specific metrics were successfully parsed"
```

### 11.3 For the training loop

```
Context: training/train_rl.py and environment/kernel_env.py are open.

Prompt: "The PPO training in train_rl.py is producing NaN rewards after
step 200. Add:
- Gradient clipping verification in the training callback
- Reward normalization wrapper around the environment
- Episode statistics logging to tensorboard including mean/max speedup"
```

### 11.4 For results visualization

```
Context: experiments/phase2_rl_baseline.py results CSV is in context.

Prompt: "Generate a matplotlib visualization script that reads
results/tables/phase2_comparison.csv and produces:
1. A grouped bar chart: PTXAS vs Random vs PPO for each kernel×size
2. A scatter plot: achieved occupancy vs speedup (one point per configuration)
3. Save both to results/plots/ as high-DPI PNG"
```

---

## 12. Troubleshooting

### CUPTI / ncu Issues

```bash
# Error: "ERR_NVGPUCTRPERM — The user does not have permission"
# Fix:
sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
# Or add current user to 'video' group:
sudo usermod -aG video $USER && newgrp video

# Error: ncu causes kernel to run ~100x slower (profiling overhead)
# This is expected. Use separate timing and CUPTI collection runs.
# timing_run: no ncu, use CUDA events
# cupti_run: use ncu, timing from ncu report (add --csv metrics)

# Error: "No kernels were profiled"
# Fix: The kernel may have already run and exited before ncu attached.
# Add a sleep or synchronization point before the kernel launch.
```

### Numba / CUDA Issues

```bash
# Error: "CUDA_ERROR_INVALID_PTX"
# Cause: Compiled PTX is not compatible with target SM
# Fix: Ensure cc=(8,6) is passed to compile_ptx for RTX 3050 Ti
python3 -c "import numba.cuda as c; print(c.get_current_device().compute_capability)"
# Should print (8, 6)

# Error: "numba.cuda.cudadrv.error.CudaSupportError: Error at driver init"
# Fix: Check CUDA is available and CUDA_HOME is set
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Error: OOM when running large matrices (N=2048)
# RTX 3050 Ti has 4GB VRAM. Two 2048x2048 float32 matrices = 2×64MB = fine
# But three matrices + activations can exceed this at N=4096
# Stick to N ≤ 2048 for training
```

### Stable-Baselines3 / RL Issues

```bash
# NaN in rewards: check reward normalization
# Add VecNormalize wrapper:
from stable_baselines3.common.vec_env import VecNormalize
env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Slow training: each env step compiles + runs a kernel (~seconds)
# This is expected. 50,000 timesteps ≈ 50,000 kernel runs ≈ hours
# Start with 5,000 timesteps to verify the training loop works

# Agent not improving: increase ent_coef to 0.05 for more exploration
# The action space is small (4×8×5×4 = 640 combinations)
# Random search should find something good quickly — if agent is worse
# than random after 10k steps, there's a bug in the reward function
```

### Quick Validation Checklist

```bash
# Run these in order to validate each phase before proceeding

# Phase 0: Can we compile a Numba kernel?
python3 -c "
import numba.cuda as cuda
import numpy as np
@cuda.jit
def test(a, b):
    i = cuda.grid(1)
    if i < a.shape[0]: b[i] = a[i] * 2.0
a = cuda.to_device(np.ones(1024, dtype=np.float32))
b = cuda.device_array(1024, dtype=np.float32)
test[32, 32](a, b)
cuda.synchronize()
print('PASS: Numba CUDA kernel compiles and runs')
"

# Phase 1: Can we time a kernel?
python3 -c "
from profiling.cuda_timer import time_kernel
from kernels.gemm import gemm_kernel_16, run_gemm
import numba.cuda as cuda
A, B, C, grid, block = run_gemm(256)
result = time_kernel(gemm_kernel_16, grid, block, (A, B, C, 256))
print(f'PASS: GEMM 256x256 = {result.mean_ms:.2f}ms ± {result.std_ms:.2f}ms')
"

# Phase 2: Can we read GPU state via NVML?
python3 -c "
from profiling.nvml_monitor import NVMLMonitor
m = NVMLMonitor()
s = m.get_state()
print(f'PASS: GPU util={s.gpu_util_pct}%, Temp={s.temperature_c}°C')
"

# Phase 3: Can we create the Gym environment?
python3 -c "
from environment.kernel_env import KernelOptimizationEnv, EpisodeConfig
env = KernelOptimizationEnv(EpisodeConfig('gemm', 256, max_steps=5))
obs, _ = env.reset()
action = env.action_space.sample()
obs, rew, term, trunc, info = env.step(action)
print(f'PASS: Env step OK, reward={rew:.4f}, speedup={info[\"speedup\"]:.1%}')
"

# Phase 4: Does training start without errors?
python3 -c "
from training.train_rl import train
from training.config import ExperimentConfig, PPOConfig
cfg = ExperimentConfig()
cfg.ppo.total_timesteps = 100   # Just 100 steps to verify
model = train(cfg)
print('PASS: PPO training loop runs without errors')
"
```

---

*Implementation plan complete. Start with Phase 0. Each phase builds on the previous one.*
*The foundational experiment (Phase 0) is the most important — run it first and verify the*
*register count vs occupancy curve exists before implementing any ML component.*
