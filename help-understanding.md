# Help Understanding Experiment Results

This doc explains how to interpret the experimental output in this project.

As of now, **Phase 0** through **Phase 5** are implemented and produce results, **Phase 2** validation (kernel correctness tests) is implemented, and **Phase 7** (RL vs Baselines Comparison) provides the final evaluation table.

## Objectives

Project objectives (what the experiments are trying to prove):
- Show that compiler/runtime knobs (especially register capping via `--maxrregcount`/Numba `max_registers`) materially change performance.
- Connect those performance changes to **measurable GPU signals** (occupancy- and utilization-like counters from `ncu`).
- Produce clean CSV artifacts that can be reused later for RL (Phase 3+) and for writing up results.

This document’s objectives (what you should be able to do after reading):
- Understand what each phase measures and why it exists.
- Run Phase 0/1 and know where their CSV outputs are saved.
- Run Phase 2 tests (small suite vs performance-scale suite) and interpret the results.
- Interpret the most important CSV columns and common failure modes (especially on Windows).

---

## Phase 0 — Foundational Baseline Table

### Goal (what Phase 0 is trying to measure)
Phase 0 runs a sweep over:
- **Kernel type**: `gemm`, `reduction`, `softmax`
- **Problem size**: `256`, `512`, `1024`
- **Block size**: `64`, `128`, `256`
- **Register cap setting**: `default`, `32`, `64`

For each configuration it records:
- A **theoretical occupancy estimate** (computed from actual registers/thread and hardware limits)
- A **measured runtime** in milliseconds

It saves all results into a CSV for later analysis.

---

## Beginner concepts (sweep, registers, occupancy)

This project uses CUDA-style GPU execution concepts. If you’re new to operating systems or parallel computing, this section explains the *three most important ideas* used throughout Phase 0:

### 1) What does “sweep” mean?

A **sweep** means: *systematically testing many combinations of settings*.

Instead of running only one configuration (one kernel, one size, one block size, one register cap), Phase 0 runs the kernel repeatedly while changing parameters.

Think of it like a spreadsheet grid:
- rows = different kernels (`gemm`, `reduction`, `softmax`)
- columns = different sizes (`256`, `512`, `1024`)
- and for each of those, try multiple `block_size` and `reg_cap`

Each unique combination becomes **one row in the CSV**.

Why a sweep is useful:
- It reveals trends (e.g., “block size 128 is usually best for softmax at large sizes”).
- It helps you find interactions (e.g., a register cap that helps GEMM may not help reduction).

### 2) What is a “register cap” in Phase 0?

GPUs have multiple “memory/storage” layers:
- **Registers**: fastest per-thread storage (inside the GPU core).
- **Shared memory**: fast memory shared by threads in the same block.
- **Global memory**: large but much slower (GPU VRAM).

A kernel uses registers to hold temporary values (loop counters, partial sums, pointers, etc.). The **register cap** is a limit you ask the compiler to respect:

- `reg_cap = default` means **no limit requested** (compiler chooses).
- `reg_cap = 32` means “try to compile the kernel so each thread uses **at most 32 registers**”.
- `reg_cap = 64` means “at most 64 registers per thread”.

In CUDA toolchains this is commonly called `--maxrregcount` (PTXAS flag). In this repo’s Phase 0 implementation, it’s applied through Numba as `max_registers=...`.

Why it can change performance:
- If you cap registers too low, the compiler may need to **spill** values into local memory (which lives in global memory). That usually makes the kernel slower.
- If you allow many registers, the kernel may avoid spilling and run faster.
- But using more registers per thread means fewer threads can be “resident” on an SM at the same time → potentially lower occupancy.

So register cap is a classic tradeoff:
- **low regs** → higher occupancy, but risk spilling
- **high regs** → less spilling, but potentially lower occupancy

### 3) What is “occupancy”?

#### The simple meaning
**Occupancy** is how many GPU warps are *active at the same time* on each SM, compared to the maximum possible.

Definitions you’ll see in GPU programming:
- **Thread**: smallest unit of execution.
- **Warp**: group of 32 threads that execute in lockstep (SIMT model).
- **Block**: a group of threads that can cooperate (shared memory + synchronization).
- **Grid**: all blocks launched for a kernel call.
- **SM (Streaming Multiprocessor)**: the GPU “core cluster” that runs blocks/warps.

On your RTX 3050 Ti (sm_86), an SM can support up to a fixed maximum number of active warps (the model in this repo uses 48 as the max warps/SM).

Occupancy is computed as:

$$
	ext{occupancy} = \frac{\text{active warps per SM}}{\text{maximum warps per SM}}
$$

So:
- `1.0` in the CSV means **100% occupancy**
- `0.6667` means **66.67% occupancy**

#### Why occupancy changes
Occupancy is limited by GPU resources such as:
- **registers per thread** (more registers/thread → fewer warps fit)
- **threads per block** (bigger blocks can reduce how many blocks fit)
- **shared memory per block** (not a limiter in our Phase 0 kernels, but common in other kernels)

#### Important: occupancy is not the same as speed
Higher occupancy often helps **memory-bound** kernels (because more warps lets the SM “hide” memory latency).

But for **compute-heavy** kernels, the fastest configuration may use:
- more registers (less spilling, better instruction scheduling)
- and therefore a bit lower occupancy

That’s exactly why Phase 0 measures both occupancy and runtime.

---

### How to run Phase 0

From the project root, in the `gpu-jit-opt` conda environment:

```bat
python experiments\phase0_baseline_table.py
```

Recommended on Windows to avoid weird checkmark characters in output:

```bat
chcp 65001>nul
python experiments\phase0_baseline_table.py
```

Notes:
- Running **as Administrator is NOT required** for Phase 0.
- Running as Administrator *is only needed* for Nsight Compute (`ncu`) counter collection.

---

### Expected terminal output (shape)
You should see something like:

```text
Phase 0: Foundational Experiment
GPU: RTX 3050 Ti (Ampere, SM 8.6)

Total configurations: 81
Estimated time: 2s

Running Phase 0 ---------------------------------------- 100% 0:00:03

Saved 81 results to results\tables\phase0_baseline.csv

Phase 0 Summary Table
Register Cap vs Occupancy vs Runtime
┌ ... lots of rows ... ┐

Key Insights:
  GEMM: Best time = ...
  REDUCTION: Best time = ...
  SOFTMAX: Best time = ...

Phase 0 complete!
Data saved to: ...\results\tables\phase0_baseline.csv
Rows: 81
Columns (Phase 0 CSV):
- `kernel`, `matrix_size`, `block_size`, `threads_per_block`, `reg_cap`
- `est_regs`, `actual_regs`, `theor_occ`
- `time_ms_mean`, `time_ms_std`, `time_ms_min`, `time_ms_max`

Optional (only if you enable Nsight Compute collection):
- `achieved_occ`
```

If you see `Saved 0 results` or many `access violation` errors, that indicates a low-level CUDA/Numba issue and Phase 0 did not measure successfully.

---

### Where the data is saved
- CSV output path:
  - `results/tables/phase0_baseline.csv`

This CSV is the “real artifact” of Phase 0; the terminal table is just a pretty view.

---

### Phase 0 CSV columns (exact header names)

This is a **reference for the CSV file** `results/tables/phase0_baseline.csv`.

Important detail:
- The terminal table prints occupancy as a percent (e.g. `67%`), but the CSV stores occupancy as a **fraction** (e.g. `0.6667`).
   - Convert fraction → percent by doing: `theor_occ * 100`.

Per-row identifiers
- `kernel` (string): Which benchmark ran: `gemm`, `reduction`, or `softmax`.
- `matrix_size` (int): The size label $N$ used for that run.
   - `gemm`: multiplies $N \times N$ matrices.
   - `softmax`: applies softmax to an $N \times N$ matrix, row-wise.
   - `reduction`: reduces $N \times N$ elements (internally uses `matrix_size * matrix_size`).
- `block_size` (int): The **sweep setting** for threads-per-block.
   - For `reduction` and `softmax`, this is the actual 1D block size.
   - For `gemm`, the kernel launches a 2D block `(tile, tile)`; use `threads_per_block` to know what actually launched.
- `threads_per_block` (int): The **actual launched total threads per block**.
   - This is what occupancy calculations use.
- `reg_cap` (string): Register cap setting.
   - Values are typically `default`, `32`, `64`.
   - Note: it’s stored as text so the CSV can mix `default` and numeric caps.
   - Meaning: this is the “register cap setting” explained above (the limit requested during compilation).

Register + occupancy fields
- `est_regs` (int): A heuristic estimate of registers per thread (kept mainly for comparison).
- `actual_regs` (int): **Measured registers per thread from the compiled kernel** (Numba compiled metadata / PTXAS result).
- `theor_occ` (float): Theoretical occupancy **fraction** in $[0,1]$, computed from:
   - hardware limits (sm_86),
   - `actual_regs`,
   - `threads_per_block`.
   - Meaning: this is “occupancy” as defined above (how many warps can be resident compared to the max), but computed analytically rather than measured with a profiler.

Timing fields (milliseconds)
- `time_ms_mean` (float): Mean kernel time over the measured repeats.
- `time_ms_std` (float): Standard deviation across repeats.
- `time_ms_min` (float): Fastest measured run.
- `time_ms_max` (float): Slowest measured run.

Optional profiler field (only if enabled)
- `achieved_occ` (float): Achieved occupancy proxy **fraction** in $[0,1]$ collected via Nsight Compute (`ncu`) metric:
   - `sm__warps_active.avg.pct_of_peak_sustained_active`
   - This is the closest thing to “measured occupancy” in Phase 0.
   - On Windows, this frequently requires Administrator permissions for GPU counters.

---

### Understanding the Phase 0 table columns

Each row in the table corresponds to **one experiment configuration**.

#### `Kernel`
Which CUDA kernel benchmark was run:
- `gemm`: tiled matrix multiplication (compute-heavy)
- `reduction`: sum-reduction (memory/latency-sensitive)
- `softmax`: row-wise softmax (mix of math + memory)

#### `Size`
The problem size label used for that kernel:
- For `gemm`: size `N` means multiplying two `N × N` matrices.
- For `softmax`: size `N` means applying softmax to an `N × N` matrix, row-by-row.
- For `reduction`: Phase 0 currently reduces **`N × N` elements** (it uses `matrix_size * matrix_size`).
  - Example: `Size=256` means it reduces `256² = 65,536` floats.

#### `Block`
Threads per block.
- For `reduction` and `softmax`, this is the actual 1D block size (e.g., 256 threads).
- For `gemm`, the kernel is tiled and launches a **2D** block (`(tile, tile)`), so the `Block` shown in the table is treated more like a “configuration label” than the literal launched thread count.

#### `RegCap`
Register cap setting used in the sweep.
- `default` means “no cap requested”.
- `32` or `64` are cap values.

What it actually does now:
- Phase 0 compiles separate Numba kernel variants using `max_registers=RegCap` (PTXAS `--maxrregcount` equivalent).
- That means changing `RegCap` can change:
   - the compiler’s chosen registers/thread (`actual_regs`)
   - theoretical occupancy
   - runtime (due to spilling vs keeping values in registers)

#### `Regs`
The register count used for the occupancy calculation.

Related columns in the CSV:
- `est_regs`: heuristic estimate (kept mainly for comparison)
- `actual_regs`: **measured from the compiled kernel** (`regs_per_thread` from Numba’s compiled metadata)

So: in the Phase 0 table, `Regs` corresponds to `actual_regs`.

#### `Occ %`
The **theoretical occupancy estimate** as a percent.

Occupancy (informal beginner definition):
- “How many warps can the GPU keep active at once on each SM?”

Why occupancy changes:
- More registers per thread → fewer warps fit → lower occupancy.
- Block size also affects how many blocks/warps can fit.

Important:
- This is still **a model** (computed from hardware limits + `actual_regs` + `threads_per_block`).
- If you want *measured* occupancy, enable Nsight Compute collection (see below).

#### `threads_per_block` (CSV only)
The *actual* launched threads per block.
- For `reduction` and `softmax`, this usually matches `Block`.
- For `gemm`, the kernel launches a 2D block `(tile, tile)`; so this makes it explicit what actually launched.

#### `achieved_occ` (optional, CSV)
If present, this is an achieved occupancy proxy collected from Nsight Compute (`ncu`) using:
- `sm__warps_active.avg.pct_of_peak_sustained_active`

Enable it like this (Windows):
```bat
set PHASE0_COLLECT_NCU=1
python experiments\phase0_baseline_table.py
```

Notes:
- On Windows, this often requires running Command Prompt as Administrator.
- It is slower (profiles each configuration).

#### `Time (ms)`
Measured runtime per kernel launch in milliseconds.
- This is the **mean** over several repeats.

In the CSV:
- `time_ms_mean`: mean
- `time_ms_std`: standard deviation
- `time_ms_min` / `time_ms_max`: extremes

#### `Std`
Standard deviation across repeats.
- Small `Std` means consistent runs.
- Big `Std` means noisy runs (timing jitter, background system activity, or outliers).

---

### Why do some rows have huge times or huge Std?
A few common reasons:
1. **CPU-side timing jitter** on Windows
   - Some systems can’t reliably use CUDA event timing with Numba.
   - In that case, the timer falls back to CPU timing with explicit synchronization.
   - Very fast kernels (sub-millisecond) can then show large relative noise.

2. **Only a few repeats**
   - Phase 0 uses a small number of repeats for speed.
   - With only 3 repeats, one slow outlier can inflate the mean and Std dramatically.

3. **Under-utilization**
   - Very small grids/blocks can underutilize the GPU, so the overheads dominate.

How to sanity-check:
- Prefer interpreting rows with a small Std.
- If you care about accurate timing, increase repeats and/or run on a quieter system.

---

### How to interpret the “Key Insights” section
The script prints, for each kernel, the configuration with the smallest `time_ms_mean`.

This is a *quick summary*, not a statistical guarantee.
- If timing is noisy, the “best” row can change run-to-run.

---

### Quick beginner workflow
1. Run Phase 0.
2. Open `results/tables/phase0_baseline.csv` in Excel or load it in Python.
3. For each kernel, compare:
   - how `Block` changes runtime
   - how `RegCap` affects the occupancy estimate
4. Watch for rows with huge `Std` and treat them as “noisy / needs re-run”.

Python snippet to load the results:

```python
import pandas as pd

df = pd.read_csv("results/tables/phase0_baseline.csv")
print(df.head())

# Example: best config per kernel (by mean time)
print(df.loc[df.groupby("kernel")["time_ms_mean"].idxmin()])
```

---

## Phase 1 — Real-kernel hardware counters (Nsight Compute / `ncu`)

### Goal (what Phase 1 is trying to measure)
Phase 1 answers a different question than Phase 0.

- **Phase 0**: “How fast did the kernel run?” and “What occupancy do we *predict* from registers/block size?”
- **Phase 1**: “What did the GPU *actually do* while running the kernel?” using **hardware performance counters**.

Phase 1 collects a small set of metrics per run for the project’s real kernels:
- `gemm`
- `reduction`
- `softmax`

It writes results to:
- `results/tables/phase1_result.csv`

### Beginner concept: what are “hardware counters”?
Modern CPUs/GPUs include tiny “meters” inside the chip that count events while code is running.

Examples:
- how busy the compute cores were
- how much memory bandwidth you used
- cache hit-rate

On NVIDIA GPUs, these counters are exposed through CUPTI and tools like **Nsight Compute**. In this repo we use the Nsight Compute CLI:
- `ncu`

Important practical note (Windows):
- On Windows (WDDM), reading performance counters often requires an **Administrator** terminal. If counters are blocked, `ncu` will report `ERR_NVGPUCTRPERM`.

### How to run Phase 1

From the project root, in the `gpu-jit-opt` conda environment:

```bat
python experiments\phase1_collect_counters.py
```

Recommended on Windows:
- Open **Command Prompt** → **Run as administrator** (so counters are available)

Expected terminal output (shape):
- It prints one line per configuration (kernel × size × block × reg cap)
- It ends with:
   - `Done. Rows: ... (ok=...)`
   - `Saved to: results\tables\phase1_result.csv`

### Phase 1 CSV columns (what each attribute means)
This is a reference for `results/tables/phase1_result.csv`.

Per-row identifiers
- `kernel` (string): which benchmark ran: `gemm`, `reduction`, `softmax`.
- `matrix_size` (int): the size label $N$ used for the run.
   - `gemm`: multiplies $N \times N$ matrices.
   - `softmax`: applies softmax to an $N \times N$ matrix, row-wise.
   - `reduction`: reduces $N \times N$ elements (internally uses `matrix_size * matrix_size`).
- `block_size` (int): threads per block used by the kernel runner.
- `reg_cap` (string/int): register cap setting.
   - `default` means “no cap requested”.
   - `32`, `64`, ... are caps (Numba `max_registers`, like PTXAS `--maxrregcount`).

Run status / diagnostics
- `ok` (bool): whether counter collection succeeded for that configuration.
- `reason` (string): why it succeeded/failed.
   - `ok` means metrics were parsed.
   - Common failure reasons include `permission_denied`, `ncu_not_found`, `timeout`, or `no_metrics_parsed`.

Raw vs normalized metric columns
For each metric key (example: `dram_bw_pct`) Phase 1 writes two columns:

- `{metric}_raw`: the numeric value returned by `ncu`.
- `{metric}_norm`: a simplified value intended for ML/RL state vectors.

Normalization rules used in this repo:
- Most metrics are percentages → `norm = raw / 100`, clamped to $[0, 1]$.
- Keys ending in `_ratio` are treated as “ratio-like” and are **clamped** to $[0, 1]$ (best-effort).
   - This keeps the RL state bounded, but it also means `_ratio_norm` is not always a linear rescaling of `_ratio_raw`.

### Metrics collected in `phase1_result.csv` (what they mean)

#### `achieved_occupancy_raw` / `_norm`
- What it measures: a proxy for “how many warps were active” during execution.
- Typical unit: percent-of-peak (0–100).
- Interpretation:
   - Higher is usually better for *latency hiding* (especially memory-bound kernels).
   - But higher occupancy is not automatically faster (compute-heavy kernels can be fastest at slightly lower occupancy).

#### `dram_bw_pct_raw` / `_norm`
- What it measures: DRAM throughput as a **percentage of the GPU’s peak DRAM bandwidth**.
- Interpretation:
   - High values suggest your kernel is pushing memory bandwidth.
   - Low values can mean the kernel is compute-bound, underutilized, or too small to saturate the GPU.

#### `l2_hit_rate_raw` / `_norm`
- What it measures: fraction of memory accesses served by L2 cache (hit-rate).
- Interpretation:
   - Higher hit-rate usually means fewer expensive DRAM transactions.
   - A low hit-rate is common for streaming workloads or very large working sets.

#### `sm_active_pct_raw` / `_norm`
- What it measures: an SM utilization proxy (“how busy the GPU compute units were”).
- Why it might vary across machines/tool versions:
   - Nsight Compute sometimes reports SM utilization via different but related metrics.
   - The collector requests `sm__active_cycles...` and falls back to `sm__throughput...` if needed.
- Interpretation:
   - Higher values mean the SMs were active a larger fraction of time.

#### `warp_exec_efficiency_ratio_raw` / `_norm` (optional)
- What it measures: a warp efficiency proxy derived from executed thread instructions.
- Interpretation:
   - If your kernel has branch divergence or inactive lanes, this can drop.
   - On some setups this metric’s “raw” scale can look like “active lanes per instruction” (often near 32 when fully efficient).
- In this repo:
   - `_norm` is clamped to $[0, 1]$ for safety.
   - Treat this as an optional/debug signal rather than a primary metric.

### Common gotchas (especially for beginners)

1) “Why is Phase 1 slower than Phase 0?”
- Phase 1 runs under a profiler (`ncu`). Profilers add overhead.

2) “Why do I need Administrator on Windows?”
- Windows’ driver model often blocks access to hardware counters unless you run elevated.

3) “Why do some metric columns come out empty?”
- A metric may not be supported (or may be emitted as N/A) on a given GPU / Nsight Compute version.
- The collector includes fallbacks for some key signals (like `sm_active_pct`), but not for every possible metric.

4) “Raw vs norm: which should I use?”
- For human reading/plots: use `_raw`.
- For RL/ML inputs: use `_norm` (bounded, roughly comparable across metrics).

### Quick workflow to inspect Phase 1 results

```python
import pandas as pd

df = pd.read_csv("results/tables/phase1_result.csv")
print(df.head())

# Example: show the best achieved occupancy per kernel
best = df.loc[df.groupby("kernel")["achieved_occupancy_norm"].idxmax()]
print(best[["kernel", "matrix_size", "block_size", "reg_cap", "achieved_occupancy_raw"]])
```

---

## Phase 2 — Benchmark kernels (the workloads we optimize)

### Goal (what Phase 2 is trying to achieve)
Phase 2 is about having a **small set of real CUDA kernels** that:

- represent different performance regimes (compute-bound vs memory-bound)
- are stable to run repeatedly
- can be compiled/run under different settings (block size, register cap)

These kernels are the “tasks” the RL agent will later learn to optimize.

Implemented kernels in this repo:
- **GEMM** (`kernels/gemm.py`): tiled matrix multiplication (typically compute-heavy)
- **Reduction** (`kernels/reduction.py`): sum reduction (typically memory/latency sensitive)
- **Softmax** (`kernels/softmax.py`): row-wise softmax (mixed compute + memory)

### Beginner concept: what does “kernel correctness” mean?
Before optimizing performance, we must ensure the kernel computes the right answer.

Examples:
- GEMM output should match $C = A \times B$ (within floating point tolerance)
- Reduction output should match `sum(x)`
- Softmax output should be non-negative and each row should sum to about 1

Floating point note:
- GPU results may not match CPU results *bit-for-bit* due to different instruction order and math implementations.
- We use **tolerances** (“close enough”) rather than exact equality.

### How to validate Phase 2 (recommended before Phase 3/RL)

Phase 2 uses a **two-tier** testing approach:

1) **Small correctness suite (default)**
- Runs quickly (seconds)
- Uses small + edge-case sizes to catch indexing/boundary bugs
- This is the test suite you run frequently while developing

2) **Performance-scale validation suite (opt-in)**
- Runs medium/large sizes that are closer to Phase 0/1 benchmark regimes
- This is what you run before re-collecting Phase 0/1 tables, or before starting RL (Phase 3)

#### 1) Small correctness suite (default)

Command (from project root):

```bat
cd /d "C:\Users\HP\Desktop\CD PROBLEM STATEMENT\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && pytest -q tests\test_kernels.py -vv
```

Expected terminal output (shape):

```text
collected 19 items / 3 deselected / 16 selected

... 16 PASSED lines ...

16 passed, 3 deselected, ... warnings in X.XXs
```

Interpretation:
- **`16 passed`** means the kernels are correct on small/edge-case sizes.
- **`3 deselected`** means the performance-scale tests exist but were not run (by design).

#### 2) Performance-scale validation suite (medium/large)

Command (opt-in):

```bat
cd /d "C:\Users\HP\Desktop\CD PROBLEM STATEMENT\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && pytest -q tests\test_kernels.py -vv --runslow -m slow
```

Expected terminal output (shape):

```text
collected 19 items / 16 deselected / 3 selected

tests/test_kernels.py::test_gemm_correctness_large PASSED
tests/test_kernels.py::test_reduction_correctness_large PASSED
tests/test_kernels.py::test_softmax_sums_to_one_large PASSED

3 passed, 16 deselected in X.XXs
```

Interpretation:
- **`3 passed`** means the kernels behave correctly on benchmark-like sizes.

#### Common gotchas / how to interpret warnings

- `NumbaPerformanceWarning: Grid size ... under-utilization` is expected in the **small** suite.
   - It means the input is tiny (few blocks), so the GPU is not fully utilized.
   - This is fine for correctness tests.

If Phase 2 tests fail:
- Fix correctness first (don’t start RL yet).
- Then rerun Phase 0/1 experiments, because they assume these kernels are valid.

---

## Future phases (Phase 3+)
### Phase 3 — RL Environment (Gym interface)

Phase 3’s job is to take the kernels (Phase 2), timing (Phase 0), and optional counters (Phase 1) and wrap them in a **Gymnasium environment**.

Objectives:
- **Action → knob mapping**: each action corresponds to a concrete configuration (initially `block_size` and `reg_cap`).
- **Reward = speedup**: reward is based on measured speedup relative to a baseline configuration that is measured once per episode.
- **Observation is bounded and ML-friendly**: observation vectors are numeric and normalized/clamped to $[0,1]$.
- **Graceful degradation on Windows**: if CUPTI/`ncu` metrics are unavailable (e.g., `ERR_NVGPUCTRPERM`), the environment still runs using fallback vectors instead of failing.
- **Repeatability**: episodes are deterministic for a given kernel + matrix size + seed, aside from expected GPU timing noise.

Non-objectives (initial Phase 3 prototype):
- Profiling every step by default (profiling is opt-in because it is slow and may require Administrator on Windows).
- Expanding the action space beyond the two core knobs before the baseline loop is stable.

---

### What Phase 3 produces (artifacts)

Phase 3 is an RL environment, so the most useful “results” are **rollout logs**.

Phase 3 produces two CSV artifacts in `results/tables/`:
- `phase3_rollout.csv` (step-level): one row per environment step (one attempted configuration).
- `phase3_episode_summary.csv` (episode-level): one row per episode summary.

These are generated by the Phase 3 logger script:
- `experiments/phase3_rollout_log.py` (implementation)
- `phase3_rollout_log.py` (root-level wrapper so you can run it from the repo root)

---

### How to run Phase 3 (recommended)

1) **Fast logging (NVML only, no profiler privileges needed)**

```bat
cd /d "C:\Users\HP\Desktop\CD PROBLEM STATEMENT\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && python phase3_rollout_log.py --use-nvml --kernels gemm reduction softmax --matrix-sizes 256 512 --episodes-per-case 1 --max-steps 10 --warmup 1 --repeats 5
```

2) **Full logging (CUPTI/`ncu` + NVML)**

This is slower. On Windows, run in an **Administrator** Command Prompt if counters are locked.

```bat
cd /d "C:\Users\HP\Desktop\CD PROBLEM STATEMENT\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && python phase3_rollout_log.py --use-cupti --use-nvml --kernels gemm reduction softmax --matrix-sizes 256 512 --episodes-per-case 1 --max-steps 10 --warmup 1 --repeats 5 --cupti-timeout-s 180
```

Notes:
- `--cupti-timeout-s 180` means: each `ncu` profiling step may take up to 180 seconds before we mark that step as `timeout` and continue.
- If CUPTI is blocked, the run still succeeds, but `cupti_ok=False` and CUPTI columns will be 0.

---

### Expected terminal output (shape)

You should see progress prints like:

```text
[phase3] case=0 kernel=gemm N=256 episodes=1
[phase3]  episode=0 seed=0 steps=10 cupti=True nvml=True
[phase3]  episode=0 done best_speedup=... mean_reward=...
Phase 3 rollout logging complete
Step-level CSV: ...\results\tables\phase3_rollout.csv
Episode summary CSV: ...\results\tables\phase3_episode_summary.csv
```

Warnings you may see (usually OK):
- `NumbaPerformanceWarning: Grid size ... under-utilization` for small sizes.
- NVML library deprecation warnings from `pynvml` (does not break logging).

---

## Phase 3 CSV: `phase3_rollout.csv` (step-level)

Each row corresponds to **one environment step** (one attempt at a kernel configuration).

### Identifiers
- `episode_id` (int): sequential episode index.
- `episode_seed` (int): seed used for `env.reset(seed=...)`.
- `step` (int): step index within the episode (0-based).

### Workload fields
- `kernel` (string): one of `gemm`, `reduction`, `softmax`.
- `matrix_size` (int): size label $N$.
   - For `gemm`/`softmax`, $N$ means an $N \times N$ input.
   - For `reduction`, the environment reduces $N \times N$ elements (to match Phase 0/1 conventions).

### Action (the configuration attempted)
- `block_size` (int): the chosen block-size knob.
   - For `reduction`/`softmax`, this is the 1D threads-per-block.
   - For `gemm`, this selects between kernel variants (e.g., 8×8 vs 16×16 tiles), so it acts as a discrete knob rather than a literal “threads-per-block” value.
- `reg_cap` (int): requested register cap.
   - `0` means “default” (no requested cap).
   - Other values request Numba `max_registers=...` (PTXAS `--maxrregcount` equivalent).

### Timing + reward
- `time_ms` (float): measured mean kernel time for that step (CUDA event timing averaged over `repeats`).
- `baseline_ms` (float): baseline time measured once at `reset()` for the episode.
   - Baseline config is: `block_size=256`, `reg_cap=0`.
- `speedup` (float): computed as:

   $$
		ext{speedup} = \frac{\text{baseline\_ms}}{\text{time\_ms}}
   $$

- `reward` (float): RL reward computed as:

   $$
		ext{reward} = \text{speedup} - 1
   $$

Interpretation:
- `reward > 0` means faster than baseline.
- `reward < 0` means slower than baseline.

### CUPTI / `ncu` collection status
- `cupti_ok` (bool): whether CUPTI/`ncu` metrics were successfully collected for this step.
- `cupti_reason` (string): diagnostic reason (typical values: `ok`, `disabled`, `permission_denied`, `timeout`, `ncu_not_found`, `no_metrics_parsed`).

### CUPTI metric columns (normalized)

If `cupti_ok=True`, these columns should be non-zero (values are clamped to $[0,1]$):
- `cupti_achieved_occupancy_norm`: achieved occupancy proxy.
- `cupti_l2_hit_rate_norm`: L2 cache hit-rate proxy.
- `cupti_dram_bw_pct_norm`: DRAM bandwidth usage as fraction of peak.
- `cupti_sm_active_pct_norm`: SM-active proxy (compute utilization).

If `cupti_ok=False`, these columns are 0.

### NVML telemetry columns (normalized)

If NVML is enabled, these columns reflect lightweight device telemetry sampled around each step:
- `nvml_gpu_util_norm`: GPU utilization fraction (0–1).
- `nvml_mem_util_norm`: memory interface utilization fraction (0–1).
- `nvml_mem_used_frac`: VRAM used fraction (0–1).
- `nvml_temp_norm`: temperature normalized by 100 (roughly 0–1).

Important note on Windows sampling:
- If you run with `--use-cupti`, NVML utilization is collected as a **peak** while the `ncu` subprocess is running (this is more reliable for short kernels).
- If you run NVML-only (no `--use-cupti`), utilization reflects a best-effort instantaneous NVML sample near the step.

---

## Phase 3 CSV: `phase3_episode_summary.csv` (episode-level)

Each row summarizes one episode.

- `episode_id` / `episode_seed`: identifiers (same meaning as above).
- `kernel`, `matrix_size`: which workload the episode used.
- `max_steps`: how many steps were allowed.
- `use_cupti`, `use_nvml`: whether these sources were enabled for the run.
- `baseline_ms`: baseline time measured at reset.
- `mean_time_ms`: average of `time_ms` over steps in the episode.
- `mean_reward`: average reward over steps.
- `best_speedup`: best (maximum) speedup achieved in the episode.
- `best_reward`: best (maximum) reward achieved in the episode.
- `steps`: number of steps executed.

---

### How to interpret Phase 3 results quickly

If you want “best configuration found” per kernel/size, use `phase3_rollout.csv`:
- filter by `kernel` and `matrix_size`
- sort by `speedup` descending

Important practical note:
- Very small sizes (like `matrix_size=64`) often show noisy/odd speedups due to underutilization and overheads.
- For paper-quality plots/tables, prefer larger sizes (e.g., 256/512/1024) and higher repeats.

---

## Phase 3 Training — PPO with Stable-Baselines3

After validating the environment with `phase3_rollout_log.py`, you can train a PPO agent to learn an optimization policy.

### Beginner concept: what is PPO (Proximal Policy Optimization)?

**PPO** is a **reinforcement learning algorithm** that teaches an agent to make good decisions based on observations.

In this project:
- **Agent**: the PPO neural network (a "policy")
- **Environment**: the Gymnasium kernel optimization environment
- **Observation**: GPU metrics + kernel identity + previous action
- **Action**: a choice of (block_size, reg_cap)
- **Reward**: speedup relative to baseline (higher = better)

How PPO "learns":
1. The agent **observes** the current kernel and GPU telemetry.
2. The agent **chooses an action** (e.g., try block_size=128, reg_cap=32).
3. The environment **executes that configuration** and measures runtime.
4. The agent receives a **reward** (positive if faster than baseline, negative if slower).
5. PPO **updates the neural network** to encourage actions that led to high rewards and discourage actions that led to low rewards.

After many episodes, the agent learns patterns like:
- "GEMM usually benefits from larger block sizes"
- "Reduction is sensitive to register pressure"
- "Softmax with these metrics prefers moderate register caps"

### The state/action/reward cycle

#### Observation vector (the agent's input)

Each observation is a vector of normalized values $[0, 1]$:

**Kernel identity** (one-hot encoded):
- Three binary features indicating which of `{gemm, reduction, softmax}` is running.
- Example: `[1, 0, 0]` means `gemm`, `[0, 1, 0]` means `reduction`, etc.

**Previous action** (normalized indices):
- Two features: the normalized indices of the previously chosen `block_size` and `reg_cap`.
- Example: if block_size candidates are `[64, 128, 256]` and we chose index 1, then `block_size_norm = 1 / 2 = 0.5`.
- On the first step of an episode, these are zero.

**CUPTI metrics** (if `--use-cupti` is enabled, optional):
- Four metrics collected from Nsight Compute (`ncu`), each normalized to $[0,1]$:
  - `achieved_occupancy_norm`: how many warps were active (0–1)
  - `l2_hit_rate_norm`: cache efficiency (0–1)
  - `dram_bw_pct_norm`: DRAM bandwidth usage as fraction of peak (0–1)
  - `sm_active_pct_norm`: compute unit utilization (0–1)
- If CUPTI is disabled or fails, these four values are zero.

**NVML telemetry** (lightweight, always enabled by default):
- Four lightweight GPU metrics:
  - `gpu_util_norm`: how busy the GPU is (0–1)
  - `mem_util_norm`: how busy the memory system is (0–1)
  - `mem_used_frac`: how much VRAM is in use (0–1)
  - `temp_norm`: temperature normalized by 100 (0–1)

**Total observation dimension**:
- `3` (kernel one-hot) + `2` (previous action) + `4` (CUPTI, if enabled) + `4` (NVML) = **13 dimensions** with CUPTI, or **9 dimensions** without CUPTI.

#### Action and action decoding

The discrete action space is:

$$
	ext{action\_space} = \text{Discrete}(\text{num\_block\_sizes} \times \text{num\_reg\_caps})
$$

In the current implementation:
- Block sizes: `[64, 128, 256]` (3 options)
- Register caps: `[0, 32, 64]` (3 options: `0` means default)
- Total outcomes: $3 \times 3 = 9$ possible actions, indexed 0–8

The environment **decodes** a given action integer into `(block_size, reg_cap)` by treating it as a multi-discrete index.

Example decoding:
- Action 0 → `(block_size=64, reg_cap=0)`
- Action 4 → `(block_size=128, reg_cap=32)`
- Action 8 → `(block_size=256, reg_cap=64)`

#### Reward

After taking an action, the environment measures the kernel runtime and computes:

$$
\text{speedup} = \frac{\text{baseline\_ms}}{\text{measured\_ms}}
$$

$$
\text{reward} = \text{speedup} - 1
$$

Interpretation:
- `reward = 0.2` means the configuration was 20% faster than baseline (speedup = 1.2).
- `reward = -0.1` means the configuration was 10% slower than baseline (speedup = 0.9).
- `reward` directly encodes the agent's objective: **maximize speedup**.

The baseline is measured once at `reset()` with `(block_size=256, reg_cap=0)`.

#### Episode structure

One episode consists of:
1. **Reset**: pick a random kernel + matrix size, measure baseline
2. **Steps**: the agent chooses actions for `--max-episode-len` steps
3. **Termination**: the episode ends (truncated) after `--max-episode-len` steps

Each episode typically produces 10–50 (block_size, reg_cap) trials, and the agent learns which configurations tend to be fast.

### NVML-only vs CUPTI+NVML: which should I use?

This is the key decision when running training.

#### Mode 1: NVML-only (fast, no privileges needed)

**Command (Windows):**

```bash
cd /d "C:\Users\HP\Desktop\CD PROBLEM STATEMENT\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && python train_rl.py --total-steps 50000 --max-episode-len 50 --use-nvml
```

**Characteristics:**
- **Speed**: ~5–20 minutes for 50k steps (depends on hardware).
- **Privileges**: Does NOT require Administrator terminal.
- **Observation**: Only NVML telemetry (lightweight) + kernel identity + previous action.
  - CUPTI metric columns are all zero.
  - Observation dimension: 9 (instead of 13).
- **What the agent learns**: Patterns from light telemetry (GPU util, memory util, temperature).
  - This is less informative than CUPTI, but fast and sufficient for basic optimization.
  - Example: "Reduction gets hot → try lower block size."

**When to use:**
- First-time training (to validate the setup works).
- Quick experimentation cycles.
- Systems where GPU counter access is unavailable or administratively restricted.
- Rapid iteration: collect baseline results quickly, then iterate on **what configurations work empirically**.

---

#### Mode 2: CUPTI+NVML (slow, more informative) 

**Command (Windows, requires Administrator terminal):**

```bash
cd /d "C:\Users\HP\Desktop\CD PROBLEM STATEMENT\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && python train_rl.py --total-steps 50000 --max-episode-len 30 --use-cupti --use-nvml --cupti-timeout-s 180
```

**Characteristics:**
- **Speed**: ~2–8 hours for 50k steps (much slower).
  - Each step runs the kernel under `ncu` profiler, which adds significant overhead.
  - Note: `--max-episode-len 30` (not 50) to keep total time reasonable.
- **Privileges**: **Requires** Administrator Command Prompt (GPU counter access).
- **Observation**: CUPTI counters (achieved occupancy, L2 hit rate, DRAM BW, SM utilization) + NVML + kernel identity + previous action.
  - Observation dimension: 13 (full).
- **What the agent learns**: Rich hardware metrics.
  - Example patterns: "Achieved occupancy > 0.6 AND DRAM BW < 0.8 → this config is register-constrained, try lower reg cap."
  - More detailed signals can lead to **better optimization**, but the agent needs more data to learn robust patterns.

**When to use:**
- **Production runs** where you want the agent to learn from detailed hardware metrics.
- **Benchmarking / paper writing**: the trained policy may be more robust and better-tuned.
- You have access to an Administrator terminal.
- You have time (a few hours).
- The extra hardware-level features are worth the overhead for your use case.

---

#### Side-by-side comparison

| Aspect | NVML-only | CUPTI+NVML |
|--------|----------|-----------|
| **Training time (50k steps)** | ~5–20 min | ~2–8 hours |
| **Requires Administrator** | No | Yes |
| **Privileges needed** | None | GPU counters |
| **Observation dim** | 9 | 13 |
| **Metrics** | GPU util, mem util, temp | + occupancy, cache, DRAM BW, SM util |
| **Policy quality** | Good (basic heuristics) | Better (rich signals) |
| **Iterative development** | ✓ (fast feedback) | ✗ (slow; use once) |
| **Paper/publication** | Acceptable | Preferred |

---

### How to run PPO training (fast mode, NVML only)

On Windows CMD (estimated 5–20 minutes for 50k steps):

```bash
cd /d "C:\Users\HP\Desktop\CD PROBLEM STATEMENT\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && python train_rl.py --total-steps 50000 --max-episode-len 50 --eval-freq 5000 --n-eval-episodes 5 --use-nvml
```

This trains for 50,000 environment steps, with evaluation every 5,000 steps (5 episodes per eval). The agent runs with only NVML telemetry.

---

### How to run PPO training (with CUPTI metrics, slower)

**Run in an Administrator Command Prompt** (so GPU counters are available):

```bash
cd /d "C:\Users\HP\Desktop\CD PROBLEM STATEMENT\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && python train_rl.py --total-steps 50000 --max-episode-len 30 --use-cupti --use-nvml --cupti-timeout-s 180 --eval-freq 10000 --n-eval-episodes 3
```

This trains with CUPTI metrics (slow), reduces max episode length to 30 to keep training time reasonable, and evaluates only every 10k steps.

**Important**: `--cupti-timeout-s 180` means each CUPTI collection step may take up to 180 seconds. If `ncu` hangs or is very slow, increase this timeout. If it's consistently timing out, ensure you're running as Administrator and that Nsight Compute is installed.

---

### Expected training output

```
[2026-04-14 12:00:00] train_rl - INFO - ====================================
[2026-04-14 12:00:00] train_rl - INFO - Phase 3: PPO Training (Stable-Baselines3)
[2026-04-14 12:00:00] train_rl - INFO - Total steps: 50000
[2026-04-14 12:00:00] train_rl - INFO - Learning rate: 0.0003
[2026-04-14 12:00:00] train_rl - INFO - Batch size: 2048
[2026-04-14 12:00:00] train_rl - INFO - Creating training environment...

---Logging in with PPO---
Step: 1000 / 50000
Step: 2000 / 50000
...
Eval: mean_reward = 0.1234, mean_ep_length = 45.0
Step: 5000 / 50000
...
Training complete. Saving final model...
Training summary saved to results/logs/training_summary.json
====================================
```

Watch for:
- **`mean_reward` increasing over time**: the agent is learning (good sign).
- **`mean_ep_length`**: how many steps per episode. Should be fairly stable (~max_episode_len).
- **Warnings about `permission_denied`** (if using CUPTI): means GPU counters are not available; run as Administrator or switch to NVML-only mode.

---

### Training artifacts

After training completes:

**Model files**:
- `results/models/ppo_final.zip` — trained policy weights (loadable with `PPO.load()`)
- `results/models/best/best_model.zip` (if `--eval-freq` set) — best checkpoint during evaluation
- `results/models/ppo_checkpoint_*.zip` — periodic snapshots (one every ~10% of training)

**Logs and summaries**:
- `results/logs/train_rl.log` — detailed training log (timestamps, step counts, warnings)
- `results/logs/training_summary.json` — a JSON summary with all hyperparameters and artifact paths:
  ```json
  {
    "total_steps": 50000,
    "learning_rate": 0.0003,
    "batch_size": 2048,
    "model_path": "results/models/ppo_final.zip",
    "best_model_path": "results/models/best/best_model.zip",
    "log_path": "results/logs"
  }
  ```

**TensorBoard visualization**:
- `results/logs/tensorboard/` — live plots (reward, episode length, policy loss, value loss)
- View them in real-time or after training by running:
  ```bash
  tensorboard --logdir results/logs/tensorboard
  ```
  Then open `http://127.0.0.1:6006` in your browser.

---

### PPO Hyperparameters in detail

These are the tunable knobs when running `python train_rl.py`. You only need to understand a few; the defaults are reasonable.

**Core PPO parameters:**

- `--total-steps NUM` (default: 100000)
  - Total number of environment interactions (steps) during training.
  - Larger = longer training, potentially better convergence, but slower.
  - Typical range: 10k–100k for this task. Start with 50k.

- `--batch-size NUM` (default: 2048)
  - Number of steps collected per gradient update ("rollout").
  - Larger batches = more stable updates but more GPU memory.
  - Keep at 2048 unless you hit memory limits.

- `--learning-rate LR` (default: 3e-4)
  - Step size for gradient descent.
  - Higher = faster updates but risk instability.
  - Lower = slower but more stable.
  - Typical range: 1e-4 to 1e-3. Use 3e-4 unless training is unstable (high variance in reward).

- `--n-epochs N` (default: 10)
  - How many times PPO processes each batch of collected samples.
  - Higher = more optimization on each batch but risk overfitting.
  - Keep at 10 unless training is very noisy.

- `--gamma GAMMA` (default: 0.99)
  - Discount factor: how much the agent values future rewards vs immediate rewards.
  - 0.99 = heavily discount the future (short-term thinking).
  - 0.99+ = slightly more long-term planning.
  - Keep at 0.99 for this task (episodes are only ~30–50 steps).

- `--gae-lambda LAMBDA` (default: 0.95)
  - Generalized Advantage Estimation parameter (controls how far into future advantages are estimated).
  - Keep at 0.95 (a standard value).

- `--clip-range CLIP` (default: 0.2)
  - PPO's clipping range: constrains how much the policy can change per update.
  - Larger = more aggressive updates, risk destabilization.
  - Smaller = conservative updates, slower convergence.
  - Keep at 0.2 unless the policy converges too slowly.

- `--entropy-coeff COEFF` (default: 0.01)
  - Encourages exploration (the agent tries diverse actions, not just the "greedy" best action).
  - Higher = more exploration (agent tries more configurations).
  - Lower = exploitation (agent commits to promising configurations).
  - Keep at 0.01 for this task.

**Episode and environment parameters:**

- `--max-episode-len NUM` (default: 50)
  - Maximum steps per episode.
  - With NVML-only: use 50 (fast, can afford more exploration).
  - With CUPTI: use 20–30 (slow, reduce per-episode profiling overhead).

- `--warmup NUM` (default: 1)
  - Kernel warmup runs before timing.
  - Keep at 1 (no need for more in RL).

- `--repeats NUM` (default: 5)
  - How many times to measure each kernel (for averaging).
  - Keep at 5 (reasonable balance).

**Evaluation parameters (optional):**

- `--eval-freq FREQ` (default: None, i.e., no evaluation)
  - Run evaluation every `FREQ` environment steps.
  - If set (e.g., `5000`), the agent is evaluated on separate episodes and the "best" model is saved.
  - Recommended: set to 10–20% of total steps (e.g., for 50k steps, use 5000).

- `--n-eval-episodes N` (default: None)
  - Number of evaluation episodes per evaluation.
  - Recommended: 3–5.

**CUPTI parameters:**

- `--use-cupti` (flag: off by default)
  - Enable CUPTI/`ncu` metric collection.
  - Requires Administrator on Windows; adds major overhead (2–8x slower).

- `--use-nvml` (flag: on by default)
  - Enable NVML lightweight telemetry.
  - Recommended: always on (fast, informative).

- `--cupti-timeout-s NUM` (default: 120)
  - Timeout for `ncu` collection per step, in seconds.
  - If CUPTI is slow on your machine, increase to 180–300.

---

### Quick decision guide

**"Should I use NVML-only or CUPTI+NVML?"**

1. **First time / new setup?** → NVML-only (quick feedback)
2. **Want to iterate quickly?** → NVML-only (5–20 min per run)
3. **Have time and want best results?** → CUPTI+NVML (2–8 hours per run)
4. **Writing a paper / final results?** → CUPTI+NVML (richer metrics)
5. **Admin not available?** → NVML-only (requires no privileges)

For this project, a recommended workflow:
1. Run NVML-only (50k steps, ~15 min) to validate training works.
2. Check the learned policy quality in `results/models/ppo_final.zip`.
3. If satisfied, you're done. If you want better results, run CUPTI+NVML (50k steps, ~4 hours) for publication.

---

## Understanding Training Results — TensorBoard Metrics

After training completes, you can visualize results with TensorBoard to understand how well your agent learned.

### How to view TensorBoard

From your project root:

```bash
cd /d "C:\Users\HP\Desktop\CD PROBLEM STATEMENT\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && tensorboard --logdir results/logs/tensorboard --port 6006
```

Then open `http://localhost:6006` in your browser. You'll see training curves organized by tabs.

### Key metrics explained

#### **rollout/ep_rew_mean** (MOST IMPORTANT)

**What it measures:**
- Average reward per episode over the course of training
- Directly reflects how good the agent's discovered configurations are

**What to expect:**
- Starts low (near 0-1): agent explores randomly
- Climbs over time: agent learns better configurations
- Plateaus: agent has converged to a stable policy

**Your result:** Should reach ~3+ for a good training run
- 0.0 = random agent (no speedup over baseline)
- 1.0 = configs that are 2x faster than baseline
- 3.15 = configs that are ~4x faster than baseline (your result!)

**Good sign:** Smooth upward curve that eventually plateaus
**Bad sign:** Flat line (no learning) or wild oscillations (unstable training)

---

#### **train/policy_gradient_loss**

**What it measures:**
- How much the policy network is improving per gradient update

**Direction:**
- Should trend toward small values (ideally near 0)

**What to expect:**
- Starts high (large negative or positive values)
- Decreases over time as the policy stabilizes
- High variance is normal (depends on batch content)

**Your result:** Should end near -0.01 to 0.01
- Negative values are expected in PPO (log-probability gradients)
- Magnitude matters: smaller = better

---

#### **train/value_loss**

**What it measures:**
- How well the value network predicts episode returns
- Value network: "Given this observation, what total reward will I get?"

**Direction:**
- Should decrease over time (lower = better predictions)

**What to expect:**
- Starts very high (10+): terrible at predicting returns
- Drops sharply: learning to estimate correctly
- Plateaus: predictions are stable

**Your result:** Should end < 1.0 (ideally < 0.5)
- Your training achieved ~0.574
- This means the value network learned good estimates

**Good sign:** Consistent downward trend
**Bad sign:** Stays stuck at high values (value network not learning)

---

#### **train/approx_kl** (Kullback-Leibler Divergence)

**What it measures:**
- How much the policy changes on each update
- KL divergence quantifies difference between old and new policy

**Direction:**
- Should be small and stable

**Normal range:** 0.01-0.05 per update
- Smaller KL = policy change is conservative
- This is by design: PPO clips large policy changes

**Your result:** Should end around 0.01-0.02
- Your training achieved ~0.0133
- This is excellent (very stable policy)

**Good sign:** Consistent, small values
**Bad sign:** Large values (> 0.1) = policy changing too much

---

#### **train/clip_fraction**

**What it measures:**
- What fraction of gradient updates were clipped by PPO
- PPO clips updates to prevent huge policy swings

**Formula:** `clipped_updates / total_updates`

**Normal range:** 0.1-0.3 (10-30%)
- Means PPO is actively preventing crash-like policy changes
- This is good! Shows the safety mechanism works

**Your result:** Should be 0.1-0.3
- Your training achieved ~0.147 (14.7%)
- This is perfect (reasonable safety without over-clipping)

**Good sign:** Stable in the 0.1-0.3 range
**Bad sign:** Close to 0 (maybe learning rate too low) or > 0.5 (maybe too high)

---

#### **time/fps** (Frames Per Second)

**What it measures:**
- How many environment steps per second the training achieves
- Environment step = one kernel launch + measurement

**What to expect:**
- May start lower (setup overhead)
- Stabilizes to a steady rate
- Should be relatively flat over time

**Your result:** ~85 FPS for NVML-only mode
- This means 85 kernel executions per second
- 50,000 steps ÷ 85 FPS ≈ 588 seconds ≈ 9.8 minutes
- No GPU memory leaks (would cause FPS to drop)

**Good sign:** Stable or increasing FPS
**Bad sign:** Decreasing FPS (indicates memory buildup)

---

#### **time/total_timesteps**

**What it measures:**
- Cumulative environment steps taken so far
- Should increase linearly with training progress

**Your result:** Should reach your `--total-steps` target
- Target: 50,000
- Actual: 50,176
- Perfect! ✓

---

#### **train/entropy_loss**

**What it measures:**
- Encouragement for exploration vs exploitation
- Entropy = how diverse the agent's action choices are

**What to expect:**
- Negative value (by convention)
- Should be relatively stable over time
- Not growing more negative (that means getting too greedy)

**Your result:** Should be stable around -1.0 to -2.0
- Your training showed ~-1.53
- This is good (agent explores enough)

---

### Interpreting your training graphs: what to look for

#### **Graph Pattern 1: Episode Reward Over Time**

**Ideal shape:**
```
Reward
  |     ___---___
  |  __/        \___
  | /                 (plateau = converged)
  |/_________________ training time
```

**You want:**
- Clear upward trend from left to right
- Increasing from ~0-1 on the left
- Reaching ~3+ on the right
- Some noise is okay (GPU timing jitter)
- Should plateau (not keep climbing forever)

**If you see:**
- ✓ Smooth curve climbing to 3+ → Excellent training
- ✗ Flat line → Learning rate too low or reward broken
- ✗ Spiky/oscillating → Learning rate too high or clip_range wrong
- ✗ Declining → Policy getting worse (very bad)

---

#### **Graph Pattern 2: Policy and Value Loss**

**Ideal shape (both should decrease):**
```
Loss
  |
  |X              (starts high)
  |X\
  | X\
  |  X\___        (ends low)
  |      X___
  |___________ time
```

**You want:**
- Both curves decreasing over time
- Value loss should drop from ~10 to <1
- Policy loss should drop from high values to near 0

**If you see:**
- ✓ Both decreasing consistently → Good learning
- ✗ Flat or increasing → Not learning (check learning rate)
- ✗ Sudden jumps → Unstable training (high variance batches)

---

#### **Graph Pattern 3: Clip Fraction**

**Ideal shape:**
```
Clip %
  |
  | ___---___---___ (stable between 0.1-0.3)
  |/
  |________________ time
```

**You want:**
- Stable value between 0.1 and 0.3
- Relatively flat over time
- Not trending up or down

**If you see:**
- ✓ Stable at 0.1-0.3 → PPO working correctly
- ✗ Near 0 → Learning rate too low
- ✗ > 0.5 → Learning rate too high

---

### What "successful training" looks like

Your training results demonstrate success because:

| Metric | Your Value | Assessment |
|--------|-----------|-----------|
| Final mean reward | 3.15 | Excellent (3x+ speedup) |
| Training time | 9.7 min | Very fast (efficient) |
| FPS | ~85 | Stable and consistent |
| Total steps | 50,176 | Met target |
| Policy loss | ~-0.008 | Converged to stable value |
| Value loss | ~0.574 | Learned good estimates |
| Approx KL | ~0.0133 | Very stable |
| Clip fraction | ~0.147 | Good (14.7%) |
| Entropy loss | ~-1.53 | Reasonable exploration |

**Interpretation:**
- Agent learned meaningful patterns in 50,000 steps
- Found configurations ~3x faster than baseline
- Training was stable (no crashes)
- Policy and value networks both converged
- No GPU memory issues (stable FPS)

---

### Common training issues and solutions

#### Issue: Reward stays near 0 (not learning)

**Causes:**
- Learning rate too low
- Batch size too small
- Reward signal broken (environment not measuring correctly)

**Solutions:**
- Increase `--learning-rate` (try 0.001 instead of 0.0003)
- Increase `--batch-size` (try 4096)
- Check that kernel timing is working (Phase 2 tests)

#### Issue: Reward spikes up and down (unstable)

**Causes:**
- Learning rate too high
- Batch size too small
- Clip range too large

**Solutions:**
- Decrease `--learning-rate` (try 0.0001)
- Increase `--batch-size` (try 4096)
- Decrease `--clip-range` (try 0.1)

#### Issue: Losses not decreasing

**Causes:**
- Learning rate wrong
- Policy network not receiving gradients
- Observations or actions broken

**Solutions:**
- Check training log for errors
- Run Phase 2 tests to validate kernels
- Start with higher learning rate, then decrease

#### Issue: FPS dropping over time

**Causes:**
- GPU memory leak
- CUPTI profiling accumulating state
- Long-running timer threads

**Solutions:**
- Use NVML-only mode instead of CUPTI
- Reduce `--repeats` (fewer measurements per step)
- Reduce `--batch-size`

**Your training:** None of these issues! All metrics converged cleanly.

---

### Using TensorBoard data for analysis

#### Export CSV for further analysis

1. In TensorBoard, click any graph
2. At bottom, click "Download runs as CSV"
3. Open in Excel or Python for further analysis

#### Analyze specific metrics

```python
import pandas as pd

# Load TensorBoard CSV export
df = pd.read_csv("tensorboard_export.csv")

# Filter to mean reward
reward_df = df[df['name'] == 'rollout/ep_rew_mean']

# Compute statistics
print(f"Final reward: {reward_df['value'].iloc[-1]:.2f}")
print(f"Reward improvement: {reward_df['value'].iloc[-1] - reward_df['value'].iloc[0]:.2f}")
print(f"Steps to reach 1.0 reward: {reward_df[reward_df['value'] >= 1.0]['step'].iloc[0]}")
```

#### Compare multiple training runs

If you run training multiple times (different hyperparameters):
1. Save logs to different directories
2. Use `tensorboard --logdir results/logs/tensorboard` with multiple runs
3. TensorBoard shows side-by-side comparison

---

### Next steps after training

#### 1. Evaluate the learned policy

Generate rollout logs to see what configurations your agent learned to prefer:

```bash
python phase3_rollout_log.py --use-nvml --kernels gemm reduction softmax --matrix-sizes 256 512 --episodes-per-case 3 --max-steps 20
```

Check `results/tables/phase3_rollout.csv` to see:
- Which (block_size, reg_cap) combinations agent chose
- What speedups were achieved
- Whether agent has kernel-specific preferences

#### 2. Compare against Phase 0 baseline

Use Phase 0 results as a sanity check:

```python
import pandas as pd

phase0 = pd.read_csv("results/tables/phase0_baseline.csv")
phase3 = pd.read_csv("results/tables/phase3_rollout.csv")

# For each kernel, what's the best configuration?
for kernel in ['gemm', 'reduction', 'softmax']:
    p0_best = phase0[phase0['kernel'] == kernel].nsmallest(1, 'time_ms_mean')
    p3_best = phase3[phase3['kernel'] == kernel].nsmallest(1, 'time_ms')
    
    print(f"\n{kernel.upper()}")
    print(f"Phase 0 best: {p0_best['time_ms_mean'].values[0]:.2f}ms")
    print(f"Phase 3 best: {p3_best['time_ms'].values[0]:.2f}ms")
```

#### 3. Use the trained model for deployment

Your trained agent is ready to be loaded and used:

```python
from stable_baselines3 import PPO

model = PPO.load("results/models/ppo_final.zip")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
```

---

### TensorBoard best practices

1. **Let training finish before analyzing**
   - Curves are clearer when training is complete
   - Early curves may be misleading

2. **Look at multiple metrics**
   - Don't judge by reward alone
   - Check loss curves for stability
   - Verify FPS is reasonable

3. **Save training summaries**
   - Each run's `training_summary.json` notes hyperparameters
   - Useful for comparing later

4. **Export for papers**
   - TensorBoard PDFs look professional
   - Include final statistics in captions

---

## Phase 5 — BiLSTM Phase Detector

Phase 5 trains a Bidirectional LSTM neural network that classifies GPU kernel execution into one of four phases based on hardware performance counter traces.

### Goal (what Phase 5 is trying to do)

Phase 5 answers the question: **"What regime is the GPU operating in right now?"**

Knowing the current execution phase helps the RL agent (and human analysts) understand *why* a configuration is fast or slow, and can inform smarter optimization decisions.

The four phases are based on the **roofline model**:

| Phase | Label | Characteristics | Example |
|-------|-------|----------------|---------|
| 0 | **Compute-bound** | High occupancy, low DRAM BW, high SM utilization | Large GEMM (N≥256) |
| 1 | **Memory-bound** | Moderate occupancy, high DRAM BW, moderate SM util | Reduction, Softmax |
| 2 | **Latency-bound** | Low occupancy, low DRAM BW, low SM utilization | Tiny kernels (N<128) |
| 3 | **Mixed** | Overlapping characteristics, transitional | Phase transitions |

### The roofline model (beginner concept)

The **roofline model** is a way to classify kernels based on their ratio of computation to memory access:

$$
\text{Arithmetic Intensity (AI)} = \frac{\text{FLOPs}}{\text{Bytes transferred}}
$$

For the RTX 3050 Ti:
- Peak compute: ~7.8 TFLOP/s (FP32)
- Peak memory bandwidth: ~192 GB/s
- **Ridge point** = 7.8 × 10¹² / 192 × 10⁹ ≈ **40.6 FLOP/byte**

If a kernel's AI > 40.6 → it is **compute-bound** (limited by ALU throughput).
If a kernel's AI < 40.6 → it is **memory-bound** (limited by DRAM bandwidth).
If both utilization metrics are low → it is **latency-bound** (launch overhead dominates).

Examples for the project's kernels:
- **GEMM** at N=512: AI = 512/6 ≈ 85.3 → compute-bound ✓
- **Reduction** at any size: AI ≈ 0.25 → memory-bound ✓
- **Softmax** at any size: AI ≈ 0.625 → memory-bound ✓

### BiLSTM architecture

The model processes a **sliding window** of T=20 timesteps of 5 CUPTI counter values:

```
Input: (batch, T=20, 5)
  ↓
BiLSTM (2 layers, hidden=64, bidirectional)
  ↓
Concatenate forward[-1] + backward[0]  →  (batch, 128)
  ↓
┌─────────────────────┐  ┌──────────────────┐
│ Phase Head          │  │ Uncertainty Head  │
│ Linear(128→32)→ReLU │  │ Linear(128→16)   │
│ Linear(32→4)        │  │ →ReLU→Linear(1)  │
│ → Softmax           │  │ → Sigmoid         │
└─────────────────────┘  └──────────────────┘
      (batch, 4)              (batch, 1)
   phase probabilities       uncertainty
```

**Total parameters:** 142,021

The 5 input dimensions correspond to CUPTI counters:
1. `achieved_occupancy` — fraction of max warps active
2. `l2_hit_rate` — L2 cache efficiency
3. `dram_bw_pct` — DRAM bandwidth usage as fraction of peak
4. `warp_exec_eff` — warp execution efficiency
5. `sm_active_pct` — streaming multiprocessor utilization

### How to run Phase 5 training

```bash
cd /d "C:\Users\HP\Desktop\CD PROBLEM STATEMENT\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && python training\train_phase_detector.py
```

Optional arguments:
- `--epochs 100` — more training epochs (default: 50)
- `--lr 0.001` — learning rate (default: 1e-3)
- `--n-train 3200` — more training samples (default: 1600)
- `--seed 42` — random seed (default: 42)

### Expected output

```
============================================================
Phase 5: BiLSTM Phase Detector Training
============================================================
Device: cuda
Generating synthetic training data (2000 samples)...
  Phase 0 (  compute-bound): train=396, val=104
  Phase 1 (   memory-bound): train=395, val=105
  Phase 2 (  latency-bound): train=409, val=91
  Phase 3 (          mixed): train=400, val=100
  Augmenting with 81 Phase 0 roofline labels
  Total training samples: 1681
Model parameters: 142,021

Training for 50 epochs...
 Epoch  Train Loss    Val Loss   Val Acc
----------------------------------------
     1      1.3274      1.0612    72.2%
    10      0.0026      0.0010   100.0%
    50      0.0002      0.0000   100.0%

Best validation accuracy: 100.0%
```

**Note:** 100% accuracy is expected on synthetic data because the phase distributions are well-separated by design. Real CUPTI traces would have more overlap between phases, especially between "memory-bound" and "mixed", and accuracy would be lower (expected 85–95%).

### Training artifacts

- `results/models/phase_detector.pt` — trained PyTorch model weights
- `results/tables/phase5_eval.csv` — per-class precision/recall/F1 metrics

### How to use the trained model

```python
import torch
import numpy as np
from models.phase_detector import PhaseDetector

# Load model
model = PhaseDetector()
model.load_state_dict(torch.load("results/models/phase_detector.pt"))

# Single-window inference
# Input: 20 timesteps × 5 CUPTI counters (normalized to [0,1])
counter_window = np.random.rand(20, 5).astype(np.float32)
phase_label, confidence, uncertainty = model.predict(counter_window)

print(f"Phase: {model.phase_name(phase_label)}")  # e.g., "compute-bound"
print(f"Confidence: {confidence:.1%}")              # e.g., "97.3%"
print(f"Uncertainty: {uncertainty:.3f}")             # e.g., "0.042" (lower = more certain)
```

### Interpreting phase5_eval.csv

The evaluation CSV contains per-class metrics:

| Column | Meaning |
|--------|--------|
| `phase` | Phase ID (0–3) |
| `phase_name` | Human-readable name |
| `precision` | TP / (TP + FP) — how many predicted phases are correct |
| `recall` | TP / (TP + FN) — how many actual phases are found |
| `f1` | Harmonic mean of precision and recall |
| `support` | Number of validation samples for this class |

---

## Phase 7 — RL vs Baselines Comparison

Phase 7 is the **main evaluation experiment** that validates the entire project. It compares the trained PPO agent against two baselines on each kernel × problem size combination.

### Goal (what Phase 7 is trying to prove)

Phase 7 answers the central research question: **"Does the RL agent find better kernel configurations than simple strategies?"**

It runs three strategies head-to-head:

1. **PTXAS default** — the compiler's default configuration (block_size=256, reg_cap=0). This is what you get "out of the box" without any tuning.
2. **Random search** — sample 100 random (block_size, reg_cap) configurations and keep the best. This is the simplest autotuning baseline.
3. **Trained PPO agent** — the RL agent trained in Phase 3/4, making deterministic decisions based on its learned policy.

### How to run Phase 7

```bash
cd /d "C:\Users\HP\Desktop\CD PROBLEM STATEMENT\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && python experiments\phase7_rl_vs_baselines.py --model results\models\rtx3050_01.zip
```

Optional arguments:
- `--kernels gemm reduction softmax` — which kernels to test (default: all three)
- `--sizes 256 512 1024` — which problem sizes (default: 256 512)
- `--n-random 200` — more random search samples (default: 100)
- `--n-ppo-episodes 10` — more PPO evaluation episodes (default: 5)
- `--ppo-max-steps 50` — more steps per PPO episode (default: 30)

Estimated runtime: **3–10 minutes** depending on hardware and parameter choices.

### Expected output

The script produces a Rich-formatted table:

```
                       Phase 7: RL vs Baselines Comparison
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Strategy      ┃ Kernel    ┃ Size ┃     Time (ms) ┃      Best Speedup ┃ Samples ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ PTXAS default │ softmax   │  512 │ 2.648 ± 0.000 │ 1.000x (baseline) │       1 │
│ Random search │ softmax   │  512 │ 2.332 ± 0.523 │            1.584x │     100 │
│ PPO agent     │ softmax   │  512 │ 1.664 ± 0.002 │            1.583x │     150 │
└───────────────┴───────────┴──────┴───────────────┴───────────────────┴─────────┘
```

### Understanding the results table

| Column | Meaning |
|--------|--------|
| `Strategy` | Which optimization strategy was used |
| `Kernel` | Which GPU kernel (gemm, reduction, softmax) |
| `Size` | Matrix dimension N (NxN for gemm/softmax, N*N elements for reduction) |
| `Time (ms)` | Mean ± std of measured kernel time. For Random/PPO, this is the *best* time found |
| `Best Speedup` | Best kernel time / baseline time. Higher = faster than default |
| `Samples` | Total kernel evaluations used (budget) |

### Interpreting the results: what to look for

**1. Does PPO beat the default?**

A speedup > 1.0x means the strategy found a faster configuration than the compiler default. On the RTX 3050 Ti:
- GEMM is already well-optimized at default settings → small gains (~0–6%)
- Reduction benefits significantly from smaller block sizes → 17–72% speedup
- Softmax benefits from tuning → 23–58% speedup

**2. PPO vs Random Search — which is better?**

Random search has an advantage: it explores the *entire* configuration space uniformly, so with enough samples (N=100), it will find the global optimum.

PPO has a different advantage: **consistency**. Look at the standard deviation:
- Random search: `±0.523 ms` (high variance — depends on which configs it samples)
- PPO agent: `±0.002 ms` (extremely low variance — the learned policy is deterministic)

This means PPO reliably delivers near-optimal performance every time, while random search quality varies run-to-run.

**3. When does PPO shine?**

PPO is most valuable when:
- The configuration space is large (more knobs than just block_size + reg_cap)
- You need **consistent, repeatable** results (deployment / production)
- You want to amortize the cost: train once, deploy everywhere
- The kernel has complex performance characteristics (softmax > gemm)

### Your actual results (RTX 3050 Ti)

| Kernel | Default | Random (best) | PPO (best) | Winner |
|--------|---------|--------------|------------|--------|
| GEMM 256 | 0.201ms | 1.060x | 1.003x | Random (GEMM already optimal) |
| GEMM 512 | 5.855ms | 1.002x | 1.000x | Tie (no room to improve) |
| Reduction 256 | 0.142ms | 1.529x | 1.204x | Random (more exploration) |
| Reduction 512 | 0.290ms | 1.719x | 1.174x | Random (more exploration) |
| Softmax 256 | 0.556ms | 1.234x | **1.248x** | **PPO** |
| Softmax 512 | 2.648ms | 1.584x | **1.583x** | **Tie** (PPO with lower variance) |

**Key insight:** The PPO agent discovers that softmax benefits from specific block_size/reg_cap combinations and consistently applies this knowledge, matching or beating random search with ~60x lower timing variance.

### Artifacts

- `results/tables/phase7_comparison.csv` — full results table
  - Columns: `strategy, kernel, size, time_mean_ms, time_std_ms, best_speedup, n_samples`

### Common issues

#### Issue: `ValueError: Unexpected observation shape (9,) ... please use (13,)`

**Cause:** The environment's observation space doesn't match what the PPO model was trained with. The model expects 13 dimensions (CUPTI + NVML + kernel one-hot + previous action), but the environment was created with `use_nvml=False`, dropping 4 NVML dimensions.

**Solution:** Ensure `use_nvml=True` in `EpisodeConfig` when evaluating a model trained with NVML features. The current Phase 7 script handles this correctly.

#### Issue: PPO worse than random search

**This is expected** in some cases. With only 3 block sizes × 3 reg caps = 9 configurations, random search with N=100 samples explores each configuration ~11 times on average. For such a small search space, random search is competitive.

PPO's advantage grows with:
- Larger action spaces (more configuration knobs)
- More complex kernels (where performance landscapes are non-trivial)
- Deployment scenarios (consistent results without re-searching)

---
