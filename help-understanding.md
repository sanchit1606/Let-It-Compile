# Help Understanding Experiment Results

This doc explains how to interpret the experimental output in this project.

As of now, **Phase 0** and **Phase 1** are implemented and produce results, and **Phase 2** validation (kernel correctness tests) is implemented.

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

Add later phases here using the same structure:
- goal
- how to run
- expected terminal output
- what each column means
- common gotchas
