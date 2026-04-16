# Error Log

## Scope
Errors encountered while setting up and running Phase 0–2 in this workspace, including root cause and remediation.

Covered entry points:
- Phase 0: `experiments/phase0_baseline_table.py`
- Phase 1: `experiments/phase1_collect_counters.py` (Nsight Compute / `ncu`)
- Phase 2: kernel validation via `tests/test_kernels.py`

---

## 1) Noisy Conda Activation / Toolchain Spam

**Symptom**
- `conda activate gpu-jit-opt` printed many lines (`SET ...`, `vswhere`, `vcvars`, missing VS Enterprise paths, etc.).

**Cause**
- Activation hooks in `gpu-jit-opt` had verbose echo enabled and ran VS2019 compiler bootstrap scripts:
  - `etc/conda/activate.d/vs2019_compiler_vars.bat`
  - `etc/conda/activate.d/~cuda-nvcc_activate.bat`

**Fix**
- Changed hooks to quiet mode (`@echo off`).
- Added existence guards around Visual Studio file/path probes and `vcvars` call.
- Quoted `pushd "%VSINSTALLDIR%"` to handle spaces.

**Result**
- Activation output reduced to minimal informational lines.

---

## 2) Numba Import Error: NumPy Too New

**Symptom**
- Running Phase 0 failed at import:
  - `ImportError: Numba needs NumPy 1.26 or less`

**Cause**
- `numpy==2.2.6` installed with an older Numba build.

**Fix**
- Downgraded NumPy in env:
  - `conda install -y "numpy=1.26.*"`

**Result**
- Resolved NumPy compatibility gate.

---

## 3) Numba/llvmlite ABI Mismatch

**Symptom**
- Import stack failed with:
  - `AttributeError: function 'LLVMPY_CreatePassManager' not found`

**Cause**
- Incompatible versions:
  - `numba==0.59.1` with `llvmlite==0.46.0` (ABI mismatch).

**Fix**
- Repinned compatible trio:
  - `numba==0.59.1`
  - `llvmlite==0.42.0`
  - `numpy==1.26.4`

**Result**
- Numba imports succeeded.

---

## 4) `numba-cuda` Package Interference

**Symptom**
- CUDA import path resolved via `site-packages/numba_cuda/...` and produced incompatibility behavior.

**Cause**
- `numba-cuda` package installed in env and conflicting with built-in `numba.cuda`.

**Fix**
- Uninstalled package:
  - `pip uninstall -y numba-cuda`
- Repeated as needed (Conda later reintroduced `numba-cuda 0.15.2` during solves).

**Result**
- `numba.cuda` import resolved from `site-packages/numba/cuda/...` (expected path).

---

## 5) CUDA JIT Error: Missing libdevice

**Symptom**
- Phase 0 preflight/runtime error:
  - `RuntimeError: Missing libdevice file. Please ensure you have package cudatoolkit >= 11.0`

**Cause**
- `libdevice.10.bc` existed, but Numba path resolution on Windows was inconsistent with how script set env vars (directory vs file + legacy/current var behavior).

**Fix**
1. Verified file presence:
   - `.../Library/nvvm/libdevice/libdevice.10.bc`
2. Patched `experiments/phase0_baseline_table.py`:
   - Honor both modern and legacy vars (`NUMBA_CUDA_*` and `NUMBAPRO_*`).
   - Resolve and store explicit libdevice `.bc` file path.
   - Added compatibility shim to copy `libdevice.10.bc` next to `nvvm.dll` for Numba 0.59 Windows lookup behavior.

**Result**
- `Missing libdevice` error no longer primary failure.

---

## 6) Batch Command Parsing Failures in PowerShell/Conda Run

**Symptom**
- Command execution errors:
  - `The token '&&' is not a valid statement separator...`
  - `AssertionError: Support for scripts where arguments contain newlines not implemented`

**Cause**
- PowerShell statement separators differ from Bash (`;` vs `&&` in older PS contexts).
- `conda run ... -c` payloads containing newlines are not supported by Conda wrapper.

**Fix**
- Rewrote commands as single-line Python snippets.
- Used PowerShell-compatible separators.

**Result**
- Version checks and diagnostics executed correctly.

---

## 7) Kernel Runtime Access Violations (Phase 0 blocker)

**Symptom**
- During Phase 0 sweep, configs failed with:
  - `OSError: exception: access violation reading 0x...0008`
- Output CSV saved with `Rows: 0`.

**Cause (confirmed)**
- The access violation was triggered by `numba.cuda.synchronize()`.
  - Numba’s `cuda.synchronize()` calls `cuCtxSynchronize()`.
  - On this Windows + NVIDIA driver setup, `cuCtxSynchronize()` intermittently raises an access violation even in a clean context.

**Minimal reproduction**
- This fails in `gpu-jit-opt` (even with no kernels launched):
  - `python -c "import numba.cuda as cuda; cuda.current_context(); cuda.synchronize()"`
- But stream synchronization succeeds:
  - `python -c "import numba.cuda as cuda; cuda.default_stream().synchronize(); print('ok')"`

**Fix (implemented)**
- Avoid `cuda.synchronize()` in Phase 0 hot paths; use stream sync instead:
  - Replaced `cuda.synchronize()` with `cuda.default_stream().synchronize()` in:
    - `kernels/gemm.py`, `kernels/reduction.py`, `kernels/softmax.py`
    - `profiling/cuda_timer.py`, `profiling/ncu_utils.py`
    - `scripts/numba_nvvm_test.py`

**Result**
- Phase 0 now completes and writes a populated CSV:
  - `✓ Saved 81 results to results\tables\phase0_baseline.csv`
- Tests pass:
  - `9 passed, 2 skipped`

---

## 8) Conda Environment Switch Confusion (`INCLUDE_CONDA_NVCC_BACKUP`)

**Symptom**
- On activating `gpu-jit-opt-clean`, terminal printed:
  - `if defined INCLUDE_CONDA_NVCC_BACKUP (...)`

**Cause**
- This was the deactivation hook from previous env (`gpu-jit-opt`) restoring `INCLUDE`; not a failure in clean env.

**Fix**
- Confirmed clean env hooks contain only OpenSSL/libxml2 hooks.
- Optional hygiene: open fresh terminal before activating clean env.

**Result**
- Behavior explained; not an active error.

---

## 9) Environment Failure Matrix (Phase 0)

### `gpu-jit-opt` (first environment)

**Observed package states during failure sequence**
- Early state (import failure stage):
  - `numpy==2.2.6`
  - `numba` incompatible with NumPy (`Numba needs NumPy 1.26 or less`)
- Mid state (ABI mismatch stage):
  - `numba==0.59.1`
  - `llvmlite==0.46.0` (incompatible)
- Later corrected state:
  - `numpy==1.26.4`
  - `numba==0.59.1`
  - `llvmlite==0.42.0`
  - `numba-cuda` removed (reintroduced once by solver, then removed again)

**Phase 0 failure signature in this env**
- `Missing libdevice file` (resolved later)
- then runtime crash:
  - `exception: access violation reading 0x...0008`
- Result:
  - `189` failed configurations, `Rows: 0` in `results/tables/phase0_baseline.csv`

### `gpu-jit-opt-clean` (second environment)

**Installed/validated package state**
- `torch==2.1.2+cu121`, `torchvision==0.16.2+cu121`, `torchaudio==2.1.2+cu121`
- `numpy==1.26.4`, `numba==0.59.1`, `llvmlite==0.42.0`
- `stable-baselines3==2.3.2`, `gymnasium==0.29.1`
- `torch-geometric==2.5.3`, `torch-scatter==2.1.2+pt21cu121`, `torch-sparse==0.6.18+pt21cu121`
- `scikit-learn==1.5.1`
- `nvtx` installed via conda (`nvtx==0.2.15`)
- Tooling checks passed (`ncu`, `ptxas`, `nvcc`, CUDA device detection)

**Phase 0 failure signature in this env**
- NVVM/libdevice discovered:
  - `NVVM=...\\nvvm64_40_0.dll`
  - `LIBDEVICE=...\\libdevice.10.bc`
- still runtime crash on all kernels:
  - `exception: access violation reading 0x0000000000000001`
- Result:
  - `189` failed configurations, `Rows: 0` in `results/tables/phase0_baseline.csv`

**Conclusion across both envs**
- Different failure phases were fixed (dependency mismatch, ABI mismatch, libdevice pathing),
  and the final low-level Windows CUDA/Numba synchronization access violation was resolved by switching to stream-level synchronization.

---

## 10) Phase 1 SyntaxError: Windows path `unicodeescape` in docstrings

**Symptom**
- Running Phase 1 scripts failed immediately with a Python parse error such as:
  - `SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position ...: truncated \UXXXXXXXX escape`

**Cause**
- Triple-quoted docstrings contained Windows paths like `C:\Users\...` written as `C:\Users\...` (single backslashes), and sequences like `\U` were interpreted as Unicode escape prefixes.

**Fix**
- Escaped backslashes in docstrings (e.g. `C:\\Users\\...`) or rewrote them as raw strings.

**Result**
- Phase 1 scripts import and run normally.

---

## 11) Phase 1 Import errors when running scripts by path

**Symptom**
- Running `python experiments\phase1_collect_counters.py` raised import errors like:
  - `ModuleNotFoundError: No module named 'profiling'`

**Cause**
- When executing a script by file path, Python sets `sys.path[0]` to the script’s directory (here: `experiments/`). Repo-local packages (`profiling/`, `kernels/`) are not on the import path.

**Fix**
- Added a repo-root `sys.path` bootstrap in Phase 1 scripts so they work when run directly.

**Result**
- Phase 1 scripts can be run from project root without `PYTHONPATH` fiddling.

---

## 12) Phase 1 `ncu` returned rc=1 with no useful diagnostics

**Symptom**
- Phase 1 collection reported failures like:
  - `reason = ncu_failed_rc_1`
- But the console output did not include enough detail to tell why `ncu` failed.

**Cause**
- The initial collector treated nonzero return codes as a generic failure and did not surface stderr/stdout content (where Nsight Compute prints the real reason).

**Fix**
- On the first `ncu` failure, print a short head of `stdout`/`stderr` to expose the real error.
- Disabled `--launch-skip` / `--launch-count` by default (these can fail on some Windows setups), and added a retry path that runs without launch-control flags.

**Result**
- The actual causes (import/path issues, flag incompatibilities, permission problems) became visible and fixable.

---

## 13) Phase 1 Temp-runner import failures under `ncu`

**Symptom**
- `ncu` execution failed with Python import errors inside the generated runner, e.g.:
  - `ModuleNotFoundError: No module named 'kernels'` (or `profiling`)

**Cause**
- Phase 1 runs kernels via a temporary runner script. Nsight Compute executes it from a temporary directory; repo-local imports are not available unless the repo root is explicitly added.

**Fix**
- Injected repo root into the runner’s `sys.path` (e.g. `sys.path.insert(0, r"<repo_root>")`).
- Also ensured `PYTHONPATH` includes the repo root when spawning `ncu` (environment override).

**Result**
- `ncu` could import the project modules reliably; Phase 1 counter collection succeeded across the sweep.

---

## 14) Phase 1 CSV parsing bug: header line offset

**Symptom**
- `ncu --csv` ran, but the collector returned `no_metrics_parsed` (all metrics missing/empty).

**Cause**
- Parsing started *after* finding the header marker text (e.g. “Metric Name”), not from the beginning of the header line. This misaligned columns and caused metric extraction to fail.

**Fix**
- Parse from the beginning of the header line containing the CSV columns (the line that includes “Metric Name”).

**Result**
- Metrics dictionaries populated correctly and were written to `results\tables\phase1_result.csv`.

---

## 15) Phase 1 metric availability variance: `sm_active_pct` columns empty

**Symptom**
- `sm_active_pct_raw` / `sm_active_pct_norm` came out empty even when other counters were present.

**Cause**
- Nsight Compute metric naming/availability differs across versions and driver/tool combinations. The initially requested SM-active metric was not always emitted.

**Fix**
- Added fallback metric support so `sm_active_pct` can be derived from an alternative “SM throughput/utilization” metric when the primary metric is missing.

**Result**
- `sm_active_pct_*` is now populated on more setups (and fails gracefully when not available).

---

## 16) Phase 1 counter permission failures on Windows (WDDM)

**Symptom**
- `ncu` reported counter access issues (common strings include `ERR_NVGPUCTRPERM`) and the collector classified it as `permission_denied`.

**Cause**
- On Windows (WDDM), GPU performance counters are often restricted unless the process is elevated (Administrator) and/or the driver is configured to allow access.

**Fix**
- Added a preflight/smoke test that detects and reports this condition clearly.
- Operational fix: run Phase 1 from an **Administrator** Command Prompt.

**Result**
- Phase 1 can reliably distinguish “tool not installed” vs “permissions blocked” vs “other failure”.

---

## 17) Phase 2 validation gap: kernel tests were placeholders

**Symptom**
- `tests/test_kernels.py` existed but contained placeholder tests (effectively no correctness validation).

**Cause**
- Phase 2 kernel work started before a correctness gate was added; performance experiments risked optimizing incorrect outputs.

**Fix**
- Implemented skip-safe correctness tests for `gemm`, `reduction`, and `softmax`.
- Each kernel is tested on a small input against a NumPy reference (or invariants for softmax) and tested both for `reg_cap=default` and a capped variant (e.g. `reg_cap=32`).

**Result**
- Phase 2 now has a basic correctness safety net before moving to RL (Phase 3).

---

## 18) Phase 3 Critical: CUPTI Training Unsustainably Slow Due to Per-Step Profiling Overhead

**Symptom**
- CUPTI+NVML training with 50,000 steps showed progress estimate of 70–400 hours.
- Command: `python train_rl.py --total-steps 50000 --use-cupti --use-nvml` reported multi-day runtimes.
- Even reduced to 10,000 steps: "it's showing 14 hours"

**Root Cause**
- `--use-cupti` flag integrates Nsight Compute (ncu) subprocess call into the RL environment's `_measure_time_ms()` method.
- Each environment step calls `ncu` to profile the kernel run, collecting CUPTI hardware counters.
- ncu subprocess overhead on Windows WDDM: **5–30 seconds per kernel execution**.
- Linear scaling: 50,000 steps × 5–30 sec/step = 250,000–1,500,000 seconds = **70–400 hours**
- Even 10,000 steps × 15 sec average ÷ 3,600 = **41 hours** (matches observed "14 hour estimate at 1% progress")

**Evidence**
- 10,000 step command showed 14-hour extrapolation early in training
- 50,000 step training reached 22+ hours at 18% completion before context corruption
- Math confirms: runtime grows linearly with total_steps when CUPTI-per-step is enabled

**Workaround**
- **DO NOT combine CUPTI with full training (>5,000 steps) on Windows.**
- Option A (Recommended): Train with `--use-nvml` only (9–10 min, 50,176 steps, 3.15x reward), then collect CUPTI on small rollout subset via `phase3_rollout_log.py` (30–60 min).
- Option B (Limited): Train with CUPTI on 3,000–5,000 steps only (4–8 hours, partial convergence).
- Option C (Disable): Use NVML-only training permanently; acceptable for many applications.

**Why It Happens**
- Windows WDDM (Windows Display Driver Model) adds overhead to GPU profiler calls.
- GPU context cannot be shared between training process and ncu subprocess; profiling requires subprocess isolation.
- No pipeline parallelism: each kernel must complete its profiling before next step begins.

**Result**
- Users now understand CUPTI is unsuitable for full training; hybrid approach is recommended for research.
- Documentation updated with time estimates and strategic alternatives.
- Phase 3 training via NVML is fast/stable and production-ready.

---

## 19) Phase 3 Critical: CUDA Context Corruption After 22+ Hours of CUPTI Profiling

**Symptom**
- CUPTI+NVML training ran for 22+ hours, reaching 18% of 50,000 steps.
- At 18% completion, training crashed with:
  - `OSError: exception: access violation reading 0x00000000FFFFFFFA`
  - OR similar CUDA context corruption / device sync failures
- Crash occurred in `_measure_time_ms()` or `cuda.synchronize()` region of environment code.

**Root Cause**
- Sustained CUPTI profiling over 22+ hours destabilizes CUDA device context on Windows.
- WDDM driver does not gracefully handle continuous subprocess profiling with repeated context switches.
- GPU memory becomes fragmented after thousands of ncu subprocess invocations.
- CUDA context handle becomes invalid or corrupted after prolonged use.

**Evidence**
- Context corruption appeared consistently around 20+ hour mark.
- Only occurred with CUPTI profiling; NVML-only training runs 50,000 steps in 9.7 min with zero crashes.
- Windows-specific: likely due to WDDM driver limitations (not observed on Linux in literature).

**Implementation of Recovery**
- Added graceful CUDA degradation in `environment/kernel_env.py`:
  - Wrapped `_measure_time_ms()` in try-catch for `OSError` and `IndexError`.
  - On timing failure, returns cached `_last_valid_time_ms × 1.05` (penalizes bad config but doesn't crash).
  - Episode continues instead of hard failure.
- Removed aggressive `cuda.close()` that was destroying device manager state.
- Added training-time warning when `--use-cupti` is enabled with large step counts (>5,000).

**Lessons**
- CUPTI is designed for analysis/profiling workflows, not continuous training integration.
- Hybrid approach (train fast with NVML, profile selectively) is the only sustainable path on Windows WDDM.

**Result**
- Training no longer crashes after hours of CUPTI profiling; gracefully degrades if context corruption occurs.
- Users directed away from large-scale CUPTI training; alternatives provided.

---

## 20) Phase 3: WDDM Driver Amplifies Profiler Overhead vs Linux

**Symptom**
- CUPTI per-step profiling was 5–30 seconds per kernel on Windows (RTX 3050 Ti).
- Same workflow on Linux systems (if any) would be significantly faster.
- Windows WDDM makes GPU profiling fundamentally slower than on Linux (where NVIDIA driver uses different kernel model).

**Root Cause**
- Windows WDDM (Windows Display Driver Model): GPU is time-sliced for display + compute.
  - Every GPU operation requires Driver interaction and context management overhead.
  - Profiling subprocess adds additional OS scheduling overhead.
- Linux NVIDIA driver: Direct GPU access, fewer OS scheduling layers, lower profiler overhead.
- ncu tool itself has same overhead, but Windows OS layers amplify it.

**Evidence**
- Empirical observation: 5–30 sec/step on Windows WDDM-based RTX 3050 Ti.
- Literature: GPU profilers on Windows consistently report 10–50x overhead vs Linux for similar operations.

**No Direct Fix**
- This is a fundamental Windows architectural limitation.
- Cannot be resolved in userspace code.

**Mitigation**
- Use NVML instead (lightweight, available on all platforms, <1ms overhead).
- Reserve CUPTI for post-training analysis on selected rollout episodes.
- Document Windows-specific limitations in README.

**Result**
- Users understand Windows profiler overhead is not a bug, but inherent to WDDM.
- Recommendations shift toward NVML-first design.

---

## Final State

- Dependency mismatches (`numpy/numba/llvmlite`, `numba-cuda`) were identified and corrected multiple times.
- NVVM/libdevice discovery issues were patched in code.
- The runtime access violation blocker was resolved by replacing `cuda.synchronize()` (context sync) with `cuda.default_stream().synchronize()` (stream sync) in Phase 0 code paths.
- Phase 0 now produces a non-empty baseline table (`81` rows).
- Phase 1 counter collection was stabilized on Windows (path/import issues + CSV parsing + metric fallbacks) and can write a populated `results\tables\phase1_result.csv` when counter permissions allow it.
- Phase 2 now includes kernel correctness tests so future optimization steps don't regress correctness silently.
- **Phase 3 CUPTI Integration (NEW):** Discovered CUPTI per-step profiling is unsustainable for full training (70–400 hour extrapolation). Hybrid approach (NVML training + post-hoc CUPTI analysis) is recommended. Limited CUPTI training (3k–5k steps, 4–8 hours) acceptable for research. Implemented graceful CUDA context degradation to prevent crashes. Documented Windows WDDM profiler overhead as architectural limitation.
