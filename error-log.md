# Error Log

## Scope
Errors encountered while setting up and running `experiments/phase0_baseline_table.py` in this workspace, including root cause and remediation.

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

## 7) Kernel Runtime Access Violations (Current Blocking Issue)

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
  but final blocker persists as a low-level Windows CUDA JIT/runtime access violation.

---

## Final State

- Dependency mismatches (`numpy/numba/llvmlite`, `numba-cuda`) were identified and corrected multiple times.
- NVVM/libdevice discovery issues were patched in code.
- The runtime access violation blocker was resolved by replacing `cuda.synchronize()` (context sync) with `cuda.default_stream().synchronize()` (stream sync) in Phase 0 code paths.
- Phase 0 now produces a non-empty baseline table (`81` rows).
