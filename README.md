<div align="center">

# Let It Compile

## An RL Approach to Adaptive Register Allocation for GPU Kernel Optimization Across the Stack

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Numba](https://img.shields.io/badge/Numba-CUDA-success?logo=nvidia)](https://numba.pydata.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL-orange)](https://gymnasium.farama.org/)

**Research Goal:** Demonstrate that a reinforcement learning (RL) agent, conditioned on CUPTI hardware counter signals, can adaptively select GPU compilation and launch parameters (currently: register cap via `--maxrregcount`/Numba `max_registers` and block size; planned: shared-memory tuning) to outperform PTXAS static defaults across multiple NVIDIA GPU architectures, including RTX 3050 Ti (Ampere), GeForce RTX 4060 (Ada Lovelace), and NVIDIA L40 (Ada Lovelace), with further evaluation planned on data center-class GPUs such as A100 and H100 for real-world, large-scale CUDA workloads.

</div>

---

## Overview

This project is organized as a **phased prototype** for automated GPU kernel compilation optimization:

1. **Phase 0** — Baseline sweep (reg cap → theoretical occupancy → runtime)
2. **Phase 1** — Hardware counter collection via Nsight Compute (`ncu`)
3. **Phase 2** — Benchmark kernels + correctness validation (GEMM, reduction, softmax)
4. **Phase 3** — RL environment + rollouts + PPO training (Gymnasium + Stable-Baselines3)

Planned / optional extensions:
- **Phase 4+** — Phase detection models (BiLSTM) and IR encoders (GNN) for richer state representations

### Current Status (what you can run today)

- **Docs site:** `docs/index.html` (includes the full “Help Understanding Results” guide embedded on-page)
- **Phase 0:** `experiments/phase0_baseline_table.py` → `results/tables/phase0_baseline.csv`
- **Phase 1:** `experiments/phase1_collect_counters.py` (Admin recommended on Windows) → `results/tables/phase1_result.csv`
- **Phase 2:** `pytest tests/test_kernels.py` (plus opt-in slow tests)
- **Phase 3 rollouts:** `phase3_rollout_log.py` → `results/tables/phase3_rollout.csv`, `results/tables/phase3_episode_summary.csv`
- **Phase 3 PPO training:** `train_rl.py` → `results/models/*.zip` + TensorBoard logs

## Documentation

Complete documentation is available in the `docs/` folder. Open it in your browser:

- **Quick Launch (Windows):** Double-click `open_docs.bat`
- **Quick Launch (Any OS):** `python open_docs.py`
- **Local HTTP server:** `python open_docs.py --server --port 8000`
- **Manual:** Open `docs/index.html` directly in your browser

The documentation includes:
- Getting started guide
- Detailed explanation of all phases
- GPU concepts (occupancy, registers, metrics)
- Installation & setup instructions
- Result interpretation guides
- API reference
- Troubleshooting tips
- Developer information

## Hardware Target

- **GPU:** NVIDIA RTX 3050 Ti Laptop GPU
- **Compute Capability:** SM 8.6 (Ampere architecture)
- **VRAM:** 4 GB GDDR6
- **CUDA Cores:** 2048 (20 SMs)

## Project Structure

```
gpu-jit-opt/
├── docs/               # Static documentation site (GitHub Pages compatible)
├── open_docs.bat        # Windows docs launcher
├── open_docs.py         # Cross-platform docs launcher (optional local server)
├── kernels/             # CUDA kernel definitions (Numba CUDA)
├── profiling/           # Profiling + telemetry (ncu/CUPTI, NVML, timing)
├── compiler/            # Compilation control (PTXAS helpers, occupancy calc)
├── environment/         # Gymnasium RL environment (state/action/reward)
├── training/            # PPO training implementation
├── experiments/         # Phase runners (Phase 0/1/3)
├── scripts/             # Convenience entrypoints
├── results/             # Generated outputs (tables, models, logs)
├── tests/               # Kernel correctness + environment/profiling tests
├── requirements.txt     # Pinned pip dependencies (recommended)
└── pyproject.toml       # Project metadata & tool config
```

## Installation

### Recommended Setup (Conda + pip)

The easiest way to get a known-good set of versions (especially on Windows + Numba) is to use the pinned `requirements.txt`.

```bat
conda create -n gpu-jit-opt python=3.10 -y
conda activate gpu-jit-opt

pip install -r requirements.txt
```

Notes:
- `requirements.txt` includes the correct PyTorch CUDA 12.1 wheel index settings.
- On Windows, `nvtx` is intentionally skipped by default (see `requirements.txt`).

### Verify Installation

```bat
python check-for-packages.py
```

Optional sanity check:

```bat
python check_cuda.py
```

Expected output: All packages installed, CUDA available, RTX 3050 Ti detected.

### Phase 0: Foundational Experiment (Start Here)

```bat
python experiments/phase0_baseline_table.py
```

This produces a CSV table showing:
- How `--maxrregcount` affects warp occupancy
- The relationship between occupancy and kernel runtime
- Which register cap values are optimal for different kernel types

**Output:** `results/tables/phase0_baseline.csv`  
**Time:** ~3 seconds

### Phase 1: Hardware Counters (Optional, `ncu`)

Collect Nsight Compute counters for real kernels and write a CSV.

Windows note: run from an **Administrator** Command Prompt if counters are blocked.

```bat
python experiments/phase1_collect_counters.py
```

**Output:** `results/tables/phase1_result.csv`

Convenience wrapper:

```bat
python scripts\phase1_show_counters.py
```

### Phase 2: Kernel Correctness (Recommended before RL)

```bat
pytest -q tests\test_kernels.py -vv
```

Opt-in validation on medium/large sizes:

```bat
pytest -q tests\test_kernels.py -vv --runslow -m slow
```

### Phase 3: PPO Training (NVML-only — Recommended)

```bat
python train_rl.py --total-steps 50000 --max-episode-len 50 --use-nvml
```

Trains a PPO agent to select optimization parameters (block size, register cap).

**Time:** ~15-30 minutes  
**Output:** Trained model in `results/models/<run_tag>.zip` (exact path is recorded in the training summary JSON)  
**Monitoring:** `tensorboard --logdir results/logs/tensorboard`

### Phase 3: Rollout Logging + CUPTI Metrics (Optional)

After training completes, collect detailed hardware metrics:

```bat
python phase3_rollout_log.py --use-cupti --use-nvml --kernels gemm reduction softmax --matrix-sizes 256 512 --episodes-per-case 3 --max-steps 10
```

**Time:** ~30-60 minutes  
**Output:** Hardware counters (achieved occupancy, L2 hit rate, DRAM bandwidth, SM active %)

## Key Components

### Kernels (`kernels/`)
- **GEMM** — Tiled matrix multiplication (compute-bound)
- **Reduction** — Parallel sum reduction (memory-bound)
- **Softmax** — Row-wise softmax (mixed)

### Profiling (`profiling/`)
- **CUPTI Collector** — Hardware counters via `ncu` CLI
- **NVML Monitor** — Real-time GPU state (lightweight)
- **CUDA Timer** — Precise kernel timing via CUDA events

### Compiler Control (`compiler/`)
- **PTXAS Controller** — `--maxrregcount` and occupancy calculation
- **Numba Compiler** — JIT compilation wrapper
- **IR Extractor** — PTX/LLVM IR analysis

### RL Environment (`environment/`)
- **State (normalized/clamped to [0,1]):**
	- CUPTI vector (4 metrics) if enabled, else zeros
	- NVML vector (4 metrics) if enabled, else zeros
	- Kernel one-hot (3)
	- Previous action (2)
	- Total: **13 dims** with NVML enabled (default); **9 dims** if NVML is disabled
- **Action:** `MultiDiscrete([num_block_sizes, num_reg_caps])` → decoded into `(block_size, reg_cap)`
- **Reward:** Speedup relative to a per-episode baseline (`baseline_ms / time_ms - 1`)

### Training (`training/`)
- **Train RL:** PPO training loop (`training/train_rl.py`) with logs + optional evaluation
- **Wrapper:** `train_rl.py` allows `python train_rl.py ...` from repo root

## Performance Targets

| Kernel | Size | Expected Speedup |
|--------|------|------------------|
| GEMM   | 512  | 10-20%           |
| REDUCTION | 512 | 5-15%            |
| SOFTMAX | 512 | 8-18%            |

**Reference:** CuAsmRL paper achieved 26% average speedup on assembly-level optimizations.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_kernels.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Development Workflow

1. **Implement kernel** → `kernels/gemm.py`
2. **Add profiling** → `profiling/cuda_timer.py`
3. **Run Phase 2 tests** → `pytest tests/test_kernels.py -vv`
4. **Run Phase 0** → validate occupancy/runtime trends
5. **(Optional) Run Phase 1** → collect `ncu` counters (Admin recommended on Windows)
6. **Run Phase 3 rollouts** → `python phase3_rollout_log.py ...`
7. **Train agent** → `python train_rl.py ...`

## Troubleshooting

### Conda activation prints Visual Studio 2019 errors

If `conda activate gpu-jit-opt` prints lots of lines and messages like **"The system cannot find the path specified"** referencing **Visual Studio 2019**, it usually means your conda environment includes activation scripts for the **MSVC v142 (VS2019) toolset** (commonly needed for native extensions / `nvcc` toolchains), but that toolset is not installed on your machine.

Fix (recommended): keep Visual Studio 2022, but install the v142 toolset

- Open **Visual Studio Installer** → **Modify** (VS 2022 / Build Tools 2022)
- Add **Desktop development with C++** (or C++ Build Tools)
- In **Individual components**, ensure:
	- **MSVC v142 - VS 2019 C++ x64/x86 build tools**
	- A **Windows 10/11 SDK**

Alternative: install **Build Tools for Visual Studio 2019** with MSVC v142.

**CUDA not available:**
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Numba CUDA check fails:**
```bash
python -c "from numba import cuda; print(cuda.is_available())"
```

**Out of memory:**
Reduce matrix size: `MATRIX_SIZES = [256, 512, 1024]` (not 2048)

**ncu (Nsight Compute) not found:**
Optional for prototype. System-installed `ncu` is at:
```
C:\Program Files\NVIDIA\Nsight Compute 202x.x\ncu.exe
```
