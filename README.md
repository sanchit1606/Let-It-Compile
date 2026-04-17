<div align="center">

# Let It Compile

## An RL Approach to Adaptive Register Allocation for GPU Kernel Optimization Across the Stack

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Numba](https://img.shields.io/badge/Numba-CUDA-success?logo=nvidia)](https://numba.pydata.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL-orange)](https://gymnasium.farama.org/)

**Research Goal:** Demonstrate that a reinforcement learning (RL) agent, conditioned on CUPTI hardware counter signals, can adaptively select GPU compilation and launch parameters (e.g., register cap via --maxrregcount, block size, and shared memory allocation) to outperform PTXAS static defaults across multiple NVIDIA GPU architectures, including RTX 3050 Ti (Ampere), GeForce RTX 4060 (Ada Lovelace), and NVIDIA L40 (Ada Lovelace), with further evaluation planned on data center-class GPUs such as A100 and H100 for real-world, large-scale CUDA workloads.

</div>

---

## Overview

This project implements a **7-phase prototype** for automated GPU kernel compilation optimization:

1. **Phase 0** - Foundational Experiment (Baseline table: regcap → occupancy → runtime)
2. **Phase 1** - CUPTI Instrumentation (Hardware counter collection via ncu)
3. **Phase 2** - Benchmark Kernels (GEMM, reduction, softmax)
4. **Phase 3** - RL Environment (Gymnasium interface)
5. **Phase 4** - PPO Agent Training (Stable-Baselines3)
6. **Phase 5** - BiLSTM Phase Detector (Optional: classify kernel phases)
7. **Phase 6** - GNN IR Encoder (Optional: kernel structure encoding)

## 📖 Documentation

Complete documentation is available in the `docs/` folder. Open it in your browser:

- **Quick Launch (Windows):** Double-click `open_docs.bat`
- **Quick Launch (Any OS):** `python open_docs.py`
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
├── kernels/          # CUDA kernel definitions (Numba CUDA)
├── profiling/        # Hardware profiling (CUPTI, NVML, timing)
├── compiler/         # Compilation control (PTXAS, register allocation)
├── environment/      # Gymnasium RL environment
├── models/           # ML models (BiLSTM, GNN, policy networks)
├── training/         # Training scripts (PPO, phase detector)
├── experiments/      # Phase runners (0-7)
├── results/          # Auto-generated outputs (tables, plots, checkpoints)
├── tests/            # Unit and integration tests
├── requirements.txt  # Pip dependencies
├── environment.yml   # Conda environment spec
└── pyproject.toml    # Project configuration
```

## Installation

### Quick Setup

```bash
# Create conda environment
conda create -n gpu-jit-opt python=3.10 numba cudatoolkit=12.1 numpy pandas scipy scikit-learn matplotlib seaborn jupyter pytest black isort pynvml nvtx -c conda-forge -c pytorch --yes

# Activate
conda activate gpu-jit-opt

# Install PyTorch with CUDA 12.1
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining packages
pip install -r requirements.txt
```

### Verify Installation

```bash
python check-for-packages.py
```

Expected output: All packages installed, CUDA available, RTX 3050 Ti detected.

### Phase 0: Foundational Experiment (Start Here)

```bash
python experiments/phase0_baseline_table.py
```

This produces a CSV table showing:
- How `--maxrregcount` affects warp occupancy
- The relationship between occupancy and kernel runtime
- Which register cap values are optimal for different kernel types

**Output:** `results/tables/phase0_baseline.csv`  
**Time:** ~3 seconds

### Phase 3: PPO Training (NVML-only - RECOMMENDED)

```bash
python train_rl.py --total-steps 50000 --max-episode-len 50 --use-nvml
```

Trains a PPO agent to select optimization parameters (block size, register cap).

**Time:** ~15-30 minutes  
**Output:** Trained model in `results/models/ppo_final.zip`  
**Monitoring:** `tensorboard --logdir results/logs/tensorboard`

### Phase 3 Analysis: CUPTI Metrics (Optional)

After training completes, collect detailed hardware metrics:

```bash
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
- **State:** 16D vector (CUPTI counters + NVML state + kernel type + prev action)
- **Action:** Discrete choices for block size, register cap, shared memory
- **Reward:** Speedup fraction relative to PTXAS default

### Training (`training/`)
- **Config:** Centralized hyperparameter management
- **Train RL:** PPO training loop with evaluation callbacks
- **Train Phase Detector:** BiLSTM classification for kernel phases

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
3. **Run Phase 0** → Validate occupancy relationship
4. **Build RL env** → `environment/kernel_env.py`
5. **Train agent** → `training/train_rl.py`
6. **Evaluate** → `experiments/phase2_rl_baseline.py`

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
