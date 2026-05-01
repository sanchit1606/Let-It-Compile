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

Planned / priority extensions:
- **Action space expansion** — From 9 to ~1,760 configs (shared memory, L1 partition, finer register granularity)
- **Bayesian Optimization baseline** — Standard autotuning comparison via Optuna/scikit-optimize
- **Register spill cliff analysis** — Measuring the performance cliff when `--maxrregcount` crosses the kernel's actual register requirement
- **Phase detection** (BiLSTM) and **IR encoders** (GNN) for richer state representations
- **Energy-aware reward** — Performance-per-watt optimization via NVML power telemetry



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

