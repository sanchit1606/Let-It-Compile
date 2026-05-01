<div align="center">

# Let It Compile

### Adaptive GPU Compilation Optimization via Reinforcement Learning with Runtime Hardware Feedback

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-PPO-orange)](https://stable-baselines3.readthedocs.io/)

</div>

---

## Abstract

PTXAS makes register allocation decisions at compile time. The GPU only reveals what actually worked at runtime.

This framework closes that gap: a PPO-based RL agent observes live CUPTI hardware counters (achieved occupancy, L2 hit rate, DRAM bandwidth, SM utilization) and NVML telemetry (GPU utilization, temperature, power draw) to adaptively select `--maxrregcount` and block-size configurations that outperform PTXAS static heuristics. The agent treats the compiler as a black box — no PTXAS source access required, optimization happens entirely through its public interface.

**Compilation stack:** `Python → Numba JIT → LLVM IR → PTX → PTXAS → SASS → CUPTI feedback → RL agent`

---

## The Problem

GPU compilers face a fundamental tension: more physical registers per thread reduces costly spilling to local memory, but also reduces occupancy — the GPU's ability to hide memory latency via concurrent warps. PTXAS resolves this with static heuristics that cannot account for runtime workload characteristics.

This creates measurable performance gaps:
- **Memory-bound kernels** (reduction, softmax): up to 1.58× speedup over PTXAS defaults via occupancy tuning
- **Compute-bound kernels** (GEMM): well-served by defaults — the agent correctly learns minimal intervention
- **Cross-layer effects**: CPU-side Numba JIT inlining decisions cascade into GPU register pressure, invisible to PTXAS

---

## MDP Formulation

### State Space (14 dimensions, normalized to [0, 1])

| Dims | Source | Signals |
|------|--------|---------|
| 4 | CUPTI (via `ncu`) | achieved_occupancy, l2_hit_rate, dram_bw_pct, sm_active_pct |
| 4 | NVML | gpu_utilization, memory_utilization, temperature, power_draw |
| 3 | Kernel identity | One-hot: [gemm, reduction, softmax] |
| 2 | Previous action | Normalized (block_size_idx, reg_cap_idx) |
| 1 | Roofline model | arithmetic_intensity / ridge_point *(planned — dim 14)* |

### Action Space

**Current:** `MultiDiscrete([3, 3])` = 9 configurations (block_size × reg_cap)

| Parameter | Options | Values |
|-----------|---------|--------|
| Block size | 3 | 64, 128, 256 |
| Register cap | 3 | 0 (default), 32, 64 |

**Planned expansion:** `MultiDiscrete([8, 11, 5, 4])` = 1,760 configurations
- Block sizes: 32–512 · Register caps: 0–128 · Shared memory: 5 configs · L1 partition: 4 modes

### Reward

```
r = (t_baseline / t_measured) - 1
```
Positive = faster than PTXAS default. Energy-aware variant planned: `r = α·speedup + (1-α)·(1 - P/P_max)`.

---

## Current Results (RTX 3050 Ti, sm_86)

| Strategy | Kernel | Size | Time (ms) | Best Speedup | Samples |
|----------|--------|------|-----------|-------------|---------|
| PTXAS default | Softmax | 512 | 2.648 ± 0.000 | 1.000× | 1 |
| Random search | Softmax | 512 | 2.332 ± 0.523 | 1.584× | 100 |
| **PPO agent** | **Softmax** | **512** | **1.664 ± 0.002** | **1.583×** | **150** |

**Key finding:** PPO achieves near-optimal speedup with **60× lower variance** than random search (±0.002ms vs ±0.523ms).

**Known limitation:** The 3×3 = 9 configuration space is trivially searchable by exhaustive methods. Action space expansion to ~1,760 configs is the highest priority.

---

## Hardware Targets

| GPU | Architecture | Class | Status | Specification |
|-----|-------------|-------|--------|---------------|
| **RTX 3050 Ti** | Ampere (sm_86) | Consumer | ✅ Primary dev | 20 SMs · 4 GB GDDR6 · 192 GB/s |
| **RTX 4060** | Ada Lovelace (sm_89) | Consumer | 🎯 Lab GPU | 24 SMs · 8 GB GDDR6 · 272 GB/s |
| **NVIDIA L40** | Ada Lovelace (sm_89) | Server | 🎯 Cloud | 48 GB GDDR6 · 864 GB/s |
| **NVIDIA A100** | Ampere (sm_80) | Data center | 🎯 Grant target | 80 GB HBM2e · 2 TB/s |
| **NVIDIA H100** | Hopper (sm_90) | Data center | 🎯 Grant target | 80 GB HBM3 · 3.35 TB/s |

---

## Benchmark Kernels

| Kernel | Pattern | Arithmetic Intensity | Status |
|--------|---------|---------------------|--------|
| **GEMM** | Compute-bound | N/6 FLOP/byte | ✅ Implemented |
| **Reduction** | Memory-bound | 0.25 FLOP/byte | ✅ Implemented |
| **Softmax** | Memory-bound | 0.625 FLOP/byte | ✅ Implemented |
| **LayerNorm** | Memory-bound | ~1.25 FLOP/byte | 🎯 Planned |
| **Batched GEMM** | Compute-bound | N/6 FLOP/byte per batch | 🎯 Planned |
| **Fused Attention** | Mixed | Varies by phase | 🎯 Planned |

---

## Software Stack

```
Python          3.10.x
CUDA Toolkit    12.1 (PTXAS, ncu)
PyTorch         2.1.2+cu121
Numba           0.59.0
Stable-Baselines3  2.3.2
Gymnasium       0.29.1
cuda-python     12.3.0
torch-geometric 2.5.3
pynvml          11.5.0
```

---

## Implementation Status

| Phase | Description | Status |
|-------|------------|--------|
| Phase 0 | Baseline profiling (81 configs: 3 kernels × 3 sizes × 3 block sizes × 3 reg caps) | ✅ Complete |
| Phase 1 | CUPTI hardware counter collection via `ncu` | ✅ Complete |
| Phase 2 | Kernel correctness verification (pytest) | ✅ Complete |
| Phase 3 | Gymnasium RL environment + MDP formulation | ✅ Complete |
| Phase 4 | PPO training (50K steps, 9.7 min, mean reward 3.15) | ✅ Complete |
| Phase 5 | BiLSTM phase detector (142K params, roofline-labelled) | ✅ Complete |
| Phase 6 | GNN IR encoder for PTX structure embedding (18K params) | ✅ Complete |
| Phase 7 | RL vs random search vs PTXAS default comparison | ✅ Complete |

### Priority Roadmap

| Priority | Extension | Impact |
|----------|----------|--------|
| **P1** | Action space expansion (9 → 1,760 configs) | Critical for publication |
| **P2** | Bayesian Optimization baseline (Optuna) | Required for Q1 venues |
| **P3** | Register spill cliff characterization | Novel scientific contribution |
| **P4** | CPU-to-GPU inlining cascade measurement | Cross-layer validation |
| **P5** | Statistical rigor (p-values, CIs, CV) | Publication requirement |
| **P6** | Energy-aware reward (perf-per-watt) | Novel contribution |
| **P7** | MAML cross-architecture transfer (K=3 adaptation) | Generalization proof |
| **P8** | LayerNorm + Batched GEMM + Fused Attention kernels | Transformer relevance |

---

## Project Structure

```
Let-It-Compile/
├── kernels/             # CUDA kernel definitions (Numba CUDA)
├── profiling/           # CUPTI collector, NVML monitor, CUDA timer
├── compiler/            # PTXAS control, occupancy calc, IR extraction
├── environment/         # Gymnasium RL environment (state/action/reward)
├── models/              # BiLSTM phase detector, GNN IR encoder
├── training/            # PPO training loop (Stable-Baselines3)
├── experiments/         # Phase runners (Phase 0/1/7)
├── results/             # Generated outputs (tables, models, logs)
├── tests/               # Kernel correctness + environment tests
├── docs/                # Documentation site (GitHub Pages)
└── requirements.txt     # Pinned dependencies
```

---

## References

1. He & Yoneki. *CuAsmRL: Optimizing GPU SASS Schedules via Deep RL.* CGO 2025.
2. Trofin et al. *MLGO: A Machine Learning Guided Compiler Optimizations Framework.* arXiv:2101.04808.
3. Zheng et al. *Ansor: Generating High-Performance Tensor Programs for Deep Learning.* OSDI 2020.
4. Schulman et al. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
5. Williams et al. *Roofline: An Insightful Visual Performance Model.* CACM 2009.
