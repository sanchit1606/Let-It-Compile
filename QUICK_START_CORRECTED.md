# ✅ Quick Start (CORRECTED - NVML-ONLY RECOMMENDED)

## Setup Environment
```cmd
cd /d "C:\Users\HP\Desktop\CD PROBLEM STATEMENT\JIT Optimization across GPU stack"
conda activate gpu-jit-opt
```

## ✅ RECOMMENDED: Phase 0 Baseline
```cmd
python experiments\phase0_baseline_table.py
```
**Time:** ~3 seconds  
**Output:** `results/tables/phase0_baseline.csv`

## ✅ RECOMMENDED: Phase 3 Rollout Logging (NVML-only)
```cmd
python phase3_rollout_log.py --use-nvml --kernels gemm reduction softmax --matrix-sizes 256 512 --episodes-per-case 1 --max-steps 10 --warmup 1 --repeats 5
```
**Time:** ~2-5 minutes  
**Output:** Step-level and episode-level CSVs

## ✅ RECOMMENDED: PPO Training (NVML-only - FAST)
```cmd
python train_rl.py --total-steps 50000 --max-episode-len 50 --use-nvml --eval-freq 5000 --n-eval-episodes 5 --log-interval 1000
```
**Time:** ~15-30 minutes  
**Output:** Trained policy in `results/models/ppo_final.zip`

## Optional: CUPTI Analysis on Trained Policy (After PPO Completes)
```cmd
python phase3_rollout_log.py --use-cupti --use-nvml --kernels gemm reduction softmax --matrix-sizes 256 512 --episodes-per-case 3 --max-steps 10 --cupti-timeout-s 180
```
**Time:** ~30-60 minutes  
**Output:** Rich hardware metrics (occupancy, L2 hit rate, DRAM BW, SM active %)

---

## ⚠️ DO NOT USE (EXTREMELY SLOW)
```cmd
# ❌ This will take 70-400 hours! (observed: 22+ hours for 18%)
python train_rl.py --total-steps 50000 --max-episode-len 30 --use-cupti --use-nvml
```

## Kernel Correctness Tests
```cmd
# Fast suite
pytest tests\test_kernels.py -v

# Performance-scale suite (optional)
pytest tests\test_kernels.py -v --runslow
```

---

## Complete Workflow (30-45 minutes total)

```cmd
# 1. Baseline
python experiments\phase0_baseline_table.py
# (~3 sec)

# 2. Rollout logging
python phase3_rollout_log.py --use-nvml --kernels gemm reduction softmax --matrix-sizes 256 512 --episodes-per-case 1 --max-steps 10 --warmup 1 --repeats 5
# (~5 min)

# 3. PPO training (main work)
python train_rl.py --total-steps 50000 --max-episode-len 50 --use-nvml --eval-freq 5000 --n-eval-episodes 5
# (~20-30 min)

# 4. Results!
# - Trained model: results/models/ppo_final.zip
# - Training log: results/logs/train_rl.log
# - TensorBoard: tensorboard --logdir=results/logs/tensorboard
```

## If You Need Hardware Counters

After step 3 (training completes), analyze with CUPTI:
```cmd
python phase3_rollout_log.py --use-cupti --use-nvml --kernels gemm reduction softmax --matrix-sizes 256 512 --episodes-per-case 3 --max-steps 10
# (~30-60 min for detailed analysis)
```

This gives you:
- ✅ Fast training (20-30 min with NVML)
- ✅ Detailed analysis (30-60 min with CUPTI)  
- ✅ Total time: ~1-2 hours for complete pipeline
- ✅ NO 22-hour crashes!

---

See `CUPTI_PERFORMANCE_WARNING.md` for detailed explanation of the issue and recovery steps.
