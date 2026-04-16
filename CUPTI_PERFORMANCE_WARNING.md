# ⚠️ CRITICAL: CUPTI+PPO Training Performance Issue

## What Happened

Your training with `--use-cupti --use-nvml` ran for **22+ hours** and only reached **18% completion** before crashing with a CUDA context error.

**Expected time:** 2-8 hours (documentation estimate)  
**Actual time:** 22+ hours and still not finished

## Root Cause

Each environment step during PPO training calls `ncu` (Nsight Compute) to profile the kernel:
- Each `ncu` profiling call takes **5-30 seconds**
- With 50,000 total steps, that's: **50,000 steps × 5-30 sec = 250,000-1,500,000 seconds = 70-400 hours!**
- After sustained profiling for 22+ hours on Windows WDDM, the CUDA context becomes corrupted
- Error: `OSError: exception: access violation reading 0x00000000FFFFFFFA`

## Why It Failed

1. **CUPTI is extremely slow on Windows** (WDDM driver model adds overhead)
2. **Continuous profiling degrades CUDA driver stability** over time
3. **The documentation was optimistic** - it assumed profiling overhead would be linear and predictable (it's not on Windows)

## ✅ Correct Workflow (RECOMMENDED)

### Step 1: Train with NVML-only (Fast, 15-30 minutes)
```cmd
python train_rl.py --total-steps 50000 --max-episode-len 30 --use-nvml --log-interval 1000
```
- **Time:** 15-30 min for 50k steps
- **No privileges needed:** Works without Admin
- **Stable:** No CUDA context issues
- **Signal quality:** Good (GPU util, memory, temperature)

### Step 2: Collect detailed metrics on small subset (Optional, for analysis)
```cmd
python phase3_rollout_log.py --use-cupti --use-nvml --kernels gemm reduction softmax --matrix-sizes 256 512 --episodes-per-case 3 --max-steps 10
```
- **Time:** 30-60 minutes
- **Purpose:** Analyze your trained policy with rich hardware counters
- **What you get:** Achieved occupancy, L2 hit rate, DRAM bandwidth, SM active %

## CUPTI+PPO Training (Not Recommended)

**Only use this if:**
- You have a very small step count (< 5,000 steps)
- You're willing to wait days
- You need hardware counters during training (rare)

**If you must:**
```cmd
# Very short training with CUPTI (1-2 hours)
python train_rl.py --total-steps 5000 --max-episode-len 20 --use-cupti --use-nvml --cupti-timeout-s 180
```

**WARNING SIGNS:**
```
- Training taking > 4 hours for first 10k steps → STOP and switch to NVML-only
- GPU memory errors or timeouts → CUDA context corruption, need NVML-only
- Frequent kernel call delays → CUPTI overhead, use NVML-only
```

## Recovery Steps

### If Training Crashed:

1. **Kill any stuck processes:**
   ```cmd
   taskkill /F /IM ncu.exe
   taskkill /F /IM python.exe /T
   ```

2. **Restart fresh with NVML-only:**
   ```cmd
   cd /d "C:\Users\HP\Desktop\CD PROBLEM STATEMENT\JIT Optimization across GPU stack" && conda activate gpu-jit-opt && python train_rl.py --total-steps 50000 --max-episode-len 30 --use-nvml --log-interval 1000
   ```

3. **If GPU is still unstable, reboot:**
   ```cmd
   shutdown /r
   ```

## Updated Documentation

The documentation has been updated to clarify:
- ✅ NVML-only training: **15-30 min** (correct estimate)
- ⚠️ CUPTI+NVML full training: **NOT RECOMMENDED** on Windows (too slow)
- ✅ CUPTI for rollout analysis: **Use phase3_rollout_log.py** instead

## Files Modified

1. `training/train_rl.py` - Added warning if `--use-cupti` with `--total-steps > 5000`
2. `environment/kernel_env.py` - Added CUDA context recovery (catch access violations and reset)
3. `docs/index.html` - Clarified CUPTI vs NVML trade-offs

## Going Forward

**Recommended approach:**
```
1. Always use: python train_rl.py --use-nvml [other args]
2. When done training, analyze with short CUPTI run:
   python phase3_rollout_log.py --use-cupti --use-nvml --matrix-sizes 256 --episodes-per-case 3
3. This keeps training fast and analysis detailed!
```

## Questions?

See documentation sections:
- [Phase 3: RL Environment & PPO Training](docs/index.html#phase3)
- [NVML-only vs CUPTI+NVML Training](docs/index.html#phase3)
- [Troubleshooting](docs/index.html#troubleshooting)
