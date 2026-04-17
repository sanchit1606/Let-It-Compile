from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import threading
import time
import logging
import gc

import gymnasium as gym
import numpy as np

from environment.action_space import (
    KERNEL_NAMES,
    decode_action,
    make_action_space,
)
from environment.reward import speedup_reward
from environment.state_space import (
    ObservationSpec,
    build_observation,
    cupti_dict_to_vector,
    make_observation_space,
)
from kernels.gemm import run_gemm
from kernels.reduction import run_reduction
from kernels.softmax import run_softmax
from profiling.cuda_timer import time_kernel
from profiling.cupti_collector import CUPTICollector, DEFAULT_NCU_METRICS, default_state_vector


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EpisodeConfig:
    """Configuration for a single episode.

    kernel_name:
      - One of: gemm, reduction, softmax
      - Or "random" to sample a kernel at reset

    matrix_size:
      - For gemm/softmax: N means NxN
      - For reduction: we reduce N*N elements (to match Phase 0/1 convention)
    """

    kernel_name: str = "gemm"
    matrix_size: int = 512
    max_steps: int = 20

    # Timing
    warmup: int = 1
    repeats: int = 5

    # Observation sources
    use_cupti: bool = False
    use_nvml: bool = True

    # CUPTI timeouts (ncu subprocess)
    cupti_timeout_s: int = 120


class KernelOptimizationEnv(gym.Env):
    """Gymnasium environment for kernel optimization.

    Action:
      - Select (block_size, reg_cap)

    Observation (normalized [0,1], best-effort):
      - CUPTI vector (Phase 1 metrics) OR zeros if unavailable
      - NVML vector (util/mem/temp) OR zeros if unavailable
      - Kernel one-hot
      - Previous action (indices normalized)

    Reward:
      - speedup relative to per-episode baseline (default config)
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[EpisodeConfig] = None, *, obs_spec: Optional[ObservationSpec] = None):
        super().__init__()

        self.cfg = cfg or EpisodeConfig()
        self.obs_spec = obs_spec or ObservationSpec(include_nvml=self.cfg.use_nvml)

        self.action_space = make_action_space()
        self.observation_space = make_observation_space(self.obs_spec)

        self._rng = np.random.default_rng()
        self._step = 0
        self._kernel_name = "gemm"

        self._baseline_ms: float = 1.0
        self._prev_action_norm = np.zeros(2, dtype=np.float32)

        # Optional collectors (lazy init)
        self._cupti: Optional[CUPTICollector] = None
        self._nvml = None

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

    def _pick_kernel_name(self) -> str:
        if self.cfg.kernel_name == "random":
            return str(self._rng.choice(KERNEL_NAMES))
        if self.cfg.kernel_name not in KERNEL_NAMES:
            raise ValueError(f"Invalid kernel_name={self.cfg.kernel_name}. Use one of {KERNEL_NAMES} or 'random'.")
        return self.cfg.kernel_name

    def _ensure_collectors(self):
        if self.cfg.use_cupti and self._cupti is None:
            metrics = {
                "achieved_occupancy": DEFAULT_NCU_METRICS["achieved_occupancy"],
                "l2_hit_rate": DEFAULT_NCU_METRICS["l2_hit_rate"],
                "dram_bw_pct": DEFAULT_NCU_METRICS["dram_bw_pct"],
                "sm_active_pct": DEFAULT_NCU_METRICS["sm_active_pct"],
            }
            self._cupti = CUPTICollector(metrics=metrics)

        if self.cfg.use_nvml and self._nvml is None:
            try:
                from profiling.nvml_monitor import NVMLMonitor

                self._nvml = NVMLMonitor()
            except Exception:
                self._nvml = None

    def _prepare_kernel(self, kernel_name: str, block_size: int, reg_cap: int):
        """Prepare kernel with aggressive CUDA context safety.
        
        Note: Caller is responsible for freeing returned device arrays.
        """
        try:
            n = int(self.cfg.matrix_size)

            if kernel_name == "gemm":
                A, B, C, grid, block, kernel_fn = run_gemm(n, block_size=block_size, warmup=self.cfg.warmup, reg_cap=reg_cap)
                args = (A, B, C, np.int32(n))
                return grid, block, kernel_fn, args, (A, B, C)  # Return device arrays for cleanup

            if kernel_name == "reduction":
                total = int(n * n)
                x, out, grid, block, kernel_fn = run_reduction(total, block_size=block_size, warmup=self.cfg.warmup, reg_cap=reg_cap)
                args = (x, out, np.int32(total))
                return grid, block, kernel_fn, args, (x, out)

            if kernel_name == "softmax":
                x, out, grid, block, kernel_fn = run_softmax(n, block_size=block_size, warmup=self.cfg.warmup, reg_cap=reg_cap)
                args = (x, out, np.int32(n), np.int32(n))
                return grid, block, kernel_fn, args, (x, out)

            raise ValueError(f"Unknown kernel: {kernel_name}")
        except (OSError, RuntimeError) as e:
            error_msg = str(e).lower()
            if "access violation" in error_msg or "cuda" in error_msg or "invalid handle" in error_msg:
                import numba.cuda as cuda
                log = logging.getLogger(__name__)
                log.warning(f"CUDA context corrupted; attempting recovery. Error: {e}")
                
                # Aggressive context reset
                try:
                    cuda.close()
                    log.info("Closed CUDA context")
                except Exception as close_err:
                    log.debug(f"Error closing context: {close_err}")
                
                # Clear kernel caches from all kernel modules
                try:
                    from kernels.gemm import _clear_jit_cache_on_error as clear_gemm
                    from kernels.reduction import _clear_jit_cache_on_error as clear_reduction
                    from kernels.softmax import _clear_jit_cache_on_error as clear_softmax
                    clear_gemm()
                    clear_reduction()
                    clear_softmax()
                    log.info("Cleared JIT caches for all kernels")
                except Exception as cache_err:
                    log.debug(f"Error clearing caches: {cache_err}")
                
                # Force garbage collection to reclaim GPU memory
                gc.collect()
                time.sleep(0.3)  # Longer wait for full driver recovery
                
                # Retry once with fresh compilation
                try:
                    n = int(self.cfg.matrix_size)
                    if kernel_name == "gemm":
                        A, B, C, grid, block, kernel_fn = run_gemm(n, block_size=block_size, warmup=self.cfg.warmup, reg_cap=reg_cap)
                        args = (A, B, C, np.int32(n))
                        return grid, block, kernel_fn, args, (A, B, C)
                    elif kernel_name == "reduction":
                        total = int(n * n)
                        x, out, grid, block, kernel_fn = run_reduction(total, block_size=block_size, warmup=self.cfg.warmup, reg_cap=reg_cap)
                        args = (x, out, np.int32(total))
                        return grid, block, kernel_fn, args, (x, out)
                    elif kernel_name == "softmax":
                        x, out, grid, block, kernel_fn = run_softmax(n, block_size=block_size, warmup=self.cfg.warmup, reg_cap=reg_cap)
                        args = (x, out, np.int32(n), np.int32(n))
                        return grid, block, kernel_fn, args, (x, out)
                except Exception as retry_err:
                    log.error(f"Retry after context reset also failed: {retry_err}")
                    raise
            raise

    def _measure_time_ms(self, kernel_name: str, block_size: int, reg_cap: int) -> float:
        """Measure kernel time with explicit GPU memory cleanup.
        
        This function ensures device arrays are freed after measurement to prevent
        GPU memory exhaustion during long training runs.
        """
        device_arrays = None
        try:
            result = self._prepare_kernel(kernel_name, block_size, reg_cap)
            if len(result) == 5:
                grid, block, kernel_fn, args, device_arrays = result
            else:
                # Fallback for compatibility
                grid, block, kernel_fn, args = result[:4]
                device_arrays = None
            
            timing = time_kernel(kernel_fn, grid, block, args, warmup=0, repeats=self.cfg.repeats)
            self._last_valid_time_ms = float(timing.mean_ms)
            return self._last_valid_time_ms
            
        except (OSError, IndexError, RuntimeError) as e:
            # CUDA context degradation - return cached value or neutral estimate
            error_msg = str(e).lower()
            if "access violation" in error_msg or "list index" in error_msg or "no successful" in error_msg or "invalid handle" in error_msg:
                # Return a fallback time (slightly above baseline to discourage this config)
                fallback = getattr(self, '_last_valid_time_ms', 1.0)  # Default 1ms if no history
                return fallback * 1.05  # Slightly penalize unknown configs
            else:
                raise
        finally:
            # Explicitly free device arrays to prevent memory leak
            if device_arrays is not None:
                try:
                    import numba.cuda as cuda
                    cuda.default_stream().synchronize()
                    gc.collect()  # Force garbage collection
                except:
                    pass  # If cleanup fails, just continue

    def _collect_cupti_norm(self, kernel_name: str, block_size: int, reg_cap: int) -> Tuple[np.ndarray, str, bool]:
        keys = tuple(self.obs_spec.cupti_keys)

        if not self.cfg.use_cupti:
            return default_state_vector(order=keys), "disabled", False

        self._ensure_collectors()
        if self._cupti is None:
            return default_state_vector(order=keys), "collector_unavailable", False

        # Runner code executed under ncu in a temp directory.
        # We inject the repo root so imports work regardless of cwd.
        repo_root = str(_REPO_ROOT)
        n = int(self.cfg.matrix_size)

        code = f"""
import sys
sys.path.insert(0, r"{repo_root}")

import numpy as np
from numba import cuda

from kernels.gemm import run_gemm
from kernels.reduction import run_reduction
from kernels.softmax import run_softmax

kernel = {kernel_name!r}
n = {n}
block_size = {int(block_size)}
reg_cap = {int(reg_cap)}

if kernel == 'gemm':
    A, B, C, grid, block, kernel_fn = run_gemm(n, block_size=block_size, warmup=1, reg_cap=reg_cap)
    args = (A, B, C, np.int32(n))
elif kernel == 'reduction':
    total = n * n
    x, out, grid, block, kernel_fn = run_reduction(total, block_size=block_size, warmup=1, reg_cap=reg_cap)
    args = (x, out, np.int32(total))
elif kernel == 'softmax':
    x, out, grid, block, kernel_fn = run_softmax(n, block_size=block_size, warmup=1, reg_cap=reg_cap)
    args = (x, out, np.int32(n), np.int32(n))
else:
    raise ValueError(kernel)

# One representative measured launch (warmup already happened inside run_*).
kernel_fn[grid, block](*args)
cuda.default_stream().synchronize()
""".lstrip()

        res = self._cupti.collect_from_python_code(code, timeout_s=self.cfg.cupti_timeout_s)
        if not res.ok:
            return default_state_vector(order=keys), res.reason, False

        vec = cupti_dict_to_vector(res.normalized, keys)
        return vec, "ok", True

    def _collect_nvml_norm(self) -> np.ndarray:
        if not self.obs_spec.include_nvml:
            return np.zeros(4, dtype=np.float32)

        self._ensure_collectors()
        if self._nvml is None:
            return np.zeros(4, dtype=np.float32)

        try:
            return self._nvml.to_vector().astype(np.float32)
        except Exception:
            return np.zeros(4, dtype=np.float32)

    def _collect_nvml_peak_during(self, fn) -> Tuple[Optional[np.ndarray], Any]:
        """Collect peak NVML utilization while running a blocking function.

        On Windows/WDDM, NVML utilization can read as 0 for short kernels because it is
        reported over a coarse sampling window. When CUPTI is enabled, the `ncu` subprocess
        runs long enough for NVML to update; polling NVML concurrently yields a much more
        stable utilization signal.

        Returns:
          (nvml_vec_or_none, fn_result)
        """

        if not self.obs_spec.include_nvml:
            return None, fn()

        self._ensure_collectors()
        if self._nvml is None:
            return np.zeros(4, dtype=np.float32), fn()

        stop = threading.Event()
        lock = threading.Lock()

        peak_gpu = 0.0
        peak_mem = 0.0
        last_used = 0.0
        last_temp = 0.0

        def _worker():
            nonlocal peak_gpu, peak_mem, last_used, last_temp
            while not stop.is_set():
                try:
                    v = self._nvml.to_vector()
                    gpu = float(v[0]) if len(v) > 0 else 0.0
                    mem = float(v[1]) if len(v) > 1 else 0.0
                    used = float(v[2]) if len(v) > 2 else 0.0
                    temp = float(v[3]) if len(v) > 3 else 0.0
                    with lock:
                        peak_gpu = max(peak_gpu, gpu)
                        peak_mem = max(peak_mem, mem)
                        last_used = used
                        last_temp = temp
                except Exception:
                    pass

                # 20 Hz polling keeps overhead low and still catches updates.
                time.sleep(0.05)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        try:
            res = fn()
        finally:
            stop.set()
            t.join(timeout=1.0)

        with lock:
            vec = np.array([peak_gpu, peak_mem, last_used, last_temp], dtype=np.float32)
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        vec = np.clip(vec, 0.0, 1.0)
        return vec, res

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        # Aggressive GPU memory cleanup before new episode
        try:
            import numba.cuda as cuda
            gc.collect()
            cuda.default_stream().synchronize()
            log = logging.getLogger(__name__)
            log.debug("Cleaned GPU memory before episode reset")
        except Exception as e:
            pass  # Non-critical

        self._step = 0
        self._kernel_name = self._pick_kernel_name()

        # Baseline: default reg cap, standard block size.
        self._baseline_ms = self._measure_time_ms(self._kernel_name, block_size=256, reg_cap=0)
        self._last_valid_time_ms = self._baseline_ms  # Initialize fallback cache

        self._prev_action_norm = np.zeros(2, dtype=np.float32)

        # NVML: best-effort. For CUPTI runs, poll NVML during the `ncu` subprocess.
        if self.cfg.use_cupti and self.obs_spec.include_nvml:
            nvml_vec, cupti_res = self._collect_nvml_peak_during(
                lambda: self._collect_cupti_norm(self._kernel_name, block_size=256, reg_cap=0)
            )
            cupti_vec, cupti_reason, cupti_ok = cupti_res
        else:
            nvml_vec = self._collect_nvml_norm() if self.obs_spec.include_nvml else None
            cupti_vec, cupti_reason, cupti_ok = self._collect_cupti_norm(self._kernel_name, block_size=256, reg_cap=0)

        obs = build_observation(
            kernel_name=self._kernel_name,
            cupti_norm=cupti_vec,
            nvml_norm=nvml_vec,
            prev_action_norm=self._prev_action_norm,
            spec=self.obs_spec,
        )

        info = {
            "kernel": self._kernel_name,
            "matrix_size": int(self.cfg.matrix_size),
            "baseline_ms": float(self._baseline_ms),
            "cupti_ok": bool(cupti_ok),
            "cupti_reason": str(cupti_reason),
            "cupti_keys": tuple(self.obs_spec.cupti_keys),
            "cupti_vec": [float(x) for x in np.asarray(cupti_vec, dtype=np.float32).tolist()],
            "nvml_vec": None if nvml_vec is None else [float(x) for x in np.asarray(nvml_vec, dtype=np.float32).tolist()],
        }

        return obs, info

    def step(self, action):
        self._step += 1

        a = decode_action(action)

        # Normalize previous action indices into [0,1]
        block_idx = int(action[0])
        reg_idx = int(action[1])
        self._prev_action_norm = np.array([
            block_idx / max(1, (self.action_space.nvec[0] - 1)),
            reg_idx / max(1, (self.action_space.nvec[1] - 1)),
        ], dtype=np.float32)

        measured_ms = self._measure_time_ms(self._kernel_name, block_size=a.block_size, reg_cap=a.reg_cap)
        # Cache successful measurement for fallback during CUDA degradation
        self._last_valid_time_ms = measured_ms
        rr = speedup_reward(baseline_ms=self._baseline_ms, measured_ms=measured_ms)

        # NVML: best-effort. For CUPTI runs, poll NVML during the `ncu` subprocess.
        if self.cfg.use_cupti and self.obs_spec.include_nvml:
            nvml_vec, cupti_res = self._collect_nvml_peak_during(
                lambda: self._collect_cupti_norm(self._kernel_name, block_size=a.block_size, reg_cap=a.reg_cap)
            )
            cupti_vec, cupti_reason, cupti_ok = cupti_res
        else:
            nvml_vec = self._collect_nvml_norm() if self.obs_spec.include_nvml else None
            cupti_vec, cupti_reason, cupti_ok = self._collect_cupti_norm(self._kernel_name, block_size=a.block_size, reg_cap=a.reg_cap)

        obs = build_observation(
            kernel_name=self._kernel_name,
            cupti_norm=cupti_vec,
            nvml_norm=nvml_vec,
            prev_action_norm=self._prev_action_norm,
            spec=self.obs_spec,
        )

        terminated = False
        truncated = self._step >= int(self.cfg.max_steps)

        info: Dict[str, Any] = {
            "kernel": self._kernel_name,
            "matrix_size": int(self.cfg.matrix_size),
            "block_size": int(a.block_size),
            "reg_cap": int(a.reg_cap),
            "time_ms": float(measured_ms),
            "baseline_ms": float(self._baseline_ms),
            "speedup": float(rr.speedup),
            "cupti_ok": bool(cupti_ok),
            "cupti_reason": str(cupti_reason),
            "cupti_keys": tuple(self.obs_spec.cupti_keys),
            "cupti_vec": [float(x) for x in np.asarray(cupti_vec, dtype=np.float32).tolist()],
            "nvml_vec": None if nvml_vec is None else [float(x) for x in np.asarray(nvml_vec, dtype=np.float32).tolist()],
        }

        return obs, float(rr.reward), terminated, truncated, info
