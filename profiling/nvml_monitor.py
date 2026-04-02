"""
Real-time GPU monitoring via pynvml (NVIDIA Management Library).
Used for lightweight state collection BETWEEN kernel runs.
Does not require profiling privileges unlike CUPTI.
"""

import pynvml
import numpy as np
from dataclasses import dataclass


@dataclass
class GPUState:
    gpu_util_pct: float       # % of time SM was executing at least one warp
    mem_util_pct: float       # % of time memory interface was active
    mem_used_mb: float        # Used VRAM in MB
    mem_total_mb: float       # Total VRAM in MB
    temperature_c: float      # GPU temperature
    power_w: float            # Current power draw in Watts
    clock_mhz: float          # Current graphics clock in MHz


class NVMLMonitor:
    """Lightweight real-time GPU monitor using NVML."""

    def __init__(self, device_index: int = 0, *, verbose: bool = False):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self._verbose = bool(verbose)
        if self._verbose:
            name = pynvml.nvmlDeviceGetName(self.handle)
            print(f"[NVML] Monitoring: {name}")

    def get_state(self) -> GPUState:
        """Sample current GPU state. Fast (<1ms)."""
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW → W
        except pynvml.NVMLError:
            power = 0.0
        clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)

        return GPUState(
            gpu_util_pct=float(util.gpu),
            mem_util_pct=float(util.memory),
            mem_used_mb=mem.used / 1024**2,
            mem_total_mb=mem.total / 1024**2,
            temperature_c=float(temp),
            power_w=float(power),
            clock_mhz=float(clock),
        )

    def to_vector(self) -> np.ndarray:
        """Return normalized state as numpy vector for RL state space."""
        state = self.get_state()
        return np.array([
            state.gpu_util_pct / 100.0,
            state.mem_util_pct / 100.0,
            state.mem_used_mb / state.mem_total_mb,
            state.temperature_c / 100.0,   # Normalize to ~[0,1]
        ], dtype=np.float32)

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
