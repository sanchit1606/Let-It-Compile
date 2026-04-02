"""Precise CUDA kernel timing using CUDA events.

Numba provides a lightweight CUDA event API (`cuda.event`) that yields accurate
kernel timings without CPU-side scheduling noise.
"""

import numba.cuda as cuda
import numpy as np
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List


# Some Windows driver/toolkit combinations have been observed to raise
# "OSError: exception: access violation reading ..." from CUDA event timing
# even when kernel execution itself is correct. In that case, we fall back
# to CPU-side timing with explicit synchronization.
_EVENT_TIMING_OK = True


@dataclass
class TimingResult:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    all_ms: List[float]


def time_kernel(kernel_fn, grid, block, args: tuple, warmup: int = 3, repeats: int = 10) -> TimingResult:
    """
    Time a CUDA kernel with warmup iterations.

    Args:
        kernel_fn: Compiled Numba CUDA kernel (result of @cuda.jit)
        grid: Grid dimensions tuple e.g. (16, 16) or integer
        block: Block dimensions tuple e.g. (16, 16) or integer
        args: Kernel arguments tuple
        warmup: Number of warmup iterations (not measured)
        repeats: Number of measured iterations

    Returns:
        TimingResult with statistics in milliseconds
    """
    # Warmup — triggers JIT compilation on first call
    for _ in range(warmup):
        kernel_fn[grid, block](*args)
    cuda.default_stream().synchronize()

    times = []
    global _EVENT_TIMING_OK

    # Prefer CUDA event timing when available; fall back to CPU timing if event
    # APIs misbehave on this platform/driver.
    if _EVENT_TIMING_OK:
        try:
            for _ in range(repeats):
                start_evt = cuda.event(timing=True)
                end_evt = cuda.event(timing=True)
                start_evt.record()
                kernel_fn[grid, block](*args)
                end_evt.record()
                end_evt.synchronize()
                times.append(cuda.event_elapsed_time(start_evt, end_evt))
        except Exception:
            # Disable event timing for the remainder of this process.
            _EVENT_TIMING_OK = False
            times.clear()

    if not times:
        for _ in range(repeats):
            start_t = time.perf_counter()
            kernel_fn[grid, block](*args)
            cuda.default_stream().synchronize()
            end_t = time.perf_counter()
            times.append((end_t - start_t) * 1000.0)

    if not times:
        raise RuntimeError("No successful kernel runs completed")

    times_arr = np.array(times)
    return TimingResult(
        mean_ms=float(np.mean(times_arr)),
        std_ms=float(np.std(times_arr)),
        min_ms=float(np.min(times_arr)),
        max_ms=float(np.max(times_arr)),
        all_ms=times
    )
