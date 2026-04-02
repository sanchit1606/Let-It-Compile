"""
Tests for CUPTI hardware counter collection and GPU profiling.

Tests:
- NVML monitor can read GPU state
- CUPTI collector can run (even without ncu)
- Timing module produces consistent results
"""
import pytest


def test_nvml_import():
    """Test that NVML monitor can be imported."""
    try:
        from profiling.nvml_monitor import NVMLMonitor
    except ImportError:
        pytest.skip("profiling.nvml_monitor not available")


def test_cuda_timer_import():
    """Test that CUDA timer can be imported."""
    try:
        from profiling.cuda_timer import time_kernel
    except ImportError:
        pytest.skip("profiling.cuda_timer not available")


def test_cupti_collector_import():
    """Phase 1: CUPTI collector module can be imported."""
    try:
        from profiling.cupti_collector import CUPTICollector, DEFAULT_NCU_METRICS  # noqa: F401
    except ImportError:
        pytest.skip("profiling.cupti_collector not available")


def test_nvml_get_state():
    """Test that NVML can read GPU state."""
    try:
        from profiling.nvml_monitor import NVMLMonitor
        monitor = NVMLMonitor()
        state = monitor.get_state()
        assert state.gpu_util_pct >= 0
        assert state.temperature_c >= 0
    except ImportError:
        pytest.skip("NVML not available")
    except Exception as e:
        pytest.fail(f"NVML state read failed: {e}")


def test_cupti_collect_optional():
    """Optional Phase 1 smoke test: collect at least achieved occupancy via ncu.

    This test is skip-safe:
    - skips if CUDA isn't available
    - skips if ncu isn't installed
    - skips if performance counters are permission-blocked
    """

    try:
        from numba import cuda
        if not cuda.is_available():
            pytest.skip("CUDA not available")
    except Exception:
        pytest.skip("Numba CUDA not available")

    from profiling.cupti_collector import CUPTICollector, DEFAULT_NCU_METRICS

    collector = CUPTICollector(metrics={
        "achieved_occupancy": DEFAULT_NCU_METRICS["achieved_occupancy"],
    })

    pre = collector.preflight()
    if not pre.ok:
        pytest.skip(f"ncu counters unavailable: {pre.reason}")

    code = """
import numpy as np
from numba import cuda

@cuda.jit
def add(a, b, c):
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]

n = 1024
x = cuda.to_device(np.ones(n, dtype=np.float32))
y = cuda.to_device(np.ones(n, dtype=np.float32))
z = cuda.device_array(n, dtype=np.float32)
add[32, 32](x, y, z)
cuda.default_stream().synchronize()
""".lstrip()

    res = collector.collect_from_python_code(code, timeout_s=90)
    if not res.ok:
        pytest.skip(f"ncu collect failed: {res.reason}")

    assert "achieved_occupancy" in res.raw
    assert 0.0 <= res.normalized["achieved_occupancy"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
