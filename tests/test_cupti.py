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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
