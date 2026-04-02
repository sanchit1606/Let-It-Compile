"""CUPTI Access Test.

Modes:
- Default: run a simple Numba CUDA kernel and verify correctness.
- `--ncu`: run a Nsight Compute metric smoke test and print a clear message if
    `ERR_NVGPUCTRPERM` blocks performance counter collection.
"""

from __future__ import annotations

import sys

import numba.cuda as cuda
import numpy as np

@cuda.jit
def add(a, b, c):
    """Simple kernel to test CUPTI profiling"""
    i = cuda.grid(1)
    if i < a.shape[0]:
        c[i] = a[i] + b[i]

def test_cupti_access():
    """Test CUPTI by running a simple kernel"""
    print("=" * 70)
    print("CUPTI Access Test - Simple Vector Addition")
    print("=" * 70)
    
    n = 1024
    
    # Create input arrays on device
    a = cuda.to_device(np.ones(n, dtype=np.float32))
    b = cuda.to_device(np.ones(n, dtype=np.float32))
    c = cuda.device_array(n, dtype=np.float32)
    
    print(f"Input size: {n} elements")
    print(f"Block/Grid: [32, 32]")
    
    # Run kernel
    add[32, 32](a, b, c)
    
    # Verify result
    result = c.copy_to_host()
    expected = 2.0
    
    if np.allclose(result, expected):
        print(f"✓ Kernel executed successfully")
        print(f"✓ Results verified: all values ≈ {expected}")
        print("\n✓ CUPTI profiling should work with this kernel!")
        return True
    else:
        print(f"✗ Kernel execution failed")
        print(f"Expected: {expected}, Got: {result[0]}")
        return False

if __name__ == "__main__":
    try:
        if "--ncu" in sys.argv:
            try:
                from profiling.ncu_utils import ncu_metric_smoke_test

                print("=" * 70)
                print("Nsight Compute (ncu) CUPTI Permission Smoke Test")
                print("=" * 70)

                res = ncu_metric_smoke_test(timeout_s=60)
                if res.ok:
                    print("✓ CUPTI perf counters via ncu: OK")
                    sys.exit(0)

                if res.reason in {"permission_denied", "permission_denied_not_admin"}:
                    print("✗ CUPTI perf counters blocked: ERR_NVGPUCTRPERM")
                    if sys.platform.startswith("win"):
                        print("  Fix: re-run from Command Prompt 'Run as administrator'.")
                    sys.exit(2)

                print(f"✗ ncu smoke test failed: {res.reason}")
                sys.exit(1)
            except ImportError as e:
                print(f"✗ Cannot import ncu utilities: {e}")
                sys.exit(1)
        else:
            test_cupti_access()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
