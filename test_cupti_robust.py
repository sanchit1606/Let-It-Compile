"""
Robust CUPTI Access Test with CUDA Initialization
"""

import sys
print("Python version:", sys.version)

# Initialize CUDA context first
try:
    import numba
    print(f"Numba version: {numba.__version__}")
    
    from numba import cuda
    print("✓ Numba CUDA module imported")
    
    # Test CUDA availability
    try:
        num_devices = cuda.device_count()
        print(f"✓ CUDA available: {num_devices} device(s)")
        
        # Get device info
        device = cuda.get_current_device()
        print(f"✓ GPU Device: {device.name}")
        print(f"✓ Compute Capability: {device.compute_capability}")
    except Exception as e:
        print(f"✓ Numba CUDA imported (device check: {e})")
    
except Exception as e:
    print(f"✗ CUDA initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now import numpy
import numpy as np

@cuda.jit
def add(a, b, c):
    """Simple kernel to test CUPTI profiling"""
    i = cuda.grid(1)
    if i < a.shape[0]:
        c[i] = a[i] + b[i]

def test_cupti_access():
    """Test CUPTI by running a simple kernel"""
    print("\n" + "=" * 70)
    print("CUPTI Access Test - Simple Vector Addition")
    print("=" * 70)
    
    try:
        n = 1024
        
        print(f"\nAllocating device memory for {n} elements...")
        # Create input arrays on device
        a = cuda.to_device(np.ones(n, dtype=np.float32))
        b = cuda.to_device(np.ones(n, dtype=np.float32))
        c = cuda.device_array(n, dtype=np.float32)
        print("✓ Device memory allocated")
        
        print(f"Running kernel with grid=[32, 32]...")
        # Run kernel
        add[32, 32](a, b, c)
        print("✓ Kernel executed")
        
        print("Copying results back to host...")
        # Verify result
        result = c.copy_to_host()
        expected = 2.0
        print("✓ Results copied to host")
        
        if np.allclose(result, expected):
            print(f"✓ Kernel executed successfully")
            print(f"✓ Results verified: all values ≈ {expected}")
            print("\n✓ CUPTI profiling should work with this kernel!")
            return True
        else:
            print(f"✗ Kernel execution failed")
            print(f"Expected: {expected}, Got: {result[0]}")
            return False
            
    except Exception as e:
        print(f"✗ Error during kernel execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_cupti_access()
        if success:
            print("\n" + "=" * 70)
            print("✓ CUPTI Access Test PASSED")
            print("=" * 70)
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
