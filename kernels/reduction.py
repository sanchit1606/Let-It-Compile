"""
Parallel Reduction kernel (sum).
Classic memory-bound benchmark — good contrast to GEMM.

This kernel demonstrates how register allocation affects performance
on memory-bound operations.
"""

import numba.cuda as cuda
import numpy as np
from numba import float32
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Valid register cap range
MIN_REG_CAP = 8
MAX_REG_CAP = 256


@cuda.jit
def reduction_kernel(input_arr, output, n):
    """
    Parallel reduction using shared memory and tree reduction.
    Each block reduces its chunk and writes partial sum to output.
    """
    BLOCK = cuda.blockDim.x
    tid   = cuda.threadIdx.x
    bid   = cuda.blockIdx.x
    gid   = bid * BLOCK + tid

    # Shared memory — sized for max block size
    smem = cuda.shared.array(1024, dtype=float32)

    # Load from global memory
    if gid < n:
        smem[tid] = input_arr[gid]
    else:
        smem[tid] = float32(0.0)
    cuda.syncthreads()

    # Reduction in shared memory (tree reduction)
    s = BLOCK // 2
    while s > 0:
        if tid < s:
            smem[tid] += smem[tid + s]
        cuda.syncthreads()
        s //= 2

    # Write block result to global output
    if tid == 0:
        cuda.atomic.add(output, 0, smem[0])


@lru_cache(maxsize=None)
def get_reduction_kernel(reg_cap: int):
    """Return a reduction kernel compiled with an optional register cap."""

    if not reg_cap or reg_cap <= 0:
        return reduction_kernel
    
    # Clamp to valid range
    reg_cap = max(MIN_REG_CAP, min(int(reg_cap), MAX_REG_CAP))
    return cuda.jit(max_registers=reg_cap)(reduction_kernel.py_func)


def _clear_jit_cache_on_error():
    """Clear the JIT cache if CUDA context was reset."""
    try:
        get_reduction_kernel.cache_clear()
        logger.debug("Cleared reduction JIT cache due to CUDA context reset")
    except Exception as e:
        logger.debug(f"Could not clear JIT cache: {e}")


def run_reduction(N: int, block_size: int = 256, warmup: int = 1, reg_cap: int = 0):
    """
    Setup and return reduction input array and kernel parameters.

    Args:
        N: Total number of elements to reduce
        block_size: Threads per block
        warmup: Number of warmup iterations
        reg_cap: Register cap (0 for default)

    Returns:
        (x_dev, out_dev, grid, block, kernel_fn)
    """
    x = cuda.to_device(np.random.randn(N).astype(np.float32))
    out_host_zero = np.zeros(1, dtype=np.float32)
    out = cuda.to_device(out_host_zero)

    blocks = (N + block_size - 1) // block_size
    grid = (blocks,)
    block = (block_size,)

    kernel_fn = get_reduction_kernel(int(reg_cap) if reg_cap else 0)

    try:
        for _ in range(warmup):
            # Recreate output array instead of calling copy_to_device() which can fail
            # on corrupted contexts
            try:
                # Try copy_to_device first (faster)
                out.copy_to_device(out_host_zero)
            except (OSError, RuntimeError):
                # If context is degraded, recreate the array
                out = cuda.to_device(out_host_zero)
            
            kernel_fn[grid, block](x, out, np.int32(N))
            # Synchronize after each warmup iteration to detect errors early
            cuda.default_stream().synchronize()
        cuda.default_stream().synchronize()
    except Exception as e:
        logger.error(f"Reduction warmup failed with reg_cap={reg_cap}: {e}")
        _clear_jit_cache_on_error()
        raise

    # Important: the reduction uses atomic accumulation into out[0].
    # Ensure the returned output buffer is clean for the caller's measured launch.
    try:
        out.copy_to_device(out_host_zero)
    except (OSError, RuntimeError):
        # Context degradation - recreate
        out = cuda.to_device(out_host_zero)

    return x, out, grid, block, kernel_fn
