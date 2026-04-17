"""Row-wise softmax kernel.

This kernel computes softmax per row: y[i,j] = exp(x[i,j]) / sum_k(exp(x[i,k]))

Notes for Phase 0 (Windows/WDDM):
- Avoid massively redundant work (per-element recomputation of row max/sum)
    which can lead to long-running kernels and CUDA context instability.
- This implementation computes row max and sum once per block (thread 0), then
    has threads write output columns in a stride loop.
"""

import numba.cuda as cuda
import numpy as np
from numba import float32
import math
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Valid register cap range
MIN_REG_CAP = 8
MAX_REG_CAP = 256


@cuda.jit
def softmax_kernel(input_arr, output, n_rows, n_cols):
    """
    Compute row-wise softmax.

    Grid layout:
      - 1 block per row
      - threads cover columns in a stride loop

    This is not the fastest possible softmax, but it's stable and predictable
    for the Phase 0 sweep.
    """

    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    if row >= n_rows:
        return

    sh_max = cuda.shared.array(1, dtype=float32)
    sh_sum = cuda.shared.array(1, dtype=float32)

    # Compute row max and denominator once per row.
    if tid == 0:
        max_val = float32(-1e38)
        for j in range(n_cols):
            v = input_arr[row, j]
            if v > max_val:
                max_val = v
        sh_max[0] = max_val

        denom = float32(0.0)
        for j in range(n_cols):
            denom += math.exp(input_arr[row, j] - max_val)
        sh_sum[0] = denom

    cuda.syncthreads()

    max_val = sh_max[0]
    denom = sh_sum[0]

    # Write output for this row.
    for col in range(tid, n_cols, cuda.blockDim.x):
        output[row, col] = math.exp(input_arr[row, col] - max_val) / denom


@lru_cache(maxsize=None)
def get_softmax_kernel(reg_cap: int):
    """Return a softmax kernel compiled with an optional register cap."""

    if not reg_cap or reg_cap <= 0:
        return softmax_kernel
    
    # Clamp to valid range
    reg_cap = max(MIN_REG_CAP, min(int(reg_cap), MAX_REG_CAP))
    return cuda.jit(max_registers=reg_cap)(softmax_kernel.py_func)


def _clear_jit_cache_on_error():
    """Clear the JIT cache if CUDA context was reset."""
    try:
        get_softmax_kernel.cache_clear()
        logger.debug("Cleared softmax JIT cache due to CUDA context reset")
    except Exception as e:
        logger.debug(f"Could not clear JIT cache: {e}")


def run_softmax(N: int, block_size: int = 256, warmup: int = 1, reg_cap: int = 0):
    """
    Setup and return softmax input array and kernel parameters.

    Args:
        N: Number of rows (and columns for square matrix)
        block_size: Threads per block
        warmup: Number of warmup iterations
        reg_cap: Register cap (0 for default)

    Returns:
        (x_dev, out_dev, grid, block, kernel_fn)
    """
    # Input: N×N matrix
    x = cuda.to_device(np.random.randn(N, N).astype(np.float32))
    out = cuda.device_array((N, N), dtype=np.float32)

    # One block per row; threads cover columns in a stride loop.
    grid = (N,)
    block = (min(block_size, N),)

    kernel_fn = get_softmax_kernel(int(reg_cap) if reg_cap else 0)

    try:
        for _ in range(warmup):
            kernel_fn[grid, block](x, out, np.int32(N), np.int32(N))
    except Exception as e:
        logger.error(f"Softmax warmup failed with reg_cap={reg_cap}: {e}")
        _clear_jit_cache_on_error()
        raise

    # Ensure warmup finishes before returning buffers to caller.
    cuda.default_stream().synchronize()

    return x, out, grid, block, kernel_fn
