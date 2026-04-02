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
    return cuda.jit(max_registers=int(reg_cap))(reduction_kernel.py_func)


def run_reduction(N: int, block_size: int = 256, warmup: int = 1, reg_cap: int = 0):
    """
    Setup and return reduction input array and kernel parameters.

    Args:
        N: Total number of elements to reduce
        block_size: Threads per block
        warmup: Number of warmup iterations

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

    for _ in range(warmup):
        # Reset accumulator before each warmup launch.
        out.copy_to_device(out_host_zero)
        kernel_fn[grid, block](x, out, np.int32(N))
    cuda.default_stream().synchronize()

    return x, out, grid, block, kernel_fn
