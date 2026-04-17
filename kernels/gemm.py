"""
Tiled Matrix Multiplication (GEMM) kernel.

Parameterized by:
  - block_size: threads per block (square tile dimension)
  - reg_cap: maximum registers per thread hint (--maxrregcount equivalent)

This is a basic tiled GEMM that demonstrates how register allocation
affects occupancy and runtime.
"""

import numba.cuda as cuda
import numba as nb
import numpy as np
from numba import float32
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# ── Tile size — must be a compile-time constant for CUDA ────────────
TILE_SIZE = 16  # 16x16 tile = 256 threads per block

# Valid register cap range (per NVIDIA architecture best practices)
MIN_REG_CAP = 8
MAX_REG_CAP = 256


@cuda.jit
def gemm_kernel_16(A, B, C, N):
    """Tiled GEMM with 16x16 tile. Works with block_size=256 (16x16 threads)."""
    TILE = 16
    # Shared memory tiles
    sA = cuda.shared.array(shape=(TILE, TILE), dtype=float32)
    sB = cuda.shared.array(shape=(TILE, TILE), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx

    acc = float32(0.0)

    for k in range((N + TILE - 1) // TILE):
        # Load tile from A
        if row < N and k * TILE + tx < N:
            sA[ty, tx] = A[row, k * TILE + tx]
        else:
            sA[ty, tx] = float32(0.0)

        # Load tile from B
        if k * TILE + ty < N and col < N:
            sB[ty, tx] = B[k * TILE + ty, col]
        else:
            sB[ty, tx] = float32(0.0)

        cuda.syncthreads()

        # Compute partial dot product
        for i in range(TILE):
            acc += sA[ty, i] * sB[i, tx]

        cuda.syncthreads()

    if row < N and col < N:
        C[row, col] = acc


@cuda.jit
def gemm_kernel_8(A, B, C, N):
    """Tiled GEMM with 8x8 tile. Works with block_size=64 (8x8 threads)."""
    TILE = 8
    sA = cuda.shared.array(shape=(TILE, TILE), dtype=float32)
    sB = cuda.shared.array(shape=(TILE, TILE), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx

    acc = float32(0.0)

    for k in range((N + TILE - 1) // TILE):
        if row < N and k * TILE + tx < N:
            sA[ty, tx] = A[row, k * TILE + tx]
        else:
            sA[ty, tx] = float32(0.0)

        if k * TILE + ty < N and col < N:
            sB[ty, tx] = B[k * TILE + ty, col]
        else:
            sB[ty, tx] = float32(0.0)

        cuda.syncthreads()

        for i in range(TILE):
            acc += sA[ty, i] * sB[i, tx]

        cuda.syncthreads()

    if row < N and col < N:
        C[row, col] = acc


def get_gemm_kernel(block_size: int):
    """Get appropriate GEMM kernel for block size."""
    if block_size == 64:
        return gemm_kernel_8
    elif block_size in [128, 256, 512]:
        return gemm_kernel_16
    else:
        raise ValueError(f"Unsupported block_size: {block_size}. Use 64, 128, 256, or 512.")


def _validate_reg_cap(reg_cap: int) -> int:
    """Validate and clamp register cap to safe range.
    
    Args:
        reg_cap: Requested register cap (or 0 for default)
        
    Returns:
        Validated register cap value
    """
    if reg_cap <= 0:
        return 0  # Default (no cap)
    
    reg_cap = int(reg_cap)
    if reg_cap < MIN_REG_CAP or reg_cap > MAX_REG_CAP:
        clamped = max(MIN_REG_CAP, min(reg_cap, MAX_REG_CAP))
        logger.warning(f"Register cap {reg_cap} out of range [{MIN_REG_CAP}, {MAX_REG_CAP}]; clamping to {clamped}")
        return clamped
    
    return reg_cap


@lru_cache(maxsize=None)
def _get_gemm_kernel_with_reg_cap(block_size: int, reg_cap: int):
    """Return a GEMM kernel compiled with an optional register cap.

    Uses Numba's `max_registers` option (PTXAS --maxrregcount equivalent).
    """

    base = get_gemm_kernel(block_size)

    # 0 means "PTXAS default" (no cap).
    if not reg_cap or reg_cap <= 0:
        return base

    # Re-JIT the original Python function with a register cap.
    # Clamp to valid range before compilation
    reg_cap = _validate_reg_cap(reg_cap)
    if reg_cap <= 0:
        return base
    
    return cuda.jit(max_registers=reg_cap)(base.py_func)


def _clear_jit_cache_on_error():
    """Clear the JIT cache if CUDA context was reset."""
    try:
        _get_gemm_kernel_with_reg_cap.cache_clear()
        logger.debug("Cleared JIT cache due to CUDA context reset")
    except Exception as e:
        logger.debug(f"Could not clear JIT cache: {e}")


def run_gemm(N: int, block_size: int = 256, warmup: int = 1, reg_cap: int = 0):
    """
    Setup and return GEMM input arrays and kernel parameters.

    Args:
        N: Matrix dimension (N×N)
        block_size: Threads per block
        warmup: Number of warmup iterations (triggers JIT compilation)
        reg_cap: Register cap (0 for default)

    Returns:
        (A_dev, B_dev, C_dev, grid, block, kernel_fn)
        
    Note: Caller must synchronize and free A, B, C device arrays after use.
    """
    # Validate and clamp register cap
    reg_cap = _validate_reg_cap(reg_cap)
    
    kernel_fn = _get_gemm_kernel_with_reg_cap(block_size, reg_cap)

    # Determine tile size from kernel
    tile = 8 if block_size == 64 else 16

    # Create random matrices on GPU
    A = cuda.to_device(np.random.randn(N, N).astype(np.float32))
    B = cuda.to_device(np.random.randn(N, N).astype(np.float32))
    C = cuda.device_array((N, N), dtype=np.float32)

    grid = ((N + tile - 1) // tile, (N + tile - 1) // tile)
    block = (tile, tile)

    # Warmup (triggers JIT compilation)
    try:
        for _ in range(warmup):
            kernel_fn[grid, block](A, B, C, np.int32(N))
        cuda.default_stream().synchronize()
    except Exception as e:
        # If warmup fails, clean up and signal error
        try:
            A.copy_to_host()
            B.copy_to_host()
            C.copy_to_host()
        except:
            pass
        logger.error(f"GEMM warmup failed with reg_cap={reg_cap}: {e}")
        _clear_jit_cache_on_error()
        raise

    return A, B, C, grid, block, kernel_fn
