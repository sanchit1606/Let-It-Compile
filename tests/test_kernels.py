"""
Tests for CUDA kernel implementations.

Tests:
- Numba CUDA JIT compilation works
- Kernels run without segfaults
- Output correctness (partial checks)
"""
import pytest


def _assert_allclose(a, b, *, rtol: float, atol: float):
    import numpy as np

    assert np.allclose(a, b, rtol=rtol, atol=atol)


def _assert_close_scalar(a, b, *, rtol: float, atol: float):
    import numpy as np

    assert np.isfinite(a)
    assert np.isclose(a, b, rtol=rtol, atol=atol)


def _skip_if_no_cuda():
    try:
        from numba import cuda

        if not cuda.is_available():
            pytest.skip("CUDA not available")
    except Exception:
        pytest.skip("Numba CUDA not available")


def test_numba_cuda_import():
    """Test that Numba CUDA is available."""
    _skip_if_no_cuda()


def test_gemm_import():
    """Test that GEMM kernel can be imported."""
    _skip_if_no_cuda()
    from kernels.gemm import run_gemm  # noqa: F401


def test_reduction_import():
    """Test that reduction kernel can be imported."""
    _skip_if_no_cuda()
    from kernels.reduction import run_reduction  # noqa: F401


def test_softmax_import():
    """Test that softmax kernel can be imported."""
    _skip_if_no_cuda()
    from kernels.softmax import run_softmax  # noqa: F401


@pytest.mark.parametrize("N", [15, 32, 33])
def test_gemm_correctness_fast(N: int):
    """Fast correctness: small + edge-case N (includes non-multiple-of-tile)."""

    _skip_if_no_cuda()

    import numpy as np

    from kernels.gemm import run_gemm

    # Baseline (no reg cap)
    A_dev, B_dev, C_dev, grid, block, kernel_fn = run_gemm(
        N, block_size=256, warmup=1, reg_cap=0
    )
    kernel_fn[grid, block](A_dev, B_dev, C_dev, np.int32(N))
    from numba import cuda

    cuda.default_stream().synchronize()

    A = A_dev.copy_to_host()
    B = B_dev.copy_to_host()
    C = C_dev.copy_to_host()
    C_ref = (A @ B).astype(np.float32)

    _assert_allclose(C, C_ref, rtol=1e-2, atol=1e-2)

    # Reg-capped variant should also be numerically correct
    A2_dev, B2_dev, C2_dev, grid2, block2, kernel_fn2 = run_gemm(
        N, block_size=256, warmup=1, reg_cap=32
    )
    kernel_fn2[grid2, block2](A2_dev, B2_dev, C2_dev, np.int32(N))
    cuda.default_stream().synchronize()
    A2 = A2_dev.copy_to_host()
    B2 = B2_dev.copy_to_host()
    C2 = C2_dev.copy_to_host()
    C2_ref = (A2 @ B2).astype(np.float32)
    _assert_allclose(C2, C2_ref, rtol=1e-2, atol=1e-2)


def test_gemm_correctness_block64_fast():
    """Fast correctness: exercise the 8x8 tiled GEMM path (block_size=64)."""

    _skip_if_no_cuda()

    import numpy as np
    from numba import cuda

    from kernels.gemm import run_gemm

    N = 17  # non-multiple of tile=8
    A_dev, B_dev, C_dev, grid, block, kernel_fn = run_gemm(
        N, block_size=64, warmup=1, reg_cap=0
    )
    kernel_fn[grid, block](A_dev, B_dev, C_dev, np.int32(N))
    cuda.default_stream().synchronize()

    A = A_dev.copy_to_host()
    B = B_dev.copy_to_host()
    C = C_dev.copy_to_host()
    C_ref = (A @ B).astype(np.float32)
    _assert_allclose(C, C_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("N", [1, 255, 256, 257, 4096])
def test_reduction_correctness_fast(N: int):
    """Fast correctness: small + edge-case N around block boundaries."""

    _skip_if_no_cuda()

    import numpy as np
    from numba import cuda

    from kernels.reduction import run_reduction

    x_dev, out_dev, grid, block, kernel_fn = run_reduction(
        N, block_size=256, warmup=1, reg_cap=0
    )
    kernel_fn[grid, block](x_dev, out_dev, np.int32(N))
    cuda.default_stream().synchronize()

    x = x_dev.copy_to_host()
    out = out_dev.copy_to_host()[0]
    ref = float(np.sum(x, dtype=np.float32))
    _assert_close_scalar(out, ref, rtol=1e-2, atol=1e-2)

    # Reg-capped variant should also be correct
    x2_dev, out2_dev, grid2, block2, kernel_fn2 = run_reduction(
        N, block_size=256, warmup=1, reg_cap=32
    )
    kernel_fn2[grid2, block2](x2_dev, out2_dev, np.int32(N))
    cuda.default_stream().synchronize()
    x2 = x2_dev.copy_to_host()
    out2 = out2_dev.copy_to_host()[0]
    ref2 = float(np.sum(x2, dtype=np.float32))
    _assert_close_scalar(out2, ref2, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("N", [1, 32, 33])
def test_softmax_sums_to_one_fast(N: int):
    """Fast correctness: invariants on small + edge-case N."""

    _skip_if_no_cuda()

    import numpy as np
    from numba import cuda

    from kernels.softmax import run_softmax

    x_dev, out_dev, grid, block, kernel_fn = run_softmax(N, block_size=256, warmup=1, reg_cap=0)
    kernel_fn[grid, block](x_dev, out_dev, np.int32(N), np.int32(N))
    cuda.default_stream().synchronize()

    out = out_dev.copy_to_host()
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)

    row_sums = out.sum(axis=1)
    _assert_allclose(row_sums, 1.0, rtol=1e-2, atol=1e-2)

    # Reg-capped variant should also satisfy the softmax invariants
    x2_dev, out2_dev, grid2, block2, kernel_fn2 = run_softmax(N, block_size=256, warmup=1, reg_cap=32)
    kernel_fn2[grid2, block2](x2_dev, out2_dev, np.int32(N), np.int32(N))
    cuda.default_stream().synchronize()
    out2 = out2_dev.copy_to_host()
    assert np.all(np.isfinite(out2))
    assert np.all(out2 >= 0.0)
    row_sums2 = out2.sum(axis=1)
    _assert_allclose(row_sums2, 1.0, rtol=1e-2, atol=1e-2)


@pytest.mark.slow
def test_gemm_correctness_large():
    """Slow validation: larger N similar to benchmark-scale runs."""

    _skip_if_no_cuda()

    import numpy as np
    from numba import cuda

    from kernels.gemm import run_gemm

    N = 256

    A_dev, B_dev, C_dev, grid, block, kernel_fn = run_gemm(
        N, block_size=256, warmup=1, reg_cap=0
    )
    kernel_fn[grid, block](A_dev, B_dev, C_dev, np.int32(N))
    cuda.default_stream().synchronize()

    A = A_dev.copy_to_host()
    B = B_dev.copy_to_host()
    C = C_dev.copy_to_host()
    C_ref = (A @ B).astype(np.float32)
    _assert_allclose(C, C_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.slow
def test_reduction_correctness_large():
    """Slow validation: large N spanning many blocks."""

    _skip_if_no_cuda()

    import numpy as np
    from numba import cuda

    from kernels.reduction import run_reduction

    N = 1_000_000

    x_dev, out_dev, grid, block, kernel_fn = run_reduction(
        N, block_size=256, warmup=1, reg_cap=0
    )
    kernel_fn[grid, block](x_dev, out_dev, np.int32(N))
    cuda.default_stream().synchronize()

    x = x_dev.copy_to_host()
    out = float(out_dev.copy_to_host()[0])
    ref = float(np.sum(x, dtype=np.float32))
    _assert_close_scalar(out, ref, rtol=5e-2, atol=1.0)


@pytest.mark.slow
def test_softmax_sums_to_one_large():
    """Slow validation: invariants on a larger softmax."""

    _skip_if_no_cuda()

    import numpy as np
    from numba import cuda

    from kernels.softmax import run_softmax

    N = 256
    x_dev, out_dev, grid, block, kernel_fn = run_softmax(N, block_size=256, warmup=1, reg_cap=0)
    kernel_fn[grid, block](x_dev, out_dev, np.int32(N), np.int32(N))
    cuda.default_stream().synchronize()

    out = out_dev.copy_to_host()
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    row_sums = out.sum(axis=1)
    _assert_allclose(row_sums, 1.0, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
