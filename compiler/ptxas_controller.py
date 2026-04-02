"""
PTXAS controller for computing theoretical occupancy.

For Phase 0, we don't need to actually call ptxas (which would require low-level compilation control).
Instead, we compute theoretical occupancy using known hardware limits and register counts.
"""


class OccupancyCalculator:
    """
    Compute theoretical occupancy for sm_86 (RTX 3050 Ti Ampere).

    Hardware specs for sm_86:
      - Max warps per SM: 48
      - Max threads per SM: 1536
      - Max registers per SM: 65536
      - Max shared memory per SM: 100 KB (default), up to 164 KB (opt-in)
      - Max blocks per SM: 16
      - Max threads per block: 1024
      - Warp size: 32
    """

    # sm_86 hardware constants
    MAX_WARPS_PER_SM      = 48
    MAX_REGISTERS_PER_SM  = 65536
    MAX_THREADS_PER_SM    = 1536
    MAX_BLOCKS_PER_SM     = 16
    MAX_SHARED_MEM_PER_SM = 100 * 1024  # bytes
    WARP_SIZE             = 32

    @staticmethod
    def compute_occupancy(registers_per_thread: int,
                         block_size: int,
                         shared_mem_bytes: int = 0) -> float:
        """
        Compute theoretical occupancy for a given register and block configuration.

        Args:
            registers_per_thread: Registers used per thread (estimated or measured)
            block_size: Number of threads per block
            shared_mem_bytes: Shared memory used per block in bytes

        Returns:
            Occupancy as fraction in [0, 1]
        """
        warps_per_block = (block_size + OccupancyCalculator.WARP_SIZE - 1) // OccupancyCalculator.WARP_SIZE

        # Limit 1: Register constraint
        if registers_per_thread > 0:
            warps_reg_limit = (
                OccupancyCalculator.MAX_REGISTERS_PER_SM
                // (registers_per_thread * OccupancyCalculator.WARP_SIZE)
            )
        else:
            warps_reg_limit = OccupancyCalculator.MAX_WARPS_PER_SM

        # Limit 2: Thread constraint
        blocks_thread_limit = OccupancyCalculator.MAX_THREADS_PER_SM // block_size
        warps_thread_limit = blocks_thread_limit * warps_per_block

        # Limit 3: Block count constraint
        warps_block_limit = OccupancyCalculator.MAX_BLOCKS_PER_SM * warps_per_block

        # Limit 4: Shared memory constraint (rarely limiting on Ampere)
        if shared_mem_bytes > 0:
            blocks_smem_limit = OccupancyCalculator.MAX_SHARED_MEM_PER_SM // shared_mem_bytes
            warps_smem_limit = blocks_smem_limit * warps_per_block
        else:
            warps_smem_limit = OccupancyCalculator.MAX_WARPS_PER_SM

        # Active warps is the minimum of all constraints
        active_warps = min(
            warps_reg_limit,
            warps_thread_limit,
            warps_block_limit,
            warps_smem_limit,
            OccupancyCalculator.MAX_WARPS_PER_SM
        )

        # Occupancy = fraction of max warps utilized
        occupancy = active_warps / OccupancyCalculator.MAX_WARPS_PER_SM
        return occupancy

    @staticmethod
    def estimate_register_count(kernel_name: str, block_size: int) -> int:
        """
        Estimate register count per thread for a kernel.
        These are empirical estimates for the Phase 0 kernels.

        Args:
            kernel_name: "gemm", "reduction", or "softmax"
            block_size: Threads per block

        Returns:
            Estimated registers per thread
        """
        # These are typical values for our kernels
        # In production, use ptxas --verbose output for accuracy
        if kernel_name == "gemm":
            # GEMM is compute-heavy, uses many registers for loop unrolling
            # Typical: 64-100 registers per thread
            return min(100, 40 + block_size // 16)
        elif kernel_name == "reduction":
            # Reduction is simple, fewer registers
            # Typical: 20-40 registers per thread
            return min(45, 15 + block_size // 64)
        elif kernel_name == "softmax":
            # Mixed workload, medium register usage
            # Typical: 30-60 registers per thread
            return min(70, 25 + block_size // 32)
        else:
            return 40  # Default estimate
