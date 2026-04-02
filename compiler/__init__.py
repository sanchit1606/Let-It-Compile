"""
GPU compilation control layer.

Modules:
- ptxas_controller.py: PTXAS --maxrregcount control and occupancy calculation
- numba_compiler.py: Numba JIT compilation with configurable parameters
- ir_extractor.py: Extract PTX/LLVM IR from compiled kernels
"""
