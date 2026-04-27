"""
Extract PTX from compiled Numba CUDA kernels and parse structural features.

These features are used to build graph representations for the GNN encoder,
upgrading the RL agent's state from PMU-counters-only to
PMU-counters + kernel-structure.

Usage:
    from compiler.ir_extractor import extract_ptx, ptx_to_graph_features, ptx_to_pyg_graph

    # Extract PTX from a kernel
    ptx = extract_ptx(gemm_kernel_16)

    # Get flat feature dict (for debugging / analysis)
    features = ptx_to_graph_features(ptx)

    # Get PyG Data object (for GNN encoder)
    graph = ptx_to_pyg_graph(ptx)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch_geometric.data import Data

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


# ── PTX extraction ──────────────────────────────────────────────────


def extract_ptx(kernel_fn, arg_types: Optional[tuple] = None,
                cc: Tuple[int, int] = (8, 6)) -> str:
    """
    Extract PTX source from a Numba CUDA kernel.

    Args:
        kernel_fn: Numba CUDA kernel function (decorated with @cuda.jit).
                   Must have a `.py_func` attribute.
        arg_types: Tuple of Numba types matching kernel signature.
                   If None, uses a default signature for the project's kernels.
        cc: Compute capability tuple, default (8,6) for RTX 3050 Ti (sm_86).

    Returns:
        PTX source code as string.
    """
    from numba.cuda import compiler as cuda_compiler
    from numba import types

    pyfunc = kernel_fn.py_func if hasattr(kernel_fn, "py_func") else kernel_fn

    if arg_types is None:
        # Default: (float32[:,:], float32[:,:], float32[:,:], int32)
        # Works for gemm; reduction/softmax have similar signatures.
        arg_types = (
            types.Array(types.float32, 2, "C"),
            types.Array(types.float32, 2, "C"),
            types.Array(types.float32, 2, "C"),
            types.int32,
        )

    ptx, _resty = cuda_compiler.compile_ptx(
        pyfunc=pyfunc,
        sig=arg_types,
        cc=cc,
        device=False,
        fastmath=False,
    )

    # Numba may return bytes on some versions
    if isinstance(ptx, bytes):
        ptx = ptx.decode("utf-8", errors="replace")

    return ptx


# ── Flat feature extraction ─────────────────────────────────────────


def ptx_to_graph_features(ptx_source: str) -> Dict[str, float]:
    """
    Extract structural features from PTX source for GNN input.

    Returns a dict of normalized scalar features suitable for building
    a PyTorch Geometric Data object.  This is a simplified version for
    the prototype.

    Feature set:
      - n_instructions: total non-directive, non-comment lines
      - n_loads: ld.* instructions (global/shared/local memory reads)
      - n_stores: st.* instructions (memory writes)
      - n_fma: fused multiply-add / multiply-add instructions
      - n_branches: branch and set-predicate instructions
      - n_sync: bar.sync (syncthreads) calls
      - n_registers: float32 register count from PTX header
      - shared_mem_bytes: shared memory declared in PTX
      - n_loop_bodies: estimated from backward branch patterns
      - arithmetic_intensity_proxy: n_fma / (n_loads + n_stores)
    """
    lines = ptx_source.split("\n")

    features: Dict[str, float] = {
        "n_instructions": 0,
        "n_loads": 0,
        "n_stores": 0,
        "n_fma": 0,
        "n_branches": 0,
        "n_sync": 0,
        "n_registers": 0,
        "shared_mem_bytes": 0,
        "n_loop_bodies": 0,
        "n_add": 0,
        "n_mul": 0,
        "n_cvt": 0,        # type conversion instructions
        "n_mov": 0,         # move instructions
        "n_setp": 0,        # set predicate (comparison)
    }

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("."):
            continue

        features["n_instructions"] += 1

        # Memory operations
        if "ld." in stripped:
            features["n_loads"] += 1
        if "st." in stripped:
            features["n_stores"] += 1

        # Compute operations
        if "fma" in stripped or "mad" in stripped:
            features["n_fma"] += 1
        if stripped.startswith("add") or "\tadd" in stripped:
            features["n_add"] += 1
        if stripped.startswith("mul") or "\tmul" in stripped:
            features["n_mul"] += 1

        # Control flow
        if stripped.startswith("bra") or stripped.startswith("@"):
            features["n_branches"] += 1
        if stripped.startswith("setp") or "\tsetp" in stripped:
            features["n_setp"] += 1

        # Synchronization
        if "bar.sync" in stripped:
            features["n_sync"] += 1

        # Data movement
        if stripped.startswith("cvt") or "\tcvt" in stripped:
            features["n_cvt"] += 1
        if stripped.startswith("mov") or "\tmov" in stripped:
            features["n_mov"] += 1

    # Extract register counts from PTX header (.reg directives)
    # Look for all register types: .f32, .f64, .b32, .b64, .pred
    total_regs = 0
    for match in re.finditer(r"\.reg\s+\.\w+\s+%\w+<(\d+)>", ptx_source):
        total_regs += int(match.group(1))
    features["n_registers"] = total_regs

    # Extract shared memory size
    smem_match = re.search(
        r"\.shared\s+\.align\s+\d+\s+\.b8\s+\w+\[(\d+)\]", ptx_source
    )
    if smem_match:
        features["shared_mem_bytes"] = int(smem_match.group(1))

    # Estimate loop bodies from backward branches (bra targets before the bra)
    features["n_loop_bodies"] = max(0, features["n_sync"] // 2)

    # Arithmetic intensity proxy
    total_mem_ops = features["n_loads"] + features["n_stores"]
    if total_mem_ops > 0:
        features["arithmetic_intensity_proxy"] = features["n_fma"] / total_mem_ops
    else:
        features["arithmetic_intensity_proxy"] = 0.0

    return features


# ── Graph construction for PyG ───────────────────────────────────────


# The 10 node-level features used for the GNN
_NODE_FEATURE_KEYS = [
    "n_instructions",
    "n_loads",
    "n_stores",
    "n_fma",
    "n_add",
    "n_mul",
    "n_branches",
    "n_sync",
    "n_cvt",
    "n_mov",
]

# Number of node features in the GNN input
NODE_FEATURE_DIM = len(_NODE_FEATURE_KEYS)


def _build_basic_blocks(ptx_source: str) -> List[Dict[str, float]]:
    """
    Split PTX into basic blocks and compute per-block feature vectors.

    A basic block boundary occurs at:
      - Labels (e.g., `$L__BB0_1:`)
      - Branch instructions (bra)
      - bar.sync (synchronization barrier)

    Returns a list of feature dicts, one per basic block.
    """
    lines = ptx_source.split("\n")

    blocks: List[List[str]] = []
    current_block: List[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines, comments, and directives
        if not stripped or stripped.startswith("//"):
            continue

        # Label → start a new block
        if stripped.endswith(":") and not stripped.startswith("."):
            if current_block:
                blocks.append(current_block)
            current_block = [stripped]
            continue

        # Directives are metadata, not instructions
        if stripped.startswith("."):
            continue

        current_block.append(stripped)

        # Branch or sync → end block
        if stripped.startswith("bra") or "bar.sync" in stripped or stripped.startswith("ret"):
            blocks.append(current_block)
            current_block = []

    if current_block:
        blocks.append(current_block)

    # Ensure we have at least one block
    if not blocks:
        blocks = [["nop"]]

    # Compute per-block features
    block_features: List[Dict[str, float]] = []
    for block_lines in blocks:
        feats = {k: 0.0 for k in _NODE_FEATURE_KEYS}
        for inst in block_lines:
            if inst.endswith(":"):
                continue  # label, not an instruction
            feats["n_instructions"] += 1
            if "ld." in inst:
                feats["n_loads"] += 1
            if "st." in inst:
                feats["n_stores"] += 1
            if "fma" in inst or "mad" in inst:
                feats["n_fma"] += 1
            if inst.startswith("add") or "\tadd" in inst:
                feats["n_add"] += 1
            if inst.startswith("mul") or "\tmul" in inst:
                feats["n_mul"] += 1
            if inst.startswith("bra") or inst.startswith("@") or inst.startswith("setp"):
                feats["n_branches"] += 1
            if "bar.sync" in inst:
                feats["n_sync"] += 1
            if inst.startswith("cvt") or "\tcvt" in inst:
                feats["n_cvt"] += 1
            if inst.startswith("mov") or "\tmov" in inst:
                feats["n_mov"] += 1
        block_features.append(feats)

    return block_features


def ptx_to_pyg_graph(ptx_source: str) -> "Data":
    """
    Convert PTX source into a PyTorch Geometric Data object.

    The graph structure is:
      - Each **node** is a basic block from the PTX
      - Each **edge** connects consecutive basic blocks (control flow)
      - Node features are per-block instruction counts (normalized)
      - A global feature vector stores kernel-wide statistics

    Returns:
        torch_geometric.data.Data with:
          - x: (num_nodes, NODE_FEATURE_DIM) node features
          - edge_index: (2, num_edges) control flow edges
          - global_features: (1, 5) kernel-wide features
    """
    if not _HAS_PYG:
        raise ImportError(
            "PyTorch Geometric is required for ptx_to_pyg_graph. "
            "Install with: pip install torch-geometric"
        )

    block_features = _build_basic_blocks(ptx_source)
    n_blocks = len(block_features)

    # Node feature matrix
    x = np.zeros((n_blocks, NODE_FEATURE_DIM), dtype=np.float32)
    for i, feats in enumerate(block_features):
        for j, key in enumerate(_NODE_FEATURE_KEYS):
            x[i, j] = feats.get(key, 0.0)

    # Normalize: divide by max per column (avoid div-by-zero)
    col_max = x.max(axis=0, keepdims=True)
    col_max = np.where(col_max == 0, 1.0, col_max)
    x = x / col_max

    # Edge index: sequential control flow (block i → block i+1)
    # Plus backward edges for loops (if bar.sync detected)
    edges_src: List[int] = []
    edges_dst: List[int] = []

    for i in range(n_blocks - 1):
        # Forward edge
        edges_src.append(i)
        edges_dst.append(i + 1)
        # Backward edge (bidirectional for message passing)
        edges_src.append(i + 1)
        edges_dst.append(i)

    # Add loop-back edges for blocks ending with bar.sync
    # (these typically loop back to the start of the tile loop)
    for i, feats in enumerate(block_features):
        if feats.get("n_sync", 0) > 0 and i > 0:
            edges_src.append(i)
            edges_dst.append(0)  # Loop back to first block
            edges_src.append(0)
            edges_dst.append(i)

    if not edges_src:
        # Self-loop for single-node graph
        edges_src = [0]
        edges_dst = [0]

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    # Global features: kernel-wide statistics
    total_features = ptx_to_graph_features(ptx_source)
    global_vec = np.array([
        total_features.get("n_registers", 0) / 128.0,  # normalize by max regs
        total_features.get("shared_mem_bytes", 0) / (48 * 1024),  # normalize by max smem
        total_features.get("arithmetic_intensity_proxy", 0),
        total_features.get("n_sync", 0) / 10.0,
        total_features.get("n_instructions", 0) / 500.0,
    ], dtype=np.float32).clip(0.0, 1.0)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        global_features=torch.tensor(global_vec, dtype=torch.float32).unsqueeze(0),
        num_nodes=n_blocks,
    )

    return data


# ── Kernel-specific extraction helpers ───────────────────────────────


def extract_kernel_graph(kernel_name: str) -> "Data":
    """
    Extract a PyG graph from one of the project's benchmark kernels.

    Args:
        kernel_name: One of 'gemm', 'reduction', 'softmax'

    Returns:
        torch_geometric.data.Data
    """
    from numba import types

    if kernel_name == "gemm":
        from kernels.gemm import gemm_kernel_16
        kernel_fn = gemm_kernel_16
        arg_types = (
            types.Array(types.float32, 2, "C"),
            types.Array(types.float32, 2, "C"),
            types.Array(types.float32, 2, "C"),
            types.int32,
        )
    elif kernel_name == "reduction":
        from kernels.reduction import reduction_kernel
        kernel_fn = reduction_kernel
        arg_types = (
            types.Array(types.float32, 1, "C"),
            types.Array(types.float32, 1, "C"),
            types.int32,
        )
    elif kernel_name == "softmax":
        from kernels.softmax import softmax_kernel
        kernel_fn = softmax_kernel
        arg_types = (
            types.Array(types.float32, 2, "C"),
            types.Array(types.float32, 2, "C"),
            types.int32,
            types.int32,
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}. Use 'gemm', 'reduction', or 'softmax'.")

    ptx = extract_ptx(kernel_fn, arg_types)
    return ptx_to_pyg_graph(ptx)
