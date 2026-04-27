"""
GNN IR Encoder — Graph Neural Network for kernel structure encoding.

This module takes a PyTorch Geometric graph (produced by compiler/ir_extractor.py)
and encodes it into a fixed-size embedding vector.  This embedding can be
concatenated with the PMU-counter observation to give the RL agent a richer
state representation.

Architecture:
    Input:  PyG Data (basic-block graph from PTX)
    GCN:    3 × GCNConv layers  →  per-node embeddings
    Pool:   global_mean_pool    →  graph-level embedding
    MLP:    Linear(hidden→64)   →  final embedding (64-dim)
    Concat: global_features (5) →  fused with graph embedding
    Output: (batch, embed_dim)  →  fixed-size kernel representation

The output vector can be appended to the RL observation vector to
upgrade the agent from PMU-only to PMU+structure.

Usage:
    from models.gnn_encoder import GNNEncoder
    from compiler.ir_extractor import extract_kernel_graph

    encoder = GNNEncoder()
    graph = extract_kernel_graph("gemm")
    embedding = encoder(graph)  # (1, embed_dim=69)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, global_mean_pool

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

from compiler.ir_extractor import NODE_FEATURE_DIM


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for GPU kernel IR structure.

    Converts a basic-block control-flow graph into a fixed-size embedding.

    Architecture:
      GCNConv(in, hidden) → ReLU → Dropout
      GCNConv(hidden, hidden) → ReLU → Dropout
      GCNConv(hidden, hidden) → ReLU
      global_mean_pool → (batch, hidden)
      Linear(hidden, graph_embed_dim) → ReLU
      concat(graph_embed, global_features) → (batch, embed_dim)

    Args:
        node_feat_dim: Input node feature dimension (default: NODE_FEATURE_DIM=10)
        hidden_dim: Hidden dimension for GCN layers (default: 64)
        graph_embed_dim: Output dim of graph-level embedding before global concat (default: 64)
        global_feat_dim: Dimension of kernel-wide global features (default: 5)
        dropout: Dropout rate for GCN layers (default: 0.1)
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = 64,
        graph_embed_dim: int = 64,
        global_feat_dim: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not _HAS_PYG:
            raise ImportError(
                "PyTorch Geometric is required for GNNEncoder. "
                "Install with: pip install torch-geometric"
            )

        self.hidden_dim = hidden_dim
        self.graph_embed_dim = graph_embed_dim
        self.global_feat_dim = global_feat_dim

        # GCN layers
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Graph-level projection
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, graph_embed_dim),
            nn.ReLU(),
        )

        # Fusion layer: merge graph embedding with global features
        self.fusion = nn.Sequential(
            nn.Linear(graph_embed_dim + global_feat_dim, graph_embed_dim + global_feat_dim),
            nn.ReLU(),
        )

        # Final output dimension
        self.embed_dim = graph_embed_dim + global_feat_dim

    @property
    def output_dim(self) -> int:
        """The dimension of the output embedding vector."""
        return self.embed_dim

    def forward(self, data: "Data") -> torch.Tensor:
        """
        Encode a kernel graph into a fixed-size embedding.

        Args:
            data: PyTorch Geometric Data object with:
                  - x: (num_nodes, node_feat_dim) node features
                  - edge_index: (2, num_edges) edge indices
                  - global_features: (1, global_feat_dim) kernel-wide features
                  - batch: (num_nodes,) batch assignment (for batched graphs)

        Returns:
            embedding: (batch_size, embed_dim) fixed-size kernel representation
        """
        x, edge_index = data.x, data.edge_index

        # Batch vector (for single graph, all nodes belong to batch 0)
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GCN message passing
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)

        # Graph-level pooling: mean over all nodes
        graph_embed = global_mean_pool(x, batch)  # (batch_size, hidden_dim)

        # Project to embedding dimension
        graph_embed = self.graph_proj(graph_embed)  # (batch_size, graph_embed_dim)

        # Get global features
        if hasattr(data, "global_features"):
            global_feats = data.global_features
            if global_feats.dim() == 1:
                global_feats = global_feats.unsqueeze(0)
            # Handle batched global features
            if global_feats.size(0) != graph_embed.size(0):
                global_feats = global_feats.expand(graph_embed.size(0), -1)
        else:
            global_feats = torch.zeros(
                graph_embed.size(0), self.global_feat_dim,
                device=graph_embed.device, dtype=graph_embed.dtype,
            )

        # Concatenate graph embedding + global features
        fused = torch.cat([graph_embed, global_feats], dim=-1)  # (batch, embed_dim)
        fused = self.fusion(fused)

        return fused

    def encode_kernel(self, kernel_name: str) -> torch.Tensor:
        """
        Convenience method: extract PTX, build graph, and encode in one step.

        Args:
            kernel_name: One of 'gemm', 'reduction', 'softmax'

        Returns:
            embedding: (1, embed_dim) kernel structure embedding
        """
        from compiler.ir_extractor import extract_kernel_graph

        graph = extract_kernel_graph(kernel_name)
        self.eval()
        with torch.no_grad():
            return self.forward(graph)


class KernelStructureCache:
    """
    Cache kernel structure embeddings to avoid re-extracting PTX every step.

    During RL training, the kernel structure doesn't change between steps
    (only the block_size/reg_cap change). This cache stores the GNN embedding
    for each kernel and returns it instantly.

    Usage:
        cache = KernelStructureCache(encoder)
        emb = cache.get("gemm")  # First call: extracts PTX + runs GNN
        emb = cache.get("gemm")  # Cached: instant
    """

    def __init__(self, encoder: GNNEncoder):
        self.encoder = encoder
        self._cache: dict[str, torch.Tensor] = {}

    def get(self, kernel_name: str) -> torch.Tensor:
        """Get the cached embedding for a kernel, computing if necessary."""
        if kernel_name not in self._cache:
            self._cache[kernel_name] = self.encoder.encode_kernel(kernel_name)
        return self._cache[kernel_name]

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def precompute(self, kernel_names: Optional[list[str]] = None) -> None:
        """Precompute and cache embeddings for all kernels."""
        if kernel_names is None:
            kernel_names = ["gemm", "reduction", "softmax"]
        for name in kernel_names:
            self.get(name)
