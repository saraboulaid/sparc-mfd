import pickle
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def load_adjacency(adj_path: Path, threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    with open(adj_path, "rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*align should be passed as Python or NumPy boolean.*",
            )
            _, _, adj = pickle.load(f, encoding="latin1")

    adj = np.array(adj, dtype=np.float32)
    adj[adj < threshold] = 0.0
    np.fill_diagonal(adj, 1.0)

    src, dst = np.nonzero(adj > 0)
    w = adj[src, dst]
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)
    return edge_index, edge_weight, adj


def build_distance_matrix(adj: np.ndarray, fallback_m: float = 500.0) -> np.ndarray:
    """Build pairwise road distances (m) from adjacency weights."""
    n = adj.shape[0]
    dist = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)

    src, dst = np.nonzero(adj > 0)
    for s, d in zip(src, dst):
        if s == d:
            continue
        w = float(adj[s, d])
        dist[s, d] = fallback_m / max(w, 1e-3)

    for k in range(n):
        dist = np.minimum(dist, dist[:, k][:, None] + dist[k, :][None, :])
    dist[~np.isfinite(dist)] = fallback_m * n
    return dist.astype(np.float32)
