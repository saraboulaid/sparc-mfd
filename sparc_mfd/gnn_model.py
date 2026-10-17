import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        out, _ = self.gru(x)
        return out[:, -1, :].reshape(b, n, -1)


class SpatialGraphAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        b, n, h = z.shape
        src, dst = edge_index[0], edge_index[1]
        q = self.w_q(z).reshape(b, n, self.heads, self.head_dim)
        k = self.w_k(z).reshape(b, n, self.heads, self.head_dim)
        v = self.w_v(z).reshape(b, n, self.heads, self.head_dim)
        q_dst = q[:, dst]
        k_src = k[:, src]
        v_src = v[:, src]
        scores = (q_dst * k_src).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = scores + torch.log(edge_weight.clamp_min(1e-8))[None, :, None]
        alpha = torch.softmax(scores, dim=1)
        alpha = self.dropout(alpha)
        messages = alpha[..., None] * v_src
        out = torch.zeros((b, n, self.heads, self.head_dim), device=z.device, dtype=z.dtype)
        out.index_add_(1, dst, messages)
        return self.out_proj(out.reshape(b, n, h))


class RiskGNN(nn.Module):
    """GRU + GAT encoder predicting risk mean and log-variance."""

    def __init__(self, in_features: int = 5, hidden_dim: int = 64, heads: int = 4):
        super().__init__()
        self.temporal = TemporalEncoder(in_features, hidden_dim)
        self.spatial = SpatialGraphAttention(hidden_dim, heads=heads)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.spatial(self.temporal(x), edge_index, edge_weight)
        out = self.head(z)
        mu = F.softplus(out[..., 0])
        logvar = out[..., 1]
        return mu, logvar


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    var = torch.exp(logvar).clamp_min(1e-4)
    return 0.5 * (logvar + (y - mu) ** 2 / var).mean()
