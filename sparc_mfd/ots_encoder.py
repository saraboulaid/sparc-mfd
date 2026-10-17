"""OTS target computation and OTS-STEncoder (dual-branch GRU + GAT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OTSConfig:
    horizon: int = 1
    lookback: int = 3


def compute_ots_components(
    speed_h: np.ndarray,
    horizon: int = 1,
    lookback: int = 3,
    q_thresholds: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    t, n = speed_h.shape
    V = np.zeros((t, n), dtype=np.float32)
    D = np.zeros((t, n), dtype=np.float32)
    P = np.zeros((t, n), dtype=np.float32)
    if q_thresholds is None:
        q_thresholds = np.quantile(speed_h, 0.20, axis=0)

    for i in range(n):
        q = float(q_thresholds[i])
        for t0 in range(max(0, t - horizon)):
            fut = speed_h[t0 + 1 : t0 + horizon + 1, i]
            if len(fut) == 0:
                continue
            past = speed_h[max(0, t0 - lookback) : t0 + 1, i]
            V[t0, i] = float(np.std(fut)) if len(fut) > 1 else 0.0
            D[t0, i] = max(0.0, float(np.median(past)) - float(np.min(fut)))
            P[t0, i] = float(np.mean(fut < q))
    return {"V": V, "D": D, "P": P}


def compute_ots_scalar(components: Dict[str, np.ndarray], stats: Dict[str, float]) -> np.ndarray:
    parts = []
    for key in ("V", "D", "P"):
        mu, sig = stats[f"{key}_mean"], stats[f"{key}_std"]
        parts.append((components[key] - mu) / sig)
    return np.linalg.norm(np.stack(parts, axis=-1), axis=-1).astype(np.float32)


def fit_ots_stats(components: Dict[str, np.ndarray], train_end: int) -> Dict:
    stats = {}
    for key in ("V", "D", "P"):
        sub = components[key][:train_end]
        stats[f"{key}_mean"] = float(np.mean(sub))
        stats[f"{key}_std"] = float(np.std(sub) + 1e-6)
    stats["q_thresholds"] = np.quantile(components["V"][:train_end], 0.20, axis=0).astype(np.float32)
    return stats


class TemporalEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        out, _ = self.gru(x)
        return out[:, -1, :].reshape(b, n, -1)


class SpatialGraphAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z, edge_index, edge_weight):
        b, n, h = z.shape
        src, dst = edge_index[0], edge_index[1]
        q = self.w_q(z).reshape(b, n, self.heads, self.head_dim)
        k = self.w_k(z).reshape(b, n, self.heads, self.head_dim)
        v = self.w_v(z).reshape(b, n, self.heads, self.head_dim)
        scores = (q[:, dst] * k[:, src]).sum(-1) / math.sqrt(self.head_dim)
        scores = scores + torch.log(edge_weight.clamp_min(1e-8))[None, :, None]
        alpha = torch.softmax(scores, dim=1)
        messages = alpha[..., None] * v[:, src]
        out = torch.zeros(b, n, self.heads, self.head_dim, device=z.device, dtype=z.dtype)
        out.index_add_(1, dst, messages)
        return self.out_proj(out.reshape(b, n, h))


class OTSSTEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, heads: int = 4):
        super().__init__()
        self.trend_enc = TemporalEncoder(5, hidden_dim // 2)
        self.vol_enc = TemporalEncoder(2, hidden_dim // 2)
        self.fuse = nn.Linear(hidden_dim, hidden_dim)
        self.spatial = SpatialGraphAttention(hidden_dim, heads)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)
        self.aux_head = nn.Linear(hidden_dim, 3)

    def forward(self, x_trend, x_vol, edge_index, edge_weight):
        z = torch.cat([self.trend_enc(x_trend), self.vol_enc(x_vol)], dim=-1)
        z = F.relu(self.fuse(z))
        z = self.spatial(z, edge_index, edge_weight)
        mu = F.softplus(self.mu_head(z).squeeze(-1))
        logvar = self.logvar_head(z).squeeze(-1)
        aux = F.relu(self.aux_head(z))
        return mu, logvar, z, aux


def ots_loss(y_true, mu, logvar, aux, aux_target, beta=0.3):
    var = torch.exp(logvar).clamp_min(1e-4)
    nll = 0.5 * (logvar + (y_true - mu) ** 2 / var).mean()
    return nll + beta * F.mse_loss(aux, aux_target)
