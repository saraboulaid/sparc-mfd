"""GKF-Fusion: learned Graph-Kriging fusion per sensor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class GKFConfig:
    hidden_dim: int = 32


class GKFFusion(nn.Module):
    """alpha_i = sigma(w @ [h_i, krig_var_i, observed_i])"""

    def __init__(self, embed_dim: int = 64, cfg: Optional[GKFConfig] = None):
        super().__init__()
        cfg = cfg or GKFConfig()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(
        self,
        mu_gnn: torch.Tensor,
        mu_krig: torch.Tensor,
        krig_var: torch.Tensor,
        embedding: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = observed_mask.float().unsqueeze(-1)
        x = torch.cat([embedding, krig_var.unsqueeze(-1), obs], dim=-1)
        alpha = torch.sigmoid(self.net(x)).squeeze(-1)
        mu = alpha * mu_gnn + (1.0 - alpha) * mu_krig
        var = alpha * krig_var + (1.0 - alpha) * krig_var
        return mu, var


def fuse_numpy(
    mu_gnn: np.ndarray,
    mu_krig: np.ndarray,
    krig_var: np.ndarray,
    embedding: np.ndarray,
    observed_mask: np.ndarray,
    model: GKFFusion,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        mu, var = model(
            torch.from_numpy(mu_gnn).float(),
            torch.from_numpy(mu_krig).float(),
            torch.from_numpy(krig_var).float(),
            torch.from_numpy(embedding).float(),
            torch.from_numpy(observed_mask),
        )
    return mu.numpy(), var.numpy()
