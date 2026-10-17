"""Lightweight Frank-Wolfe traffic assignment on sensor graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .routing import RoutingMethod, dijkstra, sample_od_pairs


@dataclass
class AssignmentConfig:
    max_iter: int = 20
    convergence_tol: float = 1e-4
    step_size: float = 0.3


def frank_wolfe_assignment(
    graph: Dict[str, np.ndarray],
    od_pairs: List[Tuple[int, int]],
    od_demands: np.ndarray,
    cost_method: RoutingMethod,
    regime: int = 0,
    cfg: AssignmentConfig | None = None,
) -> Dict[str, float]:
    """
    Simplified link-flow assignment via iterative shortest-path loading.
    """
    cfg = cfg or AssignmentConfig()
    n = graph["speed"].shape[0]
    link_flow = np.zeros((n, n), dtype=np.float64)
    total_demand = float(np.sum(od_demands))

    from .mfd_regime import MFDRegime

    mfd_regime = MFDRegime(regime)

    for _ in range(cfg.max_iter):
        aux_flow = np.zeros_like(link_flow)
        for (o, d), demand in zip(od_pairs, od_demands):
            path, _, _ = dijkstra(graph, o, d, cost_method, mfd_regime)
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                aux_flow[u, v] += demand

        diff = np.abs(aux_flow - link_flow).sum()
        link_flow = (1.0 - cfg.step_size) * link_flow + cfg.step_size * aux_flow
        if diff / max(total_demand, 1.0) < cfg.convergence_tol:
            break

    risk_exposure = 0.0
    travel_cost = 0.0
    ots = graph["mu_ots"]
    dist = graph["dist"]
    speed = graph["speed"]
    for u in range(n):
        for v in range(n):
            if link_flow[u, v] <= 0:
                continue
            f = link_flow[u, v]
            travel_cost += f * (float(dist[u, v]) / 1000.0 / max(float(speed[v]), 5.0))
            risk_exposure += f * float(ots[v])

    return {
        "total_flow": float(link_flow.sum()),
        "travel_cost": travel_cost,
        "risk_exposure": risk_exposure,
        "n_iter_effective": cfg.max_iter,
    }
