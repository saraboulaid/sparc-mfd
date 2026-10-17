"""Decision Support Layer: Pareto routes and Decision Quality Index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class RouteCandidate:
    path: List[int]
    travel_time: float
    rii: float
    label: str


def route_instability_index(risk_along_path: List[float]) -> float:
    return float(np.sum(risk_along_path)) if risk_along_path else 0.0


def pareto_filter(candidates: List[RouteCandidate]) -> List[RouteCandidate]:
    """Keep non-dominated routes (minimize time and RII)."""
    pareto = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if other is c:
                continue
            if other.travel_time <= c.travel_time and other.rii <= c.rii:
                if other.travel_time < c.travel_time or other.rii < c.rii:
                    dominated = True
                    break
        if not dominated:
            pareto.append(c)
    return sorted(pareto, key=lambda x: x.travel_time)


def recommend_route(
    candidates: List[RouteCandidate],
    time_budget_pct: float = 0.10,
    baseline_time: float = 1.0,
) -> Tuple[RouteCandidate, List[RouteCandidate]]:
    """Pick best RII among routes within time budget vs baseline."""
    pareto = pareto_filter(candidates)
    if not pareto:
        return candidates[0], []
    max_time = baseline_time * (1.0 + time_budget_pct)
    feasible = [c for c in pareto if c.travel_time <= max_time] or pareto
    best = min(feasible, key=lambda c: c.rii)
    return best, pareto


def decision_quality_index(rii_chosen: float, rii_baseline: float, rii_oracle: float) -> float:
    denom = rii_baseline - rii_oracle
    if abs(denom) < 1e-8:
        return 1.0
    return float(np.clip(1.0 - (rii_chosen - rii_oracle) / denom, 0.0, 1.0))


def summarize_decision_support(
    dqi_scores: List[float],
    pareto_sizes: List[int],
) -> Dict[str, float]:
    return {
        "dqi_mean": float(np.mean(dqi_scores)) if dqi_scores else 0.0,
        "dqi_median": float(np.median(dqi_scores)) if dqi_scores else 0.0,
        "pareto_size_mean": float(np.mean(pareto_sizes)) if pareto_sizes else 0.0,
    }
