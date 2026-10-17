"""Stability-aware routing with MGAC costs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import heapq
import numpy as np

from .mgac import MGACParams, compute_mgac_weights, mgac_edge_cost
from .mfd_regime import MFDRegime, RegimeWeights, get_regime_weights


class RoutingMethod(str, Enum):
    TIME_ONLY = "time_only"
    OTS_INSTANT = "ots_instant"
    OTS_CVAR = "ots_cvar_fixed"
    OTS_ROBUST = "ots_robust"
    SPARC_OTS = "sparc_ots"
    SPARC_NO_GKF = "sparc_no_gkf"
    SPARC_NO_MGAC = "sparc_no_mgac"
    ORACLE = "oracle"


@dataclass
class RoutingConfig:
    cvar_alpha: float = 0.9
    lambda_ots: float = 1.0
    robust_gamma: float = 0.5
    min_speed_kmh: float = 5.0
    use_mgac: bool = True
    mgac_params: Optional[MGACParams] = None


@dataclass
class RoutingTopology:
    """Sparse adjacency for E3 routing (neighbors + edge distances in metres)."""

    neighbors: List[List[Tuple[int, float]]]
    n_nodes: int
    source: str

    @classmethod
    def from_direct_distances(cls, dist_direct: np.ndarray, source: str = "csv_direct") -> "RoutingTopology":
        n = int(dist_direct.shape[0])
        neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                d = float(dist_direct[u, v])
                if np.isfinite(d) and d > 0:
                    neighbors[u].append((v, d))
        return cls(neighbors=neighbors, n_nodes=n, source=source)

    @classmethod
    def from_adjacency(cls, adj: np.ndarray, fallback_m: float = 500.0, source: str = "adj_mx") -> "RoutingTopology":
        n = int(adj.shape[0])
        neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                w = float(adj[u, v])
                if w > 0:
                    neighbors[u].append((v, fallback_m / max(w, 1e-3)))
        return cls(neighbors=neighbors, n_nodes=n, source=source)

    @classmethod
    def from_dense_distances(cls, dist_matrix: np.ndarray, source: str = "legacy_dense") -> "RoutingTopology":
        n = int(dist_matrix.shape[0])
        neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                d = float(dist_matrix[u, v])
                if np.isfinite(d):
                    neighbors[u].append((v, d))
        return cls(neighbors=neighbors, n_nodes=n, source=source)

    def mean_degree(self) -> float:
        if self.n_nodes == 0:
            return 0.0
        return float(np.mean([len(nbrs) for nbrs in self.neighbors]))


def travel_time_hours(dist_m: float, speed_kmh: float, min_speed: float = 5.0) -> float:
    return (dist_m / 1000.0) / max(float(speed_kmh), min_speed)


def cvar_upper(values: np.ndarray, alpha: float = 0.9) -> float:
    if len(values) == 0:
        return 0.0
    q = np.quantile(values, alpha)
    tail = values[values >= q]
    return float(np.mean(tail)) if len(tail) else float(q)


def edge_cost(
    dist_m: float,
    speed_kmh: float,
    mu_ots: float,
    sigma_ots: float,
    ots_samples: Optional[np.ndarray],
    method: RoutingMethod,
    regime: MFDRegime = MFDRegime.FREE_FLOW,
    cfg: Optional[RoutingConfig] = None,
    mfd_state: Optional[np.ndarray] = None,
) -> float:
    cfg = cfg or RoutingConfig()
    tau = travel_time_hours(dist_m, speed_kmh, cfg.min_speed_kmh)
    cvar = cvar_upper(ots_samples, cfg.cvar_alpha) if ots_samples is not None else mu_ots * 1.1

    if method == RoutingMethod.TIME_ONLY:
        return tau
    if method == RoutingMethod.OTS_INSTANT:
        return tau + cfg.lambda_ots * mu_ots
    if method == RoutingMethod.OTS_CVAR:
        return tau + cfg.lambda_ots * cvar
    if method == RoutingMethod.OTS_ROBUST:
        return tau + cfg.lambda_ots * (mu_ots + cfg.robust_gamma * sigma_ots)

    use_mgac = cfg.use_mgac and method in (RoutingMethod.SPARC_OTS, RoutingMethod.ORACLE)
    if use_mgac and cfg.mgac_params is not None and mfd_state is not None:
        w = compute_mgac_weights(cfg.mgac_params, mfd_state)
        return mgac_edge_cost(tau, mu_ots, sigma_ots, cvar, w)

    rw: RegimeWeights = get_regime_weights(regime)
    return tau + rw.omega * mu_ots + rw.kappa * cvar + 0.2 * sigma_ots


def build_routing_graph(dist_matrix, speed, mu_ots, sigma_ots=None):
    n = dist_matrix.shape[0]
    if sigma_ots is None:
        sigma_ots = np.zeros(n, dtype=np.float32)
    return {
        "dist": dist_matrix,
        "speed": speed.astype(np.float32),
        "mu_ots": mu_ots.astype(np.float32),
        "sigma_ots": sigma_ots.astype(np.float32),
    }


def _metrics_from_predecessors(
    graph: Dict[str, np.ndarray],
    topology: RoutingTopology,
    prev: np.ndarray,
    origin: int,
    destination: int,
    total_cost: float,
    cfg: RoutingConfig,
    return_path: bool,
) -> Tuple[List[int], float, Dict[str, float]]:
    dist_lookup: Dict[Tuple[int, int], float] = {}
    for u, nbrs in enumerate(topology.neighbors):
        for v, d in nbrs:
            dist_lookup[(u, v)] = d

    speed = graph["speed"]
    mu_ots = graph["mu_ots"]

    if not np.isfinite(total_cost) or prev[destination] < 0:
        return [], np.inf, {"travel_time": np.inf, "rii": np.inf, "cvar": np.inf}

    path: List[int] = []
    cur = destination
    while cur >= 0:
        path.append(int(cur))
        if cur == origin:
            break
        cur = int(prev[cur])
    if not path or path[-1] != origin:
        return [], np.inf, {"travel_time": np.inf, "rii": np.inf, "cvar": np.inf}
    path.reverse()

    if not return_path:
        tt, rii, risks = 0.0, 0.0, []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            d_m = dist_lookup.get((u, v), float(graph["dist"][u, v]))
            tt += travel_time_hours(d_m, float(speed[v]), cfg.min_speed_kmh)
            rii += float(mu_ots[v])
            risks.append(float(mu_ots[v]))
        return [], total_cost, {
            "travel_time": tt,
            "rii": rii,
            "cvar": cvar_upper(np.array(risks), cfg.cvar_alpha) if risks else 0.0,
            "path_length": 0,
        }

    tt, rii, risks = 0.0, 0.0, []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        d_m = dist_lookup.get((u, v), float(graph["dist"][u, v]))
        tt += travel_time_hours(d_m, float(speed[v]), cfg.min_speed_kmh)
        rii += float(mu_ots[v])
        risks.append(float(mu_ots[v]))
    return path, total_cost, {
        "travel_time": tt,
        "rii": rii,
        "cvar": cvar_upper(np.array(risks), cfg.cvar_alpha) if risks else 0.0,
        "path_length": len(path),
    }


def build_cost_csr(
    graph: Dict[str, np.ndarray],
    topology: RoutingTopology,
    method: RoutingMethod,
    regime: MFDRegime,
    cfg: RoutingConfig,
    ots_history: Optional[np.ndarray],
    mfd_state: Optional[np.ndarray],
):
    from scipy.sparse import csr_matrix

    rows, cols, data = [], [], []
    speed = graph["speed"]
    mu_ots = graph["mu_ots"]
    sigma_ots = graph["sigma_ots"]
    for u, nbrs in enumerate(topology.neighbors):
        for v, dist_m in nbrs:
            samples = ots_history[:, v] if ots_history is not None else None
            c = edge_cost(
                dist_m,
                float(speed[v]),
                float(mu_ots[v]),
                float(sigma_ots[v]),
                samples,
                method,
                regime,
                cfg,
                mfd_state,
            )
            rows.append(u)
            cols.append(v)
            data.append(c)
    n = topology.n_nodes
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def dijkstra_legacy(
    graph: Dict[str, np.ndarray],
    origin: int,
    destination: int,
    method: RoutingMethod,
    regime: MFDRegime = MFDRegime.FREE_FLOW,
    cfg: Optional[RoutingConfig] = None,
    ots_history: Optional[np.ndarray] = None,
    mfd_state: Optional[np.ndarray] = None,
    return_path: bool = True,
) -> Tuple[List[int], float, Dict[str, float]]:
    cfg = cfg or RoutingConfig()
    topology = RoutingTopology.from_dense_distances(graph["dist"], source="legacy_dense")
    return dijkstra_sparse(
        graph, topology, origin, destination, method, regime, cfg, ots_history, mfd_state, return_path
    )


def dijkstra_sparse(
    graph: Dict[str, np.ndarray],
    topology: RoutingTopology,
    origin: int,
    destination: int,
    method: RoutingMethod,
    regime: MFDRegime = MFDRegime.FREE_FLOW,
    cfg: Optional[RoutingConfig] = None,
    ots_history: Optional[np.ndarray] = None,
    mfd_state: Optional[np.ndarray] = None,
    return_path: bool = True,
) -> Tuple[List[int], float, Dict[str, float]]:
    cfg = cfg or RoutingConfig()
    speed = graph["speed"]
    mu_ots = graph["mu_ots"]
    sigma_ots = graph["sigma_ots"]

    pq = [(0.0, origin)]
    best = {origin: 0.0}
    prev: Dict[int, Optional[int]] = {origin: None}

    while pq:
        cost, u = heapq.heappop(pq)
        if cost > best.get(u, np.inf):
            continue
        if u == destination:
            break
        for v, dist_m in topology.neighbors[u]:
            samples = ots_history[:, v] if ots_history is not None else None
            c = edge_cost(
                dist_m,
                float(speed[v]),
                float(mu_ots[v]),
                float(sigma_ots[v]),
                samples,
                method,
                regime,
                cfg,
                mfd_state,
            )
            nc = cost + c
            if nc < best.get(v, np.inf):
                best[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc, v))

    if destination not in best:
        return [], np.inf, {"travel_time": np.inf, "rii": np.inf, "cvar": np.inf}

    prev_arr = np.full(topology.n_nodes, -1, dtype=np.int32)
    for node, p in prev.items():
        if p is not None:
            prev_arr[node] = p
    return _metrics_from_predecessors(
        graph, topology, prev_arr, origin, destination, best[destination], cfg, return_path
    )


def dijkstra_scipy(
    graph: Dict[str, np.ndarray],
    topology: RoutingTopology,
    origin: int,
    destination: int,
    method: RoutingMethod,
    regime: MFDRegime = MFDRegime.FREE_FLOW,
    cfg: Optional[RoutingConfig] = None,
    ots_history: Optional[np.ndarray] = None,
    mfd_state: Optional[np.ndarray] = None,
    return_path: bool = True,
) -> Tuple[List[int], float, Dict[str, float]]:
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra

    cfg = cfg or RoutingConfig()
    w = build_cost_csr(graph, topology, method, regime, cfg, ots_history, mfd_state)
    dist_matrix, predecessors = sp_dijkstra(
        w, directed=True, indices=origin, return_predecessors=True, min_only=False
    )
    if dist_matrix.ndim == 1:
        total = float(dist_matrix[destination])
        prev_row = predecessors
    else:
        total = float(dist_matrix[0, destination])
        prev_row = predecessors[0]
    return _metrics_from_predecessors(
        graph, topology, prev_row, origin, destination, total, cfg, return_path
    )


def dijkstra(
    graph: Dict[str, np.ndarray],
    origin: int,
    destination: int,
    method: RoutingMethod,
    regime: MFDRegime = MFDRegime.FREE_FLOW,
    cfg: Optional[RoutingConfig] = None,
    ots_history: Optional[np.ndarray] = None,
    mfd_state: Optional[np.ndarray] = None,
    return_path: bool = True,
    topology: Optional[RoutingTopology] = None,
    backend: str = "scipy",
) -> Tuple[List[int], float, Dict[str, float]]:
    """Route with optional sparse topology and scipy/python backend."""
    if backend == "legacy" or topology is None:
        return dijkstra_legacy(
            graph, origin, destination, method, regime, cfg, ots_history, mfd_state, return_path
        )
    if backend == "scipy":
        try:
            return dijkstra_scipy(
                graph, topology, origin, destination, method, regime, cfg, ots_history, mfd_state, return_path
            )
        except ImportError:
            pass
    return dijkstra_sparse(
        graph, topology, origin, destination, method, regime, cfg, ots_history, mfd_state, return_path
    )


def sample_od_pairs(dist_matrix, n_pairs, min_dist_m=5000.0, seed=42):
    rng = np.random.default_rng(seed)
    n = dist_matrix.shape[0]
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 50:
        o, d = rng.integers(0, n, size=2)
        attempts += 1
        if o != d and float(dist_matrix[o, d]) >= min_dist_m:
            pairs.append((int(o), int(d)))
    while len(pairs) < n_pairs:
        o, d = rng.integers(0, n, size=2)
        if o != d:
            pairs.append((int(o), int(d)))
    return pairs[:n_pairs]
