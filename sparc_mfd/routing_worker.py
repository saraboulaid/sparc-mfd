"""Lightweight OD routing worker for multiprocessing (no torch import)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .decision_support import RouteCandidate, decision_quality_index, recommend_route
from .mfd_regime import MFDRegime
from .routing import RoutingMethod, dijkstra, precompute_node_cvar

_CTX: Dict = {}


def init_worker(ctx: Dict) -> None:
    global _CTX
    _CTX = ctx
    # CVaR of the 12-step history depends only on the node, not on the arc, the
    # origin, or the method. Compute it once per worker process and reuse it for
    # every OD pair and every Dijkstra call in this timestep.
    if "node_cvar" not in _CTX:
        rcfg = _CTX.get("rcfg")
        hist = _CTX.get("hist_slice")
        alpha = getattr(rcfg, "cvar_alpha", 0.9) if rcfg is not None else 0.9
        _CTX["node_cvar"] = precompute_node_cvar(hist, alpha)


def route_od_pair(od: Tuple[int, int]) -> Dict:
    o, d = od
    graphs = _CTX["graphs"]
    methods = [RoutingMethod(m) for m in _CTX["method_names"]]
    regime = MFDRegime(_CTX["regime"])
    rcfg = _CTX["rcfg"]
    hist_slice = _CTX.get("hist_slice")
    mfd_st = _CTX.get("mfd_st")
    topology = _CTX["topology"]
    backend = _CTX["backend"]
    node_cvar = _CTX.get("node_cvar")
    cost_csrs = _CTX.get("cost_csrs") or {}

    appends: List[Tuple[str, float, float]] = []
    dqi_vals: List[float] = []
    candidates: List[RouteCandidate] = []

    def _run(g, method, extra=None):
        return dijkstra(
            g, o, d, method, regime, rcfg, hist_slice, mfd_st,
            return_path=False, topology=topology, backend=backend,
            node_cvar=node_cvar, cost_csr=extra,
        )

    for method in methods:
        if method == RoutingMethod.ORACLE:
            g = graphs["oracle"]
        elif method == RoutingMethod.OTS_INSTANT:
            g = graphs["instant"]
        else:
            g = graphs["sparse"]
        _, _, met = _run(g, method, extra=cost_csrs.get(method.value))
        if met["rii"] < np.inf:
            appends.append((method.value, met["rii"], met["travel_time"]))
        if method == RoutingMethod.SPARC_OTS and met["rii"] < np.inf:
            candidates.append(RouteCandidate([], met["travel_time"], met["rii"], "sparc"))

    _, _, b1 = _run(graphs["sparse"], RoutingMethod.TIME_ONLY, extra=cost_csrs.get("time_only"))
    _, _, bo = _run(graphs["oracle"], RoutingMethod.ORACLE, extra=cost_csrs.get("oracle"))
    if candidates and b1["rii"] < np.inf and bo["rii"] < np.inf:
        rec, _ = recommend_route(candidates, baseline_time=b1["travel_time"])
        dqi_vals.append(decision_quality_index(rec.rii, b1["rii"], bo["rii"]))

    return {"appends": appends, "dqi": dqi_vals}
