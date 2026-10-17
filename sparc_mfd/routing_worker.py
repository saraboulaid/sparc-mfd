"""Lightweight OD routing worker for multiprocessing (no torch import)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .decision_support import RouteCandidate, decision_quality_index, recommend_route
from .mfd_regime import MFDRegime
from .routing import RoutingMethod, dijkstra

_CTX: Dict = {}


def init_worker(ctx: Dict) -> None:
    global _CTX
    _CTX = ctx


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

    appends: List[Tuple[str, float, float]] = []
    dqi_vals: List[float] = []
    candidates: List[RouteCandidate] = []

    for method in methods:
        if method == RoutingMethod.ORACLE:
            g = graphs["oracle"]
        elif method == RoutingMethod.OTS_INSTANT:
            g = graphs["instant"]
        else:
            g = graphs["sparse"]
        _, _, met = dijkstra(
            g, o, d, method, regime, rcfg, hist_slice, mfd_st,
            return_path=False, topology=topology, backend=backend,
        )
        if met["rii"] < np.inf:
            appends.append((method.value, met["rii"], met["travel_time"]))
        if method == RoutingMethod.SPARC_OTS and met["rii"] < np.inf:
            candidates.append(RouteCandidate([], met["travel_time"], met["rii"], "sparc"))

    _, _, b1 = dijkstra(
        graphs["sparse"], o, d, RoutingMethod.TIME_ONLY, regime, rcfg, hist_slice, mfd_st,
        return_path=False, topology=topology, backend=backend,
    )
    _, _, bo = dijkstra(
        graphs["oracle"], o, d, RoutingMethod.ORACLE, regime, rcfg, hist_slice, mfd_st,
        return_path=False, topology=topology, backend=backend,
    )
    if candidates and b1["rii"] < np.inf and bo["rii"] < np.inf:
        rec, _ = recommend_route(candidates, baseline_time=b1["travel_time"])
        dqi_vals.append(decision_quality_index(rec.rii, b1["rii"], bo["rii"]))

    return {"appends": appends, "dqi": dqi_vals}
