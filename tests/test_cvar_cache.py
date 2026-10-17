"""Cached per-node CVaR is bit-identical to the per-arc recomputation it replaces."""

from __future__ import annotations

import numpy as np

from sparc_mfd.mfd_regime import MFDRegime
from sparc_mfd.mgac import MGACParams
from sparc_mfd.routing import (
    RoutingConfig,
    RoutingMethod,
    RoutingTopology,
    build_cost_csr,
    cvar_upper,
    edge_cost,
    precompute_node_cvar,
)


def test_precompute_matches_per_column_cvar():
    rng = np.random.default_rng(0)
    hist = rng.uniform(0.2, 4.0, size=(12, 40))
    cached = precompute_node_cvar(hist, alpha=0.9)
    assert cached is not None
    assert cached.shape == (40,)
    for v in range(40):
        assert cached[v] == cvar_upper(hist[:, v], 0.9)


def test_precompute_none_on_empty_history():
    assert precompute_node_cvar(None) is None
    assert precompute_node_cvar(np.zeros((0, 5))) is None


def test_edge_cost_accepts_precomputed_cvar():
    hist = np.linspace(0.1, 2.0, 12)
    expected = cvar_upper(hist, 0.9)
    from_samples = edge_cost(1500.0, 40.0, 1.0, 0.2, hist, RoutingMethod.OTS_CVAR)
    from_cache = edge_cost(
        1500.0, 40.0, 1.0, 0.2, None, RoutingMethod.OTS_CVAR, cvar=expected
    )
    assert from_samples == from_cache


def test_cost_csr_matches_per_arc_samples():
    n = 8
    adj = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(adj, 0.0)
    dist = np.full((n, n), 2000.0)
    np.fill_diagonal(dist, 0.0)
    topo = RoutingTopology.from_adjacency_distances(adj, dist)
    rng = np.random.default_rng(1)
    graph = {
        "speed": rng.uniform(20, 60, n).astype(np.float32),
        "mu_ots": rng.uniform(0.5, 2.0, n).astype(np.float32),
        "sigma_ots": rng.uniform(0.1, 0.5, n).astype(np.float32),
        "dist": dist.astype(np.float32),
    }
    hist = rng.uniform(0.2, 3.0, (12, n)).astype(np.float32)
    cfg = RoutingConfig(
        use_mgac=True,
        mgac_params=MGACParams(
            u_omega=np.zeros(3, dtype=np.float32),
            u_kappa=np.zeros(3, dtype=np.float32),
            u_rho=np.zeros(3, dtype=np.float32),
        ),
    )
    mfd = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    cached = build_cost_csr(
        graph, topo, RoutingMethod.SPARC_OTS, MFDRegime.FREE_FLOW, cfg, hist, mfd,
        node_cvar=precompute_node_cvar(hist, cfg.cvar_alpha),
    ).toarray()
    expected = np.zeros((n, n), dtype=np.float64)
    for u, nbrs in enumerate(topo.neighbors):
        for v, dist_m in nbrs:
            expected[u, v] = edge_cost(
                dist_m,
                float(graph["speed"][v]),
                float(graph["mu_ots"][v]),
                float(graph["sigma_ots"][v]),
                hist[:, v],
                RoutingMethod.SPARC_OTS,
                MFDRegime.FREE_FLOW,
                cfg,
                mfd,
            )
    np.testing.assert_allclose(cached, expected, rtol=0, atol=0)


def test_distance_lookup_is_cached():
    n = 6
    adj = np.ones((n, n))
    np.fill_diagonal(adj, 0)
    dist = np.full((n, n), 1000.0)
    np.fill_diagonal(dist, 0)
    topo = RoutingTopology.from_adjacency_distances(adj, dist)
    first = topo.distance_lookup()
    second = topo.distance_lookup()
    assert first is second
    assert first[(0, 1)] == 1000.0
