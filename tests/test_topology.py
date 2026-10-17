"""Tests for routing topologies and reachability-aware OD sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sparc_mfd.data_loader import (
    ensure_dataset,
    load_direct_distances_csv,
    read_h5_sensor_ids,
)
from sparc_mfd.graph import load_adjacency
from sparc_mfd.routing import RoutingTopology, sample_od_pairs


@pytest.fixture(scope="module")
def metr_la():
    h5, adj_path, dist_path = ensure_dataset(Path("data/metr-la"), "metr-la")
    _, _, adj = load_adjacency(adj_path)
    sensor_ids = read_h5_sensor_ids(h5)
    dist_direct = load_direct_distances_csv(dist_path, adj.shape[0], sensor_ids=sensor_ids)
    return adj, dist_direct


def test_adj_real_is_sparse_and_metric(metr_la):
    adj, dist_direct = metr_la
    topo = RoutingTopology.from_adjacency_distances(adj, dist_direct)
    dense = RoutingTopology.from_dense_distances(
        np.where(np.isfinite(dist_direct), dist_direct, 12000.0)
    )
    # The published results used the dense graph; adj_real must be far sparser.
    assert topo.mean_degree() < 10
    assert dense.mean_degree() > 100

    lengths = np.array([d for nbrs in topo.neighbors for _, d in nbrs])
    # Measured road distances between adjacent freeway detectors, not a proxy.
    assert lengths.min() > 0
    assert lengths.max() < 6000
    assert 1000 < np.median(lengths) < 5000


def test_adj_real_keeps_every_adjacency_arc(metr_la):
    adj, dist_direct = metr_la
    topo = RoutingTopology.from_adjacency_distances(adj, dist_direct)
    n_arcs = sum(len(nbrs) for nbrs in topo.neighbors)
    expected = int(((adj > 0) & ~np.eye(adj.shape[0], dtype=bool)).sum())
    assert n_arcs == expected


def test_connectivity_report_shape(metr_la):
    adj, dist_direct = metr_la
    report = RoutingTopology.from_adjacency_distances(adj, dist_direct).connectivity_report()
    assert report["source"] == "adj_real"
    assert report["n_nodes"] == adj.shape[0]
    assert 0.0 < report["reachable_od_fraction"] <= 1.0
    assert report["isolated_nodes"] >= 0


def test_od_sampling_rejects_unreachable_pairs():
    """Two disconnected cliques: no sampled pair may cross the gap."""
    n = 12
    adj = np.zeros((n, n), dtype=np.float64)
    dist = np.full((n, n), 20000.0)
    np.fill_diagonal(dist, 0.0)
    for block in (range(0, 6), range(6, 12)):
        for u in block:
            for v in block:
                if u != v:
                    adj[u, v] = 1.0
                    dist[u, v] = 6000.0
    topo = RoutingTopology.from_adjacency_distances(adj, dist)

    stats: dict = {}
    pairs = sample_od_pairs(dist, 40, min_dist_m=5000.0, seed=1, topology=topo, stats=stats)
    assert len(pairs) == 40
    for o, d in pairs:
        assert (o < 6) == (d < 6)
    assert stats["rejected_unreachable"] > 0
    assert stats["topology"] == "adj_real"


def test_od_sampling_without_topology_is_unfiltered():
    n = 8
    dist = np.full((n, n), 9000.0)
    np.fill_diagonal(dist, 0.0)
    pairs = sample_od_pairs(dist, 20, min_dist_m=5000.0, seed=2)
    assert len(pairs) == 20
    assert all(o != d for o, d in pairs)
