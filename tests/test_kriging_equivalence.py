"""Numerical equivalence tests: KrigingSolver vs legacy inv-based kriging."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sparc_mfd.graph import build_distance_matrix
from sparc_mfd.sparse_impute import (
    ImputeConfig,
    KrigingSolver,
    _graph_kriging_impute_legacy,
    graph_kriging_impute,
    mask_sensors,
)


def _make_dist(n: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10000, size=(n, 2))
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


def _equivalence_report(
    old_vals: np.ndarray,
    new_vals: np.ndarray,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict:
    diff = np.abs(old_vals.astype(np.float64) - new_vals.astype(np.float64))
    denom = np.maximum(np.abs(old_vals.astype(np.float64)), 1e-12)
    rel = diff / denom
    passed = bool(np.max(diff) < atol or np.max(rel) < rtol)
    return {
        "max_absolute_difference": float(np.max(diff)),
        "mean_absolute_difference": float(np.mean(diff)),
        "max_relative_difference": float(np.max(rel)),
        "mean_relative_difference": float(np.mean(rel)),
        "atol": atol,
        "rtol": rtol,
        "passed": passed,
    }


@pytest.fixture
def dist_matrix():
    return _make_dist(80, seed=42)


@pytest.fixture
def cfg():
    return ImputeConfig(blend_gnn=0.0)


def test_solve_matches_legacy(dist_matrix, cfg):
    rng = np.random.default_rng(123)
    n = dist_matrix.shape[0]
    y = rng.uniform(0.1, 2.0, size=n)
    for sp in (0.05, 0.2, 0.5):
        mask = mask_sensors(n, sp, seed=7)
        y_obs = np.where(mask, y, 0.0)
        old_y, old_v = _graph_kriging_impute_legacy(y_obs, mask, dist_matrix, cfg)
        solver = KrigingSolver(dist_matrix, cfg)
        new_y, new_v = solver.solve(y_obs, mask)
        rep_y = _equivalence_report(old_y, new_y)
        rep_v = _equivalence_report(old_v, new_v)
        assert rep_y["passed"], rep_y
        assert rep_v["passed"], rep_v


def test_solve_batch_matches_legacy_loop(dist_matrix, cfg):
    rng = np.random.default_rng(99)
    n = dist_matrix.shape[0]
    t_steps = 15
    y_batch = rng.uniform(0.1, 2.0, size=(t_steps, n))
    mask = mask_sensors(n, 0.1, seed=11)
    solver = KrigingSolver(dist_matrix, cfg)

    old_stack = []
    for t in range(t_steps):
        y_obs = np.where(mask, y_batch[t], 0.0)
        old_y, _ = _graph_kriging_impute_legacy(y_obs, mask, dist_matrix, cfg)
        old_stack.append(old_y)
    old_stack = np.stack(old_stack)

    new_stack, _ = solver.solve_batch(
        np.where(mask, y_batch, 0.0), mask
    )
    rep = _equivalence_report(old_stack, new_stack)
    assert rep["passed"], rep


def test_graph_kriging_impute_wrapper(dist_matrix, cfg):
    rng = np.random.default_rng(5)
    n = dist_matrix.shape[0]
    y = rng.uniform(0.2, 1.5, size=n)
    mask = mask_sensors(n, 0.2, seed=3)
    y_obs = np.where(mask, y, 0.0)
    old_y, _ = _graph_kriging_impute_legacy(y_obs, mask, dist_matrix, cfg)
    new_y, _ = graph_kriging_impute(y_obs, mask, dist_matrix, cfg)
    rep = _equivalence_report(old_y, new_y)
    assert rep["passed"], rep


def test_factorization_cache_reuse(dist_matrix, cfg):
    solver = KrigingSolver(dist_matrix, cfg)
    n = dist_matrix.shape[0]
    mask = mask_sensors(n, 0.15, seed=1)
    y = np.linspace(0.1, 1.0, n)
    y_obs = np.where(mask, y, 0.0)
    solver.solve(y_obs, mask)
    assert len(solver._cache) >= 1
    solver.solve(y_obs * 1.01, mask)
    assert len(solver._cache) == 1


def test_write_equivalence_report(tmp_path, dist_matrix, cfg):
    """Generate kriging_equivalence_report.json for manual inspection."""
    rng = np.random.default_rng(0)
    n = dist_matrix.shape[0]
    reports = []
    solver = KrigingSolver(dist_matrix, cfg)
    for sp in (0.05, 0.1, 0.2, 0.5):
        mask = mask_sensors(n, sp, seed=int(sp * 100))
        y_batch = rng.uniform(0.1, 2.0, size=(20, n))
        y_obs_batch = np.where(mask, y_batch, 0.0)
        old = []
        for t in range(20):
            oy, _ = _graph_kriging_impute_legacy(y_obs_batch[t], mask, dist_matrix, cfg)
            old.append(oy)
        old = np.stack(old)
        new, _ = solver.solve_batch(y_obs_batch, mask)
        unobs = ~mask
        old_mae = float(np.mean(np.abs(old[:, unobs] - y_batch[:, unobs])))
        new_mae = float(np.mean(np.abs(new[:, unobs] - y_batch[:, unobs])))
        rep = _equivalence_report(old, new)
        rep.update(
            sparsity=sp,
            old_imputation_mae=old_mae,
            new_imputation_mae=new_mae,
            dtype_context="float32_output",
        )
        reports.append(rep)
        solver.clear_cache()

    out = {
        "all_passed": all(r["passed"] for r in reports),
        "reports": reports,
    }
    out_path = tmp_path / "kriging_equivalence_report.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    assert out["all_passed"]
