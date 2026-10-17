#!/usr/bin/env python3
"""Benchmark E1: legacy inv-based kriging vs KrigingSolver (same protocol)."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sparc_mfd.experiment import (  # noqa: E402
    ExperimentConfig,
    _prepare_data,
    _predict_encoder,
    _train_encoder,
    set_seed,
)
from sparc_mfd.gkf_fusion import GKFFusion, fuse_numpy  # noqa: E402
from sparc_mfd.metrics import mae  # noqa: E402
from sparc_mfd.sparse_impute import (  # noqa: E402
    ImputeConfig,
    KrigingSolver,
    _graph_kriging_impute_legacy,
    mask_sensors,
)


def _run_e1_legacy(mu, emb, timesteps, dist, cfg: ExperimentConfig, gkf: GKFFusion) -> dict:
    impute_cfg = ImputeConfig(blend_gnn=0.0)
    e1 = {}
    for sp in cfg.sparsity_levels:
        mask = mask_sensors(mu.shape[1], sp, cfg.seed + int(sp * 100))
        errs = []
        for t in timesteps:
            y_obs = np.where(mask, mu[t], 0.0)
            krig, kv = _graph_kriging_impute_legacy(y_obs, mask, dist, impute_cfg)
            gkf_mu, _ = fuse_numpy(mu[t], krig, kv, emb[t], mask, gkf)
            unobs = ~mask
            if unobs.sum():
                errs.append(mae(mu[t][unobs], gkf_mu[unobs]))
        e1[f"sparsity_{int(sp*100)}"] = float(np.mean(errs)) if errs else 0.0
    return e1


def _run_e1_solver(mu, emb, timesteps, dist, cfg: ExperimentConfig, gkf: GKFFusion) -> dict:
    impute_cfg = ImputeConfig(blend_gnn=0.0)
    krig_solver = KrigingSolver(dist, impute_cfg)
    e1 = {}
    for sp in cfg.sparsity_levels:
        mask = mask_sensors(mu.shape[1], sp, cfg.seed + int(sp * 100))
        y_batch = np.stack([np.where(mask, mu[t], 0.0) for t in timesteps])
        krig_batch, kv_batch = krig_solver.solve_batch(y_batch, mask)
        errs = []
        for t_i, t in enumerate(timesteps):
            gkf_mu, _ = fuse_numpy(mu[t], krig_batch[t_i], kv_batch[t_i], emb[t], mask, gkf)
            unobs = ~mask
            if unobs.sum():
                errs.append(mae(mu[t][unobs], gkf_mu[unobs]))
        e1[f"sparsity_{int(sp*100)}"] = float(np.mean(errs)) if errs else 0.0
        krig_solver.clear_cache()
    return e1


def _benchmark_kriging_only(mu, timesteps, dist, cfg, use_solver: bool) -> tuple:
    """Time kriging imputation only (no GKF), same protocol."""
    impute_cfg = ImputeConfig(blend_gnn=0.0)
    tracemalloc.start()
    t0 = time.perf_counter()
    if use_solver:
        solver = KrigingSolver(dist, impute_cfg)
        for sp in cfg.sparsity_levels:
            mask = mask_sensors(mu.shape[1], sp, cfg.seed + int(sp * 100))
            y_batch = np.stack([np.where(mask, mu[t], 0.0) for t in timesteps])
            solver.solve_batch(y_batch, mask)
            solver.clear_cache()
    else:
        for sp in cfg.sparsity_levels:
            mask = mask_sensors(mu.shape[1], sp, cfg.seed + int(sp * 100))
            for t in timesteps:
                y_obs = np.where(mask, mu[t], 0.0)
                _graph_kriging_impute_legacy(y_obs, mask, dist, impute_cfg)
    wall = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return wall, peak


def _benchmark(fn, label: str) -> tuple:
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    wall = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"  [{label}] {wall:.2f} s | peak RAM {peak / 1e6:.1f} MB")
    return result, wall, peak


def main() -> None:
    p = argparse.ArgumentParser(description="E1 kriging benchmark")
    p.add_argument("--dataset", default="metr-la", choices=["metr-la", "pems-bay"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    cfg = ExperimentConfig(
        dataset=args.dataset,
        data_dir=Path(f"data/{args.dataset}"),
        quick=args.quick,
        epochs=3 if args.quick else 10,
    )

    set_seed(cfg.seed)
    print(f"Benchmark E1 on {args.dataset} ({'quick' if cfg.quick else 'full'} timesteps)")
    data = _prepare_data(cfg)
    model = _train_encoder(data, cfg)
    mu, sigma, emb = _predict_encoder(
        model, data["x_test"], data["x_vol_test"],
        data["edge_index"], data["edge_weight"], cfg.device,
    )
    timesteps = list(range(0, len(mu), max(1, len(mu) // (10 if cfg.quick else 20))))
    print(f"  timesteps={len(timesteps)} sensors={mu.shape[1]} sparsity_levels={cfg.sparsity_levels}")

    gkf = GKFFusion(emb.shape[-1])
    set_seed(cfg.seed)

    legacy_fn = lambda: _run_e1_legacy(mu, emb, timesteps, data["dist"], cfg, gkf)
    solver_fn = lambda: _run_e1_solver(mu, emb, timesteps, data["dist"], cfg, gkf)

    e1_legacy, t_legacy, ram_legacy = _benchmark(legacy_fn, "legacy E1 full")
    e1_solver, t_solver, ram_solver = _benchmark(solver_fn, "KrigingSolver E1 full")

    print("  [kriging-only timing]")
    t_krig_legacy, ram_krig_legacy = _benchmark_kriging_only(mu, timesteps, data["dist"], cfg, False)
    t_krig_solver, ram_krig_solver = _benchmark_kriging_only(mu, timesteps, data["dist"], cfg, True)
    print(f"  [kriging legacy] {t_krig_legacy:.2f} s | RAM {ram_krig_legacy/1e6:.1f} MB")
    print(f"  [kriging solver] {t_krig_solver:.2f} s | RAM {ram_krig_solver/1e6:.1f} MB")

    mae_diff = {k: abs(e1_legacy[k] - e1_solver[k]) for k in e1_legacy}
    max_mae_diff = max(mae_diff.values()) if mae_diff else 0.0

    report = {
        "dataset": args.dataset,
        "quick": cfg.quick,
        "n_timesteps": len(timesteps),
        "n_sensors": int(mu.shape[1]),
        "legacy": {
            "wall_seconds": round(t_legacy, 4),
            "peak_ram_mb": round(ram_legacy / (1024 * 1024), 2),
            "e1_mae": e1_legacy,
        },
        "kriging_solver": {
            "wall_seconds": round(t_solver, 4),
            "peak_ram_mb": round(ram_solver / (1024 * 1024), 2),
            "e1_mae": e1_solver,
        },
        "kriging_only": {
            "legacy_wall_seconds": round(t_krig_legacy, 4),
            "solver_wall_seconds": round(t_krig_solver, 4),
            "kriging_speedup": round(t_krig_legacy / max(t_krig_solver, 1e-9), 2),
            "legacy_peak_ram_mb": round(ram_krig_legacy / (1024 * 1024), 2),
            "solver_peak_ram_mb": round(ram_krig_solver / (1024 * 1024), 2),
        },
        "speedup_full_e1": round(t_legacy / max(t_solver, 1e-9), 2),
        "ram_reduction_pct": round(100 * (1 - ram_solver / max(ram_legacy, 1)), 1),
        "equivalence": {
            "max_mae_difference": max_mae_diff,
            "per_sparsity_mae_diff": mae_diff,
            "passed": max_mae_diff < 1e-4,
        },
    }

    out = args.output or Path(f"outputs/{args.dataset}/{'quick' if cfg.quick else 'full'}/e1_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nKriging speed-up: {report['kriging_only']['kriging_speedup']}x | max MAE diff: {max_mae_diff:.2e} | passed={report['equivalence']['passed']}")
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
