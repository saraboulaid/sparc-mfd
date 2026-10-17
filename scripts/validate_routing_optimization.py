#!/usr/bin/env python3
"""Non-regression check: legacy dense vs sparse topologies vs scipy backend."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sparc_mfd.experiment import ExperimentConfig, _prepare_data, set_seed  # noqa: E402
from sparc_mfd.mfd_regime import MFDConfig, MFDRegime, regime_time_series  # noqa: E402
from sparc_mfd.mgac import fit_mgac_params, mfd_embedding  # noqa: E402
from sparc_mfd.routing import (  # noqa: E402
    RoutingConfig,
    RoutingMethod,
    RoutingTopology,
    build_routing_graph,
    dijkstra,
    sample_od_pairs,
)

METHODS = [
    RoutingMethod.TIME_ONLY,
    RoutingMethod.OTS_INSTANT,
    RoutingMethod.OTS_CVAR,
    RoutingMethod.SPARC_OTS,
    RoutingMethod.ORACLE,
]
RTOL = 1e-4


def _rel_close(a: float, b: float, rtol: float = RTOL) -> bool:
    if not (abs(a) < float("inf") and abs(b) < float("inf")):
        return a == b
    if a == b:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= rtol


def _run_variant(
    graph,
    topology: RoutingTopology,
    backend: str,
    od_pairs,
    timesteps,
    speed_test,
    mu,
    sigma,
    regimes,
    acc_arr,
    v_avg,
    rcfg,
    offset,
    ots_h,
) -> dict:
    results = {}
    for t_i in timesteps:
        regime = MFDRegime(int(regimes[t_i]))
        mfd_st = mfd_embedding(float(v_avg[t_i]), float(acc_arr[t_i]))
        g_sparse = build_routing_graph(graph["dist"], speed_test[t_i], mu[t_i], sigma[t_i])
        g_oracle = build_routing_graph(graph["dist"], speed_test[t_i], mu[t_i], sigma[t_i] * 0.5)
        hist = ots_h[max(0, offset + t_i - 12) : offset + t_i + 1]
        hist_slice = hist[-12:] if len(hist) > 0 else None
        for o, d in od_pairs:
            for method in METHODS:
                g = g_oracle if method == RoutingMethod.ORACLE else g_sparse
                _, cost, met = dijkstra(
                    g, o, d, method, regime, rcfg, hist_slice, mfd_st,
                    return_path=False, topology=topology, backend=backend,
                )
                key = f"t{t_i}_o{o}_d{d}_{method.value}"
                results[key] = {
                    "cost": float(cost),
                    "rii": float(met["rii"]),
                    "travel_time": float(met["travel_time"]),
                    "cvar": float(met.get("cvar", 0.0)),
                }
    return results


def _compare_to_legacy(legacy: dict, candidate: dict) -> tuple[int, int, list]:
    matches = 0
    total = 0
    mismatches = []
    for key, ref in legacy.items():
        if key not in candidate:
            continue
        total += 1
        cand = candidate[key]
        ok = all(
            _rel_close(ref[m], cand[m])
            for m in ("cost", "rii", "travel_time", "cvar")
        )
        if ok:
            matches += 1
        else:
            mismatches.append({"key": key, "legacy": ref, "candidate": cand})
    return matches, total, mismatches


def main() -> None:
    p = argparse.ArgumentParser(description="Validate E3 routing optimizations")
    p.add_argument("--dataset", default="metr-la", choices=["metr-la", "pems-bay"])
    p.add_argument("--n-od", type=int, default=10)
    p.add_argument("--n-timesteps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/routing_regression_report.json")
    args = p.parse_args()

    cfg = ExperimentConfig(
        dataset=args.dataset,
        data_dir=Path(f"data/{args.dataset}"),
        quick=True,
        seed=args.seed,
    )
    set_seed(cfg.seed)
    t0 = time.perf_counter()
    print(f"Loading {args.dataset}...")
    data = _prepare_data(cfg)

    # Lightweight proxy for mu/sigma at test timesteps (no full E0 training)
    offset = data["test_offset"]
    speed_h = data["speed_h"]
    n_test = min(50, len(speed_h) - offset - 1)
    speed_test = speed_h[offset : offset + n_test]
    rng = __import__("numpy").random.default_rng(cfg.seed)
    mu = rng.uniform(0.5, 2.0, size=(n_test, speed_test.shape[1])).astype("float32")
    sigma = rng.uniform(0.1, 0.5, size=(n_test, speed_test.shape[1])).astype("float32")

    regimes, acc_arr, v_avg = regime_time_series(speed_test, data["dist"], MFDConfig())
    mgac = fit_mgac_params(
        [mfd_embedding(v, a) for v, a in zip(v_avg[:5], acc_arr[:5])],
        [1.0] * 5,
        [1.0] * 5,
    )
    rcfg = RoutingConfig(use_mgac=True, mgac_params=mgac)

    od_pairs = sample_od_pairs(data["dist"], args.n_od, seed=cfg.seed)
    step = max(1, n_test // args.n_timesteps)
    timesteps = list(range(0, n_test, step))[: args.n_timesteps]

    graph_stub = {"dist": data["dist"]}
    topologies = data["routing_topologies"]

    print("Running legacy dense baseline...")
    legacy = _run_variant(
        graph_stub, topologies["legacy_dense"], "legacy",
        od_pairs, timesteps, speed_test, mu, sigma,
        regimes, acc_arr, v_avg, rcfg, offset, data["ots_h"],
    )

    variants = {
        "csv_direct_python": (topologies["csv_direct"], "python"),
        "csv_direct_scipy": (topologies["csv_direct"], "scipy"),
        "adj_mx_python": (topologies["adj_mx"], "python"),
        "adj_mx_scipy": (topologies["adj_mx"], "scipy"),
    }

    report = {
        "dataset": args.dataset,
        "n_od_pairs": args.n_od,
        "timesteps": timesteps,
        "legacy_keys": len(legacy),
        "variants": {},
        "elapsed_s": 0.0,
    }

    best_topology = "csv_direct"
    best_score = -1.0
    topology_scores: dict[str, float] = {"csv_direct": 0.0, "adj_mx": 0.0}

    for name, (topo, backend) in variants.items():
        print(f"Running {name}...")
        cand = _run_variant(
            graph_stub, topo, backend,
            od_pairs, timesteps, speed_test, mu, sigma,
            regimes, acc_arr, v_avg, rcfg, offset, data["ots_h"],
        )
        matches, total, mismatches = _compare_to_legacy(legacy, cand)
        score = matches / max(total, 1)
        report["variants"][name] = {
            "topology": topo.source,
            "backend": backend,
            "matches": matches,
            "total": total,
            "match_rate": score,
            "sample_mismatches": mismatches[:5],
            "mean_degree": topo.mean_degree(),
        }
        print(f"  {name}: {matches}/{total} match ({100 * score:.1f}%)")
        src = topo.source
        if src in topology_scores:
            topology_scores[src] = max(topology_scores[src], score)
        if score > best_score:
            best_score = score
            best_topology = topo.source

    if topology_scores["csv_direct"] >= topology_scores["adj_mx"]:
        selected = "csv_direct"
    else:
        selected = "adj_mx"

    report["topology_scores"] = topology_scores
    report["selected_topology"] = selected
    report["selected_backend"] = "scipy"
    report["elapsed_s"] = time.perf_counter() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSelected topology: {selected} (backend=scipy)")
    print(f"Report -> {out}")
    print(f"Elapsed: {report['elapsed_s']:.1f}s")

    if topology_scores[selected] < 0.5:
        print("WARNING: low match rate vs legacy; review report before FULL run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
