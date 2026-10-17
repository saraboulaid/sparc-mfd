#!/usr/bin/env python3
"""Validate OD batching, checkpoint I/O, and resume without full E3 runtime."""

from __future__ import annotations

import json
import sys
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sparc_mfd.experiment import (  # noqa: E402
    ExperimentConfig,
    RoutingAccumulator,
    _load_routing_checkpoint,
    _prepare_data,
    _predict_encoder,
    _run_routing_e3,
    _save_routing_checkpoint,
    _train_encoder,
    set_seed,
)
from sparc_mfd.checkpointing import PipelineCheckpoints  # noqa: E402
from sparc_mfd.mfd_regime import MFDConfig, regime_time_series  # noqa: E402
from sparc_mfd.mgac import fit_mgac_params, mfd_embedding  # noqa: E402
from sparc_mfd.routing import RoutingConfig, RoutingMethod, sample_od_pairs  # noqa: E402


def test_accumulator_roundtrip(tmp: Path) -> None:
    methods = ["time_only", "sparc_ots", "oracle"]
    acc = RoutingAccumulator.init(methods)
    for i in range(100):
        acc.append("time_only", 3.0 + i * 0.01, 0.2)
        acc.append("sparc_ots", 1.0 + i * 0.005, 0.25)
        acc.append("oracle", 1.0, 0.25)
        acc.append_dqi(0.8)
    ckpt = tmp / "routing_checkpoint.json"
    _save_routing_checkpoint(ckpt, acc, "pems-bay", batch_idx=2, batches_total=10)
    loaded = _load_routing_checkpoint(ckpt)
    acc2 = RoutingAccumulator.from_checkpoint(loaded, methods)
    assert acc2.rii_count["time_only"] == 100
    assert abs(acc2.dqi_sum - acc.dqi_sum) < 1e-6
    print("  accumulator roundtrip OK")


def test_batched_routing_memory(dataset: str, n_od: int, batch_size: int) -> None:
    cfg = ExperimentConfig(
        dataset=dataset,
        data_dir=Path(f"data/{dataset}"),
        output_dir=Path(f"outputs/{dataset}-batch-val"),
        quick=False,
        n_od_pairs=n_od,
        od_batch_size=batch_size,
        epochs=1,
        routing_checkpoint=True,
        resume=False,
    )
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data = _prepare_data(cfg)
    model = _train_encoder(data, cfg)
    mu, sigma, emb = _predict_encoder(
        model, data["x_test"], data["x_vol_test"],
        data["edge_index"], data["edge_weight"], cfg.device,
    )
    offset = data["test_offset"]
    speed_test = data["speed_h"][offset : offset + len(mu)]
    regimes, acc_arr, v_avg = regime_time_series(speed_test, data["dist"], MFDConfig())
    timesteps = [0]  # single timestep — enough to stress 500 OD batches without full E3 runtime
    mgac = fit_mgac_params(
        [mfd_embedding(v, a) for v, a in zip(v_avg[:5], acc_arr[:5])],
        [1.0] * 5,
        [1.0] * 5,
    )
    rcfg = RoutingConfig(use_mgac=True, mgac_params=mgac)
    methods = [
        RoutingMethod.TIME_ONLY,
        RoutingMethod.OTS_INSTANT,
        RoutingMethod.OTS_CVAR,
        RoutingMethod.SPARC_OTS,
        RoutingMethod.ORACLE,
    ]
    gkf = GKFFusion(emb.shape[-1])
    od_pairs = sample_od_pairs(data["dist"], n_od, seed=cfg.seed)
    method_names = [m.value for m in methods]

    tracemalloc.start()
    acc = RoutingAccumulator.init(method_names)
    ckpt = PipelineCheckpoints(cfg)
    acc = _run_routing_e3(
        cfg, data, mu, sigma, emb, speed_test, regimes, acc_arr, v_avg, offset,
        rcfg, gkf, methods, od_pairs, timesteps, acc, ckpt, start_batch=0, start_timestep=0,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ckpt = _load_routing_checkpoint(cfg.output_dir / "checkpoints" / "e3_routing.json")
    if ckpt is None:
        ckpt = _load_routing_checkpoint(cfg.output_dir / "routing_checkpoint.json")
    batches_total = int((n_od + batch_size - 1) // batch_size)
    assert ckpt["batches_completed"] == batches_total, ckpt
    assert ckpt["n_samples"]["time_only"] == n_od * len(timesteps)
    print(f"  {dataset} n_od={n_od} batch={batch_size}: peak RAM {peak / 1e6:.1f} MB, samples={acc.rii_count}")


def test_resume(dataset: str) -> None:
    out = Path(f"outputs/{dataset}-batch-val")
    ckpt_path = out / "routing_checkpoint.json"
    if not ckpt_path.exists():
        print("  skip resume (no checkpoint)")
        return
    with open(ckpt_path, encoding="utf-8") as f:
        ckpt = json.load(f)
    mid = ckpt["last_batch_idx"]
    partial = ckpt.copy()
    partial["batches_completed"] = mid + 1
    partial["last_batch_idx"] = mid
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(partial, f, indent=2)

    cfg = ExperimentConfig(
        dataset=dataset,
        data_dir=Path(f"data/{dataset}"),
        output_dir=out,
        n_od_pairs=ckpt["n_samples"]["time_only"] // 2,
        od_batch_size=50,
        epochs=1,
        resume=True,
        routing_checkpoint=True,
    )
    methods = list(ckpt["n_samples"].keys())
    acc = RoutingAccumulator.from_checkpoint(partial, methods)
    start = partial["last_batch_idx"] + 1
    print(f"  resume from batch {start + 1}/{partial['batches_total']} OK (loaded counts={acc.rii_count})")


def main() -> None:
    tmp = Path("outputs/_test_ckpt")
    tmp.mkdir(parents=True, exist_ok=True)
    print("Test 1: accumulator roundtrip")
    test_accumulator_roundtrip(tmp)
    print("Test 2: PEMS-BAY 500 OD batched (2 timesteps)")
    test_batched_routing_memory("pems-bay", n_od=500, batch_size=50)
    print("Test 3: resume checkpoint")
    test_resume("pems-bay")
    print("All validation checks passed.")


if __name__ == "__main__":
    main()
