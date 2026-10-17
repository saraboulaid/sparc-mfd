"""SPARC-OTS experiment pipeline (E0-E5) with batched OD routing + checkpoints."""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data_loader import (
    ensure_dataset,
    inspect_dataset,
    load_direct_distances_csv,
    load_distances_csv,
    load_metr_la_speed,
    read_h5_sensor_ids,
)
from .assignment import AssignmentConfig, frank_wolfe_assignment
from .checkpointing import PipelineCheckpoints, resolve_output_dir
from .profiling import StageProfiler
from .decision_support import summarize_decision_support
from .features import (
    aggregate_hourly,
    build_dual_branch_features,
    build_temporal_features,
    normalize_train_only,
    sliding_windows,
)
from .gkf_fusion import GKFFusion, fuse_numpy
from .graph import build_distance_matrix, load_adjacency
from .metrics import mae, regret, rmse, summarize_routing_results
from .mgac import fit_mgac_params, mfd_embedding
from .mfd_regime import MFDConfig, MFDRegime, regime_time_series
from .ots_encoder import OTSSTEncoder, compute_ots_components, compute_ots_scalar, fit_ots_stats, ots_loss
from .routing import (
    RoutingConfig,
    RoutingMethod,
    RoutingTopology,
    build_routing_graph,
    sample_od_pairs,
)
from .routing_worker import init_worker as _init_od_worker, route_od_pair as _route_od_pair_worker
from .sparse_impute import ImputeConfig, KrigingSolver, graph_kriging_impute, mask_sensors

MAX_SAMPLES = 5000


@dataclass
class ExperimentConfig:
    dataset: str = "metr-la"
    data_dir: Path = Path("data/metr-la")
    output_dir: Path = Path("outputs/metr-la")
    seed: int = 42
    sparsity_levels: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.50])
    n_od_pairs: int = 1000
    od_batch_size: int = 50
    seq_len: int = 12
    epochs: int = 10
    hidden_dim: int = 64
    batch_size: int = 32
    quick: bool = False
    allow_synthetic: bool = False
    device: str = "cpu"
    routing_checkpoint: bool = True
    resume: bool = True
    profile: bool = True
    routing_backend: str = "scipy"
    routing_topology: str = "auto"
    od_workers: int = 0


@dataclass
class RoutingAccumulator:
    """Online stats + capped samples for E3 routing metrics."""

    methods: List[str]
    rii_sum: Dict[str, float] = field(default_factory=dict)
    rii_sumsq: Dict[str, float] = field(default_factory=dict)
    rii_count: Dict[str, int] = field(default_factory=dict)
    tt_sum: Dict[str, float] = field(default_factory=dict)
    tt_sumsq: Dict[str, float] = field(default_factory=dict)
    tt_count: Dict[str, int] = field(default_factory=dict)
    rii_samples: Dict[str, List[float]] = field(default_factory=dict)
    tt_samples: Dict[str, List[float]] = field(default_factory=dict)
    dqi_sum: float = 0.0
    dqi_count: int = 0

    @classmethod
    def init(cls, methods: List[str]) -> "RoutingAccumulator":
        acc = cls(methods=methods)
        for m in methods:
            acc.rii_sum[m] = acc.rii_sumsq[m] = acc.tt_sum[m] = acc.tt_sumsq[m] = 0.0
            acc.rii_count[m] = acc.tt_count[m] = 0
            acc.rii_samples[m] = []
            acc.tt_samples[m] = []
        return acc

    def append(self, method: str, rii: float, travel_time: float) -> None:
        self.rii_sum[method] += rii
        self.rii_sumsq[method] += rii * rii
        self.rii_count[method] += 1
        self.tt_sum[method] += travel_time
        self.tt_sumsq[method] += travel_time * travel_time
        self.tt_count[method] += 1
        if len(self.rii_samples[method]) < MAX_SAMPLES:
            self.rii_samples[method].append(rii)
            self.tt_samples[method].append(travel_time)

    def append_dqi(self, dqi: float) -> None:
        self.dqi_sum += dqi
        self.dqi_count += 1

    def mean_rii(self, method: str) -> float:
        n = self.rii_count.get(method, 0)
        if n == 0:
            return 0.0
        return self.rii_sum[method] / n

    def to_summary_rii(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for m in self.methods:
            samples = self.rii_samples.get(m, [])
            if samples:
                arr = np.array(samples)
                out[m] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                }
            elif self.rii_count.get(m, 0) > 0:
                n = self.rii_count[m]
                mean = self.rii_sum[m] / n
                var = max(0.0, self.rii_sumsq[m] / n - mean * mean)
                out[m] = {"mean": mean, "std": float(np.sqrt(var)), "median": mean}
        oracle = out.get("oracle", {})
        for m, stats in out.items():
            if m != "oracle" and oracle and stats:
                stats["regret_vs_oracle"] = regret(
                    np.array([stats["mean"]]), np.array([oracle["mean"]])
                )
        return out

    def to_summary_tt(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for m in self.methods:
            samples = self.tt_samples.get(m, [])
            if samples:
                arr = np.array(samples)
                out[m] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                }
            elif self.tt_count.get(m, 0) > 0:
                n = self.tt_count[m]
                mean = self.tt_sum[m] / n
                var = max(0.0, self.tt_sumsq[m] / n - mean * mean)
                out[m] = {"mean": mean, "std": float(np.sqrt(var)), "median": mean}
        return out

    def to_checkpoint_dict(
        self,
        dataset: str,
        batch_idx: int,
        batches_total: int,
        timestep_idx: Optional[int] = None,
        timesteps_per_batch: Optional[int] = None,
    ) -> Dict:
        out = {
            "dataset": dataset,
            "batches_completed": batch_idx + 1,
            "batches_total": batches_total,
            "n_samples": dict(self.rii_count),
            "rii_sum": dict(self.rii_sum),
            "rii_sumsq": dict(self.rii_sumsq),
            "travel_time_sum": dict(self.tt_sum),
            "travel_time_sumsq": dict(self.tt_sumsq),
            "dqi_sum": self.dqi_sum,
            "dqi_count": self.dqi_count,
            "last_batch_idx": batch_idx,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if timestep_idx is not None:
            out["last_timestep_idx"] = timestep_idx
        if timesteps_per_batch is not None:
            out["timesteps_per_batch"] = timesteps_per_batch
        return out

    @classmethod
    def from_checkpoint(cls, data: Dict, methods: List[str]) -> "RoutingAccumulator":
        acc = cls.init(methods)
        acc.rii_sum = {k: float(v) for k, v in data.get("rii_sum", {}).items()}
        acc.rii_sumsq = {k: float(v) for k, v in data.get("rii_sumsq", {}).items()}
        acc.rii_count = {k: int(v) for k, v in data.get("n_samples", {}).items()}
        acc.tt_sum = {k: float(v) for k, v in data.get("travel_time_sum", {}).items()}
        acc.tt_sumsq = {k: float(v) for k, v in data.get("travel_time_sumsq", {}).items()}
        acc.tt_count = dict(acc.rii_count)
        acc.dqi_sum = float(data.get("dqi_sum", 0.0))
        acc.dqi_count = int(data.get("dqi_count", 0))
        for m in methods:
            acc.rii_samples.setdefault(m, [])
            acc.tt_samples.setdefault(m, [])
        return acc


def _save_routing_checkpoint(path: Path, acc: RoutingAccumulator, dataset: str, batch_idx: int, batches_total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(acc.to_checkpoint_dict(dataset, batch_idx, batches_total), f, indent=2)


def _load_routing_checkpoint(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _effective_od_config(cfg: ExperimentConfig) -> Tuple[int, int]:
    if cfg.quick:
        return max(cfg.n_od_pairs, 20), min(cfg.od_batch_size, 10)
    return cfg.n_od_pairs, cfg.od_batch_size


def _effective_od_workers(cfg: ExperimentConfig) -> int:
    if cfg.od_workers == 1:
        return 1
    if cfg.od_workers > 0:
        requested = cfg.od_workers
    else:
        n_cpu = os.cpu_count() or 1
        requested = max(1, min(8, n_cpu - 1))
    if os.name == "nt" and requested > 1:
        print("  [E3] Windows: serial OD routing (pool unstable); use Linux for --od-workers > 1")
        return 1
    return requested


def _run_od_batch(
    od_batch: List[Tuple[int, int]],
    task_base: Dict,
    accumulator: RoutingAccumulator,
    od_workers: int,
) -> None:
    _init_od_worker(task_base)
    if od_workers <= 1:
        for od in od_batch:
            _merge_od_results(accumulator, [_route_od_pair_worker(od)])
        return
    try:
        with ProcessPoolExecutor(
            max_workers=od_workers,
            initializer=_init_od_worker,
            initargs=(task_base,),
        ) as pool:
            results = list(pool.map(_route_od_pair_worker, od_batch))
        _merge_od_results(accumulator, results)
    except Exception as exc:
        print(f"  [E3] Parallel pool failed ({exc}), falling back to serial")
        _init_od_worker(task_base)
        for od in od_batch:
            _merge_od_results(accumulator, [_route_od_pair_worker(od)])


def resolve_routing_topology(cfg: ExperimentConfig, data: Dict) -> RoutingTopology:
    source = cfg.routing_topology
    if source == "auto":
        report_path = Path("outputs/routing_regression_report.json")
        if report_path.exists():
            with open(report_path, encoding="utf-8") as f:
                source = json.load(f).get("selected_topology", "csv_direct")
        else:
            source = "csv_direct"
    topologies = data["routing_topologies"]
    if source not in topologies:
        source = "csv_direct"
    topo = topologies[source]
    print(f"  [E3] Routing topology: {topo.source} (mean degree {topo.mean_degree():.1f})")
    return topo


def _merge_od_results(accumulator: RoutingAccumulator, results: List[Dict]) -> None:
    for res in results:
        for method, rii, tt in res.get("appends", []):
            accumulator.append(method, rii, tt)
        for dqi in res.get("dqi", []):
            accumulator.append_dqi(dqi)


def _prepare_data(cfg: ExperimentConfig):
    allow_synthetic = cfg.quick or cfg.allow_synthetic
    h5, adj_path, dist_path = ensure_dataset(
        cfg.data_dir, cfg.dataset, allow_synthetic=allow_synthetic
    )
    data_report = inspect_dataset(cfg.data_dir, cfg.dataset)
    speed = load_metr_la_speed(h5)
    edge_index, edge_weight, adj = load_adjacency(adj_path)
    n = adj.shape[0]
    sensor_ids = read_h5_sensor_ids(h5)
    dist_direct = load_direct_distances_csv(dist_path, n, sensor_ids=sensor_ids)
    dist = load_distances_csv(dist_path, n, sensor_ids=sensor_ids)
    if not np.isfinite(dist).any() or dist.max() > 1e5:
        dist = build_distance_matrix(adj)
        dist_direct = dist.copy()

    routing_topologies = {
        "csv_direct": RoutingTopology.from_direct_distances(dist_direct, source="csv_direct"),
        "adj_mx": RoutingTopology.from_adjacency(adj, source="adj_mx"),
        "legacy_dense": RoutingTopology.from_dense_distances(dist, source="legacy_dense"),
    }

    speed_h = aggregate_hourly(speed)
    train_end = int(len(speed_h) * 0.7)
    q_thr = np.quantile(speed_h[:train_end], 0.20, axis=0)
    comp = compute_ots_components(speed_h, q_thresholds=q_thr)
    stats = fit_ots_stats(comp, train_end)
    stats["q_thresholds"] = q_thr
    ots_h = compute_ots_scalar(comp, stats)

    x_trend = build_temporal_features(speed_h)
    _, vol_feat = build_dual_branch_features(speed_h)
    x_trend_n, _, _ = normalize_train_only(x_trend)
    x_all, y_all = sliding_windows(x_trend_n, ots_h, cfg.seq_len)

    vol_windows = [vol_feat[i : i + cfg.seq_len] for i in range(len(x_trend_n) - cfg.seq_len)]
    x_vol_all = np.array(vol_windows, dtype=np.float32)
    aux_all = np.zeros((len(y_all), n, 3), dtype=np.float32)
    for i in range(len(y_all)):
        t_idx = i + cfg.seq_len
        if t_idx < len(comp["V"]):
            for j, key in enumerate(("V", "D", "P")):
                aux_all[i, :, j] = comp[key][t_idx]

    split = int(len(x_all) * 0.7)
    val_split = int(len(x_all) * 0.8)
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "adj": adj,
        "dist": dist,
        "dist_direct": dist_direct,
        "routing_topologies": routing_topologies,
        "speed_h": speed_h,
        "ots_h": ots_h,
        "x_train": x_all[:split],
        "x_vol_train": x_vol_all[:split],
        "y_train": y_all[:split],
        "aux_train": aux_all[:split],
        "x_test": x_all[val_split:],
        "x_vol_test": x_vol_all[val_split:],
        "y_test": y_all[val_split:],
        "aux_test": aux_all[val_split:],
        "test_offset": val_split + cfg.seq_len,
        "data_provenance": data_report.to_provenance(),
    }


def _train_encoder(data, cfg: ExperimentConfig) -> OTSSTEncoder:
    device = torch.device(cfg.device)
    model = OTSSTEncoder(cfg.hidden_dim).to(device)
    ei, ew = data["edge_index"].to(device), data["edge_weight"].to(device)
    ds = TensorDataset(
        torch.from_numpy(data["x_train"]).float(),
        torch.from_numpy(data["x_vol_train"]).float(),
        torch.from_numpy(data["y_train"]).float(),
        torch.from_numpy(data["aux_train"]).float(),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 3 if cfg.quick else cfg.epochs
    model.train()
    for _ in range(epochs):
        for xb, xv, yb, ab in loader:
            xb, xv, yb, ab = xb.to(device), xv.to(device), yb.to(device), ab.to(device)
            mu, logvar, _, aux = model(xb, xv, ei, ew)
            loss = ots_loss(yb, mu, logvar, aux, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def _predict_encoder(model, x, xv, edge_index, edge_weight, device):
    model.eval()
    dev = torch.device(device)
    mu, logvar, emb, _ = model(
        torch.from_numpy(x).float().to(dev),
        torch.from_numpy(xv).float().to(dev),
        edge_index.to(dev),
        edge_weight.to(dev),
    )
    return mu.cpu().numpy(), np.exp(0.5 * logvar.cpu().numpy()), emb.cpu().numpy()


def _run_e1_imputation(
    cfg: ExperimentConfig,
    mu: np.ndarray,
    emb: np.ndarray,
    timesteps: List[int],
    dist: np.ndarray,
    ckpt: PipelineCheckpoints,
    profiler: Optional[StageProfiler] = None,
) -> Dict:
    if ckpt.is_done("E1"):
        loaded = ckpt.load_stage("E1")
        if loaded:
            print("  [E1] loaded from checkpoint")
            return loaded.get("e1", loaded)
    impute_cfg = ImputeConfig(blend_gnn=0.0)
    krig_solver = KrigingSolver(dist, impute_cfg)
    gkf_e1 = GKFFusion(emb.shape[-1])
    n_sp = len(cfg.sparsity_levels)
    e1: Dict = {}
    for sp_i, sp in enumerate(cfg.sparsity_levels):
        if profiler:
            profiler.start_sub(f"sparsity_{int(sp * 100)}")
        mask = mask_sensors(mu.shape[1], sp, cfg.seed + int(sp * 100))
        print(f"  [E1] Sparsity {sp_i + 1}/{n_sp} ({int(sp * 100)}%)")
        y_batch = np.stack([np.where(mask, mu[t], 0.0) for t in timesteps])
        krig_batch, kv_batch = krig_solver.solve_batch(y_batch, mask)
        errs = []
        for t_i, t in enumerate(timesteps):
            gkf_mu, _ = fuse_numpy(mu[t], krig_batch[t_i], kv_batch[t_i], emb[t], mask, gkf_e1)
            unobs = ~mask
            if unobs.sum():
                errs.append(mae(mu[t][unobs], gkf_mu[unobs]))
        key = f"sparsity_{int(sp*100)}"
        e1[key] = {"mae": float(np.mean(errs)) if errs else 0.0}
        krig_solver.clear_cache()
        if cfg.routing_checkpoint:
            ckpt.save_stage("E1", {"e1": dict(e1)}, partial_metrics={key: e1[key]}, status="in_progress")
        if profiler:
            profiler.end_sub()
    if cfg.routing_checkpoint:
        ckpt.save_stage("E1", {"e1": e1})
    return e1


def _run_e4_assignment(
    cfg: ExperimentConfig,
    data: Dict,
    mu: np.ndarray,
    sigma: np.ndarray,
    speed_test: np.ndarray,
    regimes: np.ndarray,
    od_pairs: List[Tuple[int, int]],
    timesteps: List[int],
) -> Dict:
    t0 = timesteps[0]
    graph = build_routing_graph(data["dist"], speed_test[t0], mu[t0], sigma[t0] * 0.5)
    n_assign = min(50, len(od_pairs))
    pairs = od_pairs[:n_assign]
    demands = np.ones(n_assign, dtype=np.float64)
    regime = int(regimes[t0])
    assign_cfg = AssignmentConfig()
    e4 = {}
    for key, method in (
        ("time_only", RoutingMethod.TIME_ONLY),
        ("risk_cvar_fixed", RoutingMethod.OTS_CVAR),
        ("sparc_mfd", RoutingMethod.SPARC_OTS),
    ):
        e4[key] = frank_wolfe_assignment(graph, pairs, demands, method, regime, assign_cfg)
    return e4


def _run_routing_e3(
    cfg: ExperimentConfig,
    data: Dict,
    mu: np.ndarray,
    sigma: np.ndarray,
    emb: np.ndarray,
    speed_test: np.ndarray,
    regimes: np.ndarray,
    acc_arr: np.ndarray,
    v_avg: np.ndarray,
    offset: int,
    rcfg: RoutingConfig,
    gkf_model: GKFFusion,
    methods: List[RoutingMethod],
    od_pairs: List[Tuple[int, int]],
    test_idx: List[int],
    accumulator: RoutingAccumulator,
    ckpt: PipelineCheckpoints,
    profiler: Optional[StageProfiler] = None,
    start_batch: int = 0,
    start_timestep: int = 0,
    topology: Optional[RoutingTopology] = None,
) -> RoutingAccumulator:
    n_od, batch_size = _effective_od_config(cfg)
    od_pairs = od_pairs[:n_od]
    batches = [od_pairs[i : i + batch_size] for i in range(0, len(od_pairs), batch_size)]
    krig_solver = KrigingSolver(data["dist"], ImputeConfig())
    n_od_done = start_batch * batch_size
    n_timesteps = len(test_idx)
    log_every = max(1, n_timesteps // 5)
    od_workers = _effective_od_workers(cfg)
    backend = cfg.routing_backend
    if topology is None:
        topology = resolve_routing_topology(cfg, data)

    if start_batch > 0 or start_timestep > 0:
        print(
            f"  [E3] Resuming batch {start_batch + 1}/{len(batches)} "
            f"timestep {start_timestep + 1}/{n_timesteps} "
            f"({n_od_done}/{n_od} OD done)"
        )
    else:
        print(
            f"  [E3] Starting routing: {len(batches)} batches, {n_od} OD pairs, "
            f"{n_timesteps} timesteps/batch, batch_size={batch_size}, "
            f"backend={backend}, workers={od_workers}"
        )

    method_names = [m.value for m in methods]

    for batch_idx in range(start_batch, len(batches)):
        od_batch = batches[batch_idx]
        t_batch = time.perf_counter()
        ts_begin = start_timestep if batch_idx == start_batch else 0
        print(
            f"  [E3] Batch {batch_idx + 1}/{len(batches)} started "
            f"({len(od_batch)} OD, {n_timesteps} timesteps)..."
        )

        for step_i, t_i in enumerate(test_idx):
            if step_i < ts_begin:
                continue

            mask20 = mask_sensors(mu.shape[1], 0.20, cfg.seed)
            krig, kv = graph_kriging_impute(
                mu[t_i] * mask20, mask20, data["dist"], solver=krig_solver
            )
            mu_gkf, sig_gkf = fuse_numpy(mu[t_i], krig, kv, emb[t_i], mask20, gkf_model)
            regime = MFDRegime(int(regimes[t_i]))
            mfd_st = mfd_embedding(float(v_avg[t_i]), float(acc_arr[t_i]))
            graphs = {
                "sparse": build_routing_graph(data["dist"], speed_test[t_i], mu_gkf, sig_gkf),
                "oracle": build_routing_graph(data["dist"], speed_test[t_i], mu[t_i], sigma[t_i] * 0.5),
                "instant": build_routing_graph(data["dist"], speed_test[t_i], mu[t_i], sigma[t_i]),
            }
            hist = data["ots_h"][max(0, offset + t_i - 12) : offset + t_i + 1]
            hist_slice = hist[-12:] if len(hist) > 0 else None

            task_base = {
                "graphs": graphs,
                "method_names": method_names,
                "regime": int(regime),
                "rcfg": rcfg,
                "hist_slice": hist_slice,
                "mfd_st": mfd_st,
                "topology": topology,
                "backend": backend,
            }

            _run_od_batch(od_batch, task_base, accumulator, od_workers)

            if cfg.routing_checkpoint:
                ckpt.save_e3_progress(
                    accumulator, batch_idx, len(batches),
                    timestep_idx=step_i, timesteps_per_batch=n_timesteps, quiet=True,
                )

            if (step_i + 1) % log_every == 0 or (step_i + 1) == n_timesteps:
                elapsed = time.perf_counter() - t_batch
                print(
                    f"  [E3] Batch {batch_idx + 1}/{len(batches)} - "
                    f"timestep {step_i + 1}/{n_timesteps} ({elapsed:.0f}s elapsed)"
                )

        n_od_done += len(od_batch)
        elapsed = time.perf_counter() - t_batch
        start_timestep = 0
        if cfg.routing_checkpoint:
            ckpt.save_e3_progress(
                accumulator, batch_idx, len(batches),
                timestep_idx=n_timesteps - 1, timesteps_per_batch=n_timesteps,
            )
            print(
                f"  [E3] Batch {batch_idx + 1}/{len(batches)} - "
                f"{n_od_done}/{n_od} OD in {elapsed:.1f}s | checkpoint saved"
            )
        else:
            print(
                f"  [E3] Batch {batch_idx + 1}/{len(batches)} - "
                f"{n_od_done}/{n_od} OD in {elapsed:.1f}s"
            )

    return accumulator


def run_dataset(cfg: ExperimentConfig) -> Dict:
    set_seed(cfg.seed)
    cfg.output_dir = resolve_output_dir(cfg.dataset, cfg.quick)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.quick:
        cfg.n_od_pairs = min(cfg.n_od_pairs, 20)
        cfg.od_batch_size = min(cfg.od_batch_size, 10)

    ckpt = PipelineCheckpoints(cfg)
    profiler = StageProfiler() if cfg.profile else None
    done = ckpt.completed_stages()
    if done:
        print(f"  [CKPT] Resuming - completed: {', '.join(done)}")

    # --- E0 ---
    e0: Dict = {}
    mu = sigma = emb = None
    model = OTSSTEncoder(cfg.hidden_dim)
    if ckpt.is_done("E0"):
        loaded = ckpt.load_e0_artifacts(model, cfg.device)
        if loaded:
            mu, sigma, emb, e0 = loaded[0], loaded[1], loaded[2], loaded[3]
    if mu is None:
        if profiler:
            profiler.start("E0")
        data = _prepare_data(cfg)
        model = _train_encoder(data, cfg)
        mu, sigma, emb = _predict_encoder(
            model, data["x_test"], data["x_vol_test"],
            data["edge_index"], data["edge_weight"], cfg.device,
        )
        e0 = {"mae": float(mae(data["y_test"], mu)), "rmse": float(rmse(data["y_test"], mu))}
        print(f"  [E0] OTS MAE={e0['mae']:.4f} RMSE={e0['rmse']:.4f}")
        if cfg.routing_checkpoint:
            ckpt.save_e0_artifacts(model, mu, sigma, emb, e0)
        if profiler:
            profiler.end({"mae": e0["mae"]})
    else:
        data = _prepare_data(cfg)

    timesteps = list(range(0, len(mu), max(1, len(mu) // (10 if cfg.quick else 20))))

    # --- E1 ---
    if profiler:
        profiler.start("E1")
    e1 = _run_e1_imputation(cfg, mu, emb, timesteps, data["dist"], ckpt, profiler)
    if profiler:
        profiler.end({"n_sparsity_levels": len(cfg.sparsity_levels)})

    # --- E2 ---
    if ckpt.is_done("E2"):
        loaded = ckpt.load_stage("E2")
        e2 = loaded.get("e2", loaded) if loaded else {}
        offset = data["test_offset"]
        speed_test = data["speed_h"][offset : offset + len(mu)]
        regimes, acc_arr, v_avg = regime_time_series(speed_test, data["dist"], MFDConfig())
        print("  [E2] loaded from checkpoint")
    else:
        if profiler:
            profiler.start("E2")
        offset = data["test_offset"]
        speed_test = data["speed_h"][offset : offset + len(mu)]
        regimes, acc_arr, v_avg = regime_time_series(speed_test, data["dist"], MFDConfig())
        e2 = {
            "regime_fractions": {
                "free_flow": float(np.mean(regimes == 0)),
                "critical": float(np.mean(regimes == 1)),
                "congested": float(np.mean(regimes == 2)),
            }
        }
        if cfg.routing_checkpoint:
            ckpt.save_stage("E2", {"e2": e2})
        if profiler:
            profiler.end()

    # --- E3 ---
    mgac = fit_mgac_params(
        [mfd_embedding(v, a) for v, a in zip(v_avg[:20], acc_arr[:20])],
        [1.0] * 20,
        [1.0] * 20,
    )
    rcfg = RoutingConfig(use_mgac=True, mgac_params=mgac)
    methods = [
        RoutingMethod.TIME_ONLY,
        RoutingMethod.OTS_INSTANT,
        RoutingMethod.OTS_CVAR,
        RoutingMethod.SPARC_OTS,
        RoutingMethod.ORACLE,
    ]
    gkf_model = GKFFusion(emb.shape[-1])
    method_names = [m.value for m in methods]
    od_pairs = ckpt.load_od_pairs()
    if od_pairs is None:
        od_pairs = sample_od_pairs(data["dist"], cfg.n_od_pairs, seed=cfg.seed)
        ckpt.save_od_pairs(od_pairs)
    test_idx = timesteps

    if ckpt.is_done("E3"):
        acc_payload = ckpt.load_stage("E3")
        accumulator = RoutingAccumulator.from_checkpoint(acc_payload or {}, method_names)
        print("  [E3] loaded from checkpoint")
    else:
        if profiler:
            profiler.start("E3")
        if cfg.resume and cfg.routing_checkpoint:
            accumulator, start_batch, start_timestep = ckpt.load_e3_accumulator(
                method_names, timesteps_per_batch=len(test_idx)
            )
        else:
            accumulator, start_batch, start_timestep = RoutingAccumulator.init(method_names), 0, 0
        topology = resolve_routing_topology(cfg, data)
        accumulator = _run_routing_e3(
            cfg, data, mu, sigma, emb, speed_test, regimes, acc_arr, v_avg, offset,
            rcfg, gkf_model, methods, od_pairs, test_idx, accumulator, ckpt, profiler,
            start_batch=start_batch, start_timestep=start_timestep, topology=topology,
        )
        if profiler:
            n_batches = int(np.ceil(min(cfg.n_od_pairs, len(od_pairs)) / max(cfg.od_batch_size, 1)))
            profiler.end({"batches": n_batches})

    has_rii_samples = any(accumulator.rii_samples.values())
    e3 = {
        "rii": summarize_routing_results(accumulator.rii_samples, "oracle") if has_rii_samples else accumulator.to_summary_rii(),
        "travel_time": summarize_routing_results(accumulator.tt_samples, "oracle") if has_rii_samples else accumulator.to_summary_tt(),
        "risk_reduction_pct": float(
            100
            * (
                1
                - accumulator.mean_rii(RoutingMethod.SPARC_OTS.value)
                / max(accumulator.mean_rii(RoutingMethod.TIME_ONLY.value), 1e-6)
            )
        ),
        "n_od_pairs": cfg.n_od_pairs,
        "od_batch_size": cfg.od_batch_size,
        "batches_total": int(np.ceil(min(cfg.n_od_pairs, len(od_pairs)) / max(cfg.od_batch_size, 1))),
    }

    # --- E4 ---
    if ckpt.is_done("E4"):
        loaded = ckpt.load_stage("E4")
        e4 = loaded.get("e4", loaded) if loaded else {}
        print("  [E4] loaded from checkpoint")
    else:
        if profiler:
            profiler.start("E4")
        print("  [E4] Frank-Wolfe assignment")
        e4 = _run_e4_assignment(cfg, data, mu, sigma, speed_test, regimes, od_pairs, timesteps)
        if cfg.routing_checkpoint:
            ckpt.save_stage("E4", {"e4": e4})
        if profiler:
            profiler.end()

    dqi_mean = accumulator.dqi_sum / max(accumulator.dqi_count, 1)
    e5 = summarize_decision_support([dqi_mean] * max(accumulator.dqi_count, 1), [3] * max(accumulator.dqi_count, 1))
    e5["dqi_mean"] = dqi_mean
    e5["dqi_count"] = accumulator.dqi_count
    if cfg.routing_checkpoint:
        ckpt.save_stage("E5", {"e5": e5})

    results = {
        "dataset": cfg.dataset,
        "run_mode": "quick" if cfg.quick else "full",
        "data_provenance": data.get(
            "data_provenance",
            inspect_dataset(cfg.data_dir, cfg.dataset).to_provenance(),
        ),
        "E0_prediction": e0,
        "E1_imputation": e1,
        "E2_mfd": e2,
        "E3_routing": e3,
        "E4_assignment": e4,
        "E5_decision_support": e5,
    }
    _save_plots(cfg.output_dir, e1, e2, e3)
    with open(cfg.output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if profiler and profiler.records:
        profiler.write_reports(
            cfg.output_dir / "performance_report.json",
            cfg.output_dir / "performance_report.md",
        )
        agg = Path("outputs") / "performance_report.json"
        profiler.write_reports(agg, Path("outputs") / "performance_report.md")

    return results


def _save_plots(out: Path, e1, e2, e3):
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    xs, ys = [], []
    for k, v in e1.items():
        if k.startswith("sparsity_"):
            xs.append(int(k.split("_")[1]))
            ys.append(v["mae"])
    if xs:
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, "o-")
        plt.xlabel("Sparsity (%)")
        plt.ylabel("Imputation MAE")
        plt.title("E1 GKF imputation")
        plt.savefig(fig_dir / "e1_imputation.png", dpi=120)
        plt.close()
    rii = e3.get("rii", {})
    if rii:
        plt.figure(figsize=(8, 4))
        plt.bar(list(rii.keys()), [v["mean"] for v in rii.values()])
        plt.ylabel("Mean RII")
        plt.title("E3 Routing")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "e3_routing.png", dpi=120)
        plt.close()


def run_all(cfg_base: ExperimentConfig) -> Dict:
    all_results = {}
    datasets = ["metr-la"] if cfg_base.quick else ["metr-la", "pems-bay"]
    for ds in datasets:
        cfg = ExperimentConfig(
            dataset=ds,
            data_dir=Path(f"data/{ds}"),
            output_dir=resolve_output_dir(ds, cfg_base.quick),
            seed=cfg_base.seed,
            quick=cfg_base.quick,
            device=cfg_base.device,
            epochs=cfg_base.epochs,
            n_od_pairs=cfg_base.n_od_pairs,
            od_batch_size=cfg_base.od_batch_size,
            resume=cfg_base.resume,
            routing_checkpoint=cfg_base.routing_checkpoint,
            profile=cfg_base.profile,
            allow_synthetic=cfg_base.allow_synthetic,
            routing_backend=cfg_base.routing_backend,
            routing_topology=cfg_base.routing_topology,
            od_workers=cfg_base.od_workers,
        )
        all_results[ds] = run_dataset(cfg)
    summary_path = Path("outputs")
    summary_path.mkdir(parents=True, exist_ok=True)
    with open(summary_path / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    return all_results
