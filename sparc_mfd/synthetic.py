"""Synthetic traffic data for METR-LA and PEMS-BAY."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _write_h5(speed: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        grp = f.create_group("df")
        grp.create_dataset("block0_values", data=speed.astype(np.float32))


def _random_adj(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    adj = rng.random((n, n)).astype(np.float32) * 0.3
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 1.0)
    return adj


def _load_or_create_adj(data_dir: Path, n_sensors: int, seed: int) -> np.ndarray:
    adj_path = data_dir / "adj_mx.pkl"
    if adj_path.exists():
        with open(adj_path, "rb") as f:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                _, _, adj = pickle.load(f, encoding="latin1")
        adj = np.array(adj, dtype=np.float32)
        if adj.shape[0] == n_sensors:
            return adj
    adj = _random_adj(n_sensors, seed)
    with open(adj_path, "wb") as f:
        pickle.dump(("dummy", "dummy", adj), f)
    return adj


def generate_synthetic_dataset(
    data_dir: Path,
    dataset_name: str = "metr-la",
    n_days: int = 30,
    seed: int = 42,
) -> None:
    from .data_loader import DATASET_CONFIG

    cfg = DATASET_CONFIG.get(dataset_name, DATASET_CONFIG["metr-la"])
    n_sensors = cfg["n_sensors"]
    rng = np.random.default_rng(seed + hash(dataset_name) % 1000)
    adj = _load_or_create_adj(data_dir, n_sensors, seed)
    n_steps = n_days * 24 * 12

    t = np.arange(n_steps, dtype=np.float32)
    hour = (t * 5.0 / 60.0) % 24.0
    base = 62.0 - 18.0 * np.exp(-0.5 * ((hour - 8.0) / 2.5) ** 2)
    base = base - 15.0 * np.exp(-0.5 * ((hour - 17.5) / 2.0) ** 2)
    base = np.maximum(base, 25.0)
    bias = rng.normal(0, 4, n_sensors).astype(np.float32)
    noise = rng.normal(0, 3, (n_steps, n_sensors)).astype(np.float32)
    speed = np.clip(base[:, None] + bias + noise, 5.0, 75.0).astype(np.float32)

    h5_name = cfg["h5"]
    _write_h5(speed, data_dir / h5_name)
    np.save(data_dir / "speed.npy", speed)

    rows = []
    for s in range(n_sensors):
        for d in range(n_sensors):
            if s == d:
                continue
            w = float(adj[s, d])
            if w > 0:
                rows.append({"from": s, "to": d, "cost": 500.0 / max(w, 1e-3)})
    dist_name = cfg["dist"]
    pd.DataFrame(rows).to_csv(data_dir / dist_name, index=False)
    print(f"Generated synthetic {dataset_name}: {n_steps}x{n_sensors} -> {data_dir}")
