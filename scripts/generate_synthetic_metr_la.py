"""Generate METR-LA-like synthetic speed data when the real h5 is unavailable."""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _load_adj(data_dir: Path) -> np.ndarray:
    adj_path = data_dir / "adj_mx.pkl"
    with open(adj_path, "rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _, _, adj = pickle.load(f, encoding="latin1")
    return np.array(adj, dtype=np.float32)


def generate_speed_series(
    n_steps: int,
    n_sensors: int,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=np.float32)
    hour = (t * 5.0 / 60.0) % 24.0

    base = 62.0 - 18.0 * np.exp(-0.5 * ((hour - 8.0) / 2.5) ** 2)
    base = base - 15.0 * np.exp(-0.5 * ((hour - 17.5) / 2.0) ** 2)
    base = np.maximum(base, 25.0)

    sensor_bias = rng.normal(0.0, 4.0, size=n_sensors).astype(np.float32)
    noise = rng.normal(0.0, 3.0, size=(n_steps, n_sensors)).astype(np.float32)

    speed = base[:, None] + sensor_bias[None, :] + noise
    speed = np.clip(speed, 5.0, 75.0).astype(np.float32)

    # Inject localized congestion events (~7% timesteps per sensor)
    for _ in range(n_sensors * 20):
        s = rng.integers(0, n_sensors)
        start = rng.integers(0, max(1, n_steps - 12))
        dur = rng.integers(3, 12)
        speed[start : start + dur, s] *= rng.uniform(0.2, 0.5)
    return speed


def write_h5(speed: np.ndarray, out_path: Path) -> None:
    import h5py

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        grp = f.create_group("df")
        grp.create_dataset("block0_values", data=speed.astype(np.float32))


def write_distances_from_adj(adj: np.ndarray, out_path: Path) -> None:
    n = adj.shape[0]
    rows = []
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            w = float(adj[s, d])
            if w <= 0:
                continue
            rows.append({"from": s, "to": d, "cost": 500.0 / max(w, 1e-3)})
    pd.DataFrame(rows).to_csv(out_path, index=False)


def generate_synthetic_dataset(data_dir: Path, n_days: int = 30, seed: int = 42) -> None:
    adj = _load_adj(data_dir)
    n_sensors = adj.shape[0]
    n_steps = n_days * 24 * 12  # 5-min steps
    speed = generate_speed_series(n_steps, n_sensors, seed=seed)
    write_h5(speed, data_dir / "metr-la.h5")
    write_distances_from_adj(adj, data_dir / "distances_la.csv")
    print(f"Generated synthetic METR-LA: {n_steps} steps x {n_sensors} sensors -> {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--n_days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_synthetic_dataset(args.data_dir, n_days=args.n_days, seed=args.seed)


if __name__ == "__main__":
    main()
