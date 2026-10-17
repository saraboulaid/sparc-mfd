from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


DATASET_CONFIG = {
    "metr-la": {
        "h5": "metr-la.h5",
        "adj": "adj_mx.pkl",
        "dist": "distances_la.csv",
        "n_sensors": 207,
    },
    "pems-bay": {
        "h5": "pems-bay.h5",
        "adj": "adj_mx.pkl",
        "dist": "distances_bay.csv",
        "n_sensors": 325,
    },
}

# 30 days x 24 h x 12 (5-min) — signature of generate_synthetic_dataset(n_days=30)
SYNTHETIC_N_STEPS = 30 * 24 * 12
RESOLUTION_MINUTES = 5


@dataclass
class DatasetInspection:
    dataset: str
    data_dir: str
    h5_path: str
    h5_exists: bool
    adj_exists: bool
    dist_exists: bool
    shape: Optional[List[int]]
    n_timesteps: Optional[int]
    n_sensors: Optional[int]
    resolution_minutes: int
    sampling_interval_minutes: Optional[float]
    timestamp_min: Optional[str]
    timestamp_max: Optional[str]
    file_size_mb: Optional[float]
    file_hash: Optional[str]
    synthetic: bool
    data_source: str
    adj_shape: Optional[List[int]]
    adjacency_shape: Optional[List[int]]
    adj_density: Optional[float]
    adj_synthetic: Optional[bool]
    sensor_count_match: Optional[bool]
    sensor_order_alignment_ok: Optional[bool]
    sensor_order_margin: Optional[float]
    sensor_alignment_ok: Optional[bool]
    distances_source: str
    distances_synthetic: bool
    distances_rows: Optional[int]
    messages: List[str]

    def to_provenance(self) -> Dict:
        return {
            "dataset": self.dataset,
            "data_source": self.data_source,
            "synthetic": self.synthetic,
            "shape": self.shape,
            "file_size_mb": self.file_size_mb,
            "file_hash": self.file_hash,
            "resolution_minutes": self.resolution_minutes,
            "sampling_interval_minutes": self.sampling_interval_minutes,
            "timestamp_min": self.timestamp_min,
            "timestamp_max": self.timestamp_max,
            "adjacency_shape": self.adjacency_shape,
            "adj_density": self.adj_density,
            "adj_synthetic": self.adj_synthetic,
            "distances_source": self.distances_source,
            "distances_synthetic": self.distances_synthetic,
            "sensor_alignment_ok": self.sensor_alignment_ok,
            "sensor_order_alignment_ok": self.sensor_order_alignment_ok,
            "sensor_order_margin": self.sensor_order_margin,
            "h5_path": self.h5_path,
        }

    def ok_for_full(self) -> bool:
        return (
            self.h5_exists
            and self.adj_exists
            and not self.synthetic
            and self.data_source == "real_dcrnn"
            and self.sensor_count_match is True
            and self.sensor_order_alignment_ok is True
            and not self.distances_synthetic
            and self.adj_synthetic is False
            and self.n_timesteps is not None
            and self.n_timesteps > SYNTHETIC_N_STEPS
        )


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_h5_time_bounds(h5_path: Path) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[float]]:
    """Return (t_min, t_max, median_interval_minutes) from DCRNN h5 time axis."""
    if not h5_path.exists():
        return None, None, None
    try:
        group = _h5_group_name(h5_path)
        if group is None:
            return None, None, None
        with h5py.File(h5_path, "r") as f:
            g = f[group]
            if "block0_values" not in g:
                return None, None, None
            n_steps = int(g["block0_values"].shape[0])
            time_axis = None
            for axis_name in ("axis1", "axis0"):
                if axis_name not in g:
                    continue
                if len(g[axis_name]) == n_steps:
                    time_axis = g[axis_name][:]
                    break
            if time_axis is None:
                return None, None, None
        idx = pd.to_datetime(time_axis)
        if len(idx) < 2:
            return idx.min(), idx.max(), None
        deltas = pd.Series(idx[1:]) - pd.Series(idx[:-1])
        median_min = float(deltas.median().total_seconds() / 60.0)
        return idx.min(), idx.max(), median_min
    except (OSError, ValueError, TypeError):
        return None, None, None


def _infer_sampling_interval_minutes(
    n_timesteps: Optional[int],
    t_min: Optional[pd.Timestamp],
    t_max: Optional[pd.Timestamp],
    median_min: Optional[float],
) -> Optional[float]:
    if median_min is not None and median_min > 0:
        return round(median_min, 3)
    if n_timesteps and n_timesteps > 1 and t_min is not None and t_max is not None:
        span_min = (t_max - t_min).total_seconds() / 60.0
        if span_min > 0:
            return round(span_min / (n_timesteps - 1), 3)
    return float(RESOLUTION_MINUTES)


def _h5_group_name(h5_path: Path) -> Optional[str]:
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as f:
        for group in ("df", "speed"):
            if group in f and "block0_values" in f[group]:
                return group
        for key in f.keys():
            if isinstance(f[key], h5py.Group) and "block0_values" in f[key]:
                return str(key)
    return None


def _read_h5_speed(h5_path: Path) -> np.ndarray:
    group = _h5_group_name(h5_path)
    if group is None:
        df = pd.read_hdf(h5_path)
        return df.values.astype(np.float32)
    with h5py.File(h5_path, "r") as f:
        return f[group]["block0_values"][:].astype(np.float32)


def _peek_h5_shape(h5_path: Path) -> Optional[Tuple[int, int]]:
    if not h5_path.exists():
        return None
    try:
        group = _h5_group_name(h5_path)
        if group is None:
            return None
        with h5py.File(h5_path, "r") as f:
            ds = f[group]["block0_values"]
            return int(ds.shape[0]), int(ds.shape[1])
    except OSError:
        return None


def load_metr_la_speed(h5_path: Path) -> np.ndarray:
    """Load speed matrix; ignore stale speed.npy cache when h5 was replaced."""
    npy_path = h5_path.with_name("speed.npy")
    h5_shape = _peek_h5_shape(h5_path)
    if npy_path.exists() and h5_path.exists():
        if npy_path.stat().st_mtime >= h5_path.stat().st_mtime:
            cached = np.load(npy_path).astype(np.float32)
            if h5_shape is None or cached.shape == h5_shape:
                return cached
    speed = _read_h5_speed(h5_path)
    np.save(npy_path, speed)
    return speed


def read_h5_sensor_ids(h5_path: Path) -> Optional[np.ndarray]:
    """Return DCRNN sensor IDs for h5 columns (axis0), if present."""
    group = _h5_group_name(h5_path)
    if group is None:
        return None
    with h5py.File(h5_path, "r") as f:
        g = f[group]
        if "axis0" not in g or "block0_values" not in g:
            return None
        n_sensors = int(g["block0_values"].shape[1])
        if len(g["axis0"]) != n_sensors:
            return None
        raw = g["axis0"][:]
        ids = []
        for x in raw:
            if isinstance(x, (bytes, np.bytes_)):
                ids.append(int(x.decode()))
            else:
                ids.append(int(x))
        return np.array(ids, dtype=np.int64)


def _load_adj_array(adj_path: Path) -> Optional[np.ndarray]:
    if not adj_path.exists():
        return None
    import pickle
    import warnings

    with open(adj_path, "rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _, _, adj = pickle.load(f, encoding="latin1")
    return np.array(adj, dtype=np.float32)


def _inspect_distances_csv(path: Path, n_sensors: int) -> Dict:
    if not path.exists():
        return {
            "exists": False,
            "rows": 0,
            "synthetic": True,
            "source": "missing",
            "uses_sensor_ids": False,
        }
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    c_from = "from" if "from" in cols else cols[0]
    c_cost = "cost" if "cost" in cols else cols[2]
    n_rows = len(df)
    max_pairs = n_sensors * max(n_sensors - 1, 1)
    from_vals = pd.to_numeric(df[c_from], errors="coerce").dropna().astype(np.int64)
    uses_sensor_ids = bool(len(from_vals) and from_vals.max() >= n_sensors)
    cost_min = float(pd.to_numeric(df[c_cost], errors="coerce").min())

    synthetic = False
    source = "unknown"
    if uses_sensor_ids and n_rows > 1_000:
        source = "official_dcrnn"
    elif n_rows >= 0.9 * max_pairs:
        synthetic = True
        source = "synthetic_fallback"
    elif 499.0 <= cost_min <= 501.0 and n_rows < 10_000:
        synthetic = True
        source = "synthetic_fallback"
    elif n_rows > 0:
        source = "official_dcrnn"

    return {
        "exists": True,
        "rows": n_rows,
        "synthetic": synthetic,
        "source": source,
        "uses_sensor_ids": uses_sensor_ids,
    }


def _read_h5_speed_slice(h5_path: Path, n_time: int) -> np.ndarray:
    group = _h5_group_name(h5_path)
    if group is None:
        speed = _read_h5_speed(h5_path)
        return speed[:n_time]
    with h5py.File(h5_path, "r") as f:
        return f[group]["block0_values"][:n_time].astype(np.float32)


def verify_sensor_order_alignment(
    h5_path: Path,
    adj_path: Path,
    *,
    adj_threshold: float = 0.1,
    n_time: int = 2000,
    n_edges: int = 300,
    n_random: int = 300,
    min_margin: float = 0.03,
) -> Tuple[bool, float, float, float]:
    """
    Check that h5 column i aligns with adj row/col i by comparing speed correlation
    on graph edges vs random non-edges.
    """
    speed = _read_h5_speed_slice(h5_path, n_time)
    adj = _load_adj_array(adj_path)
    if adj is None:
        return False, 0.0, 0.0, 0.0
    n = adj.shape[0]
    src, dst = np.nonzero(adj > adj_threshold)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    rng = np.random.default_rng(42)
    if len(src) > n_edges:
        pick = rng.choice(len(src), n_edges, replace=False)
        src, dst = src[pick], dst[pick]

    def _corr(i: int, j: int) -> float:
        a, b = speed[:, i], speed[:, j]
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    edge_mean = float(np.mean([_corr(int(s), int(d)) for s, d in zip(src, dst)]))
    random_pairs: List[Tuple[int, int]] = []
    attempts = 0
    while len(random_pairs) < n_random and attempts < n_random * 30:
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        attempts += 1
        if i == j or adj[i, j] > adj_threshold:
            continue
        random_pairs.append((i, j))
    rand_mean = float(np.mean([_corr(i, j) for i, j in random_pairs])) if random_pairs else 0.0
    margin = edge_mean - rand_mean
    return margin >= min_margin, edge_mean, rand_mean, margin


def load_direct_distances_csv(
    path: Path,
    n_sensors: int,
    sensor_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Load pairwise road distances (m) without Floyd-Warshall (for E3 sparse routing)."""
    if not path.exists():
        return np.full((n_sensors, n_sensors), np.inf, dtype=np.float32)
    df = pd.read_csv(path)
    dist = np.full((n_sensors, n_sensors), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    cols = df.columns.tolist()
    c_from = "from" if "from" in cols else cols[0]
    c_to = "to" if "to" in cols else cols[1]
    c_cost = "cost" if "cost" in cols else cols[2]
    id_to_idx = None
    if sensor_ids is not None and len(sensor_ids) == n_sensors:
        id_to_idx = {int(sid): i for i, sid in enumerate(sensor_ids)}
    for _, row in df.iterrows():
        s_raw, d_raw = int(row[c_from]), int(row[c_to])
        if id_to_idx is not None:
            if s_raw not in id_to_idx or d_raw not in id_to_idx:
                continue
            s, d = id_to_idx[s_raw], id_to_idx[d_raw]
        else:
            s, d = s_raw, d_raw
        if 0 <= s < n_sensors and 0 <= d < n_sensors:
            dist[s, d] = float(row[c_cost])
    return dist.astype(np.float32)


def load_distances_csv(
    path: Path,
    n_sensors: int,
    sensor_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """All-pairs shortest paths (Floyd-Warshall) for kriging / MFD."""
    dist = load_direct_distances_csv(path, n_sensors, sensor_ids=sensor_ids).astype(np.float64)
    for k in range(n_sensors):
        dist = np.minimum(dist, dist[:, k][:, None] + dist[k, :][None, :])
    dist[~np.isfinite(dist)] = 1e6
    return dist.astype(np.float32)


def _load_adj_shape(adj_path: Path) -> Optional[Tuple[int, int]]:
    if not adj_path.exists():
        return None
    import pickle
    import warnings

    with open(adj_path, "rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _, _, adj = pickle.load(f, encoding="latin1")
    adj = np.array(adj)
    return int(adj.shape[0]), int(adj.shape[1])


def inspect_dataset(data_dir: Path, name: str) -> DatasetInspection:
    cfg = DATASET_CONFIG[name]
    data_dir = Path(data_dir)
    h5_path = data_dir / cfg["h5"]
    adj_path = data_dir / cfg["adj"]
    dist_path = data_dir / cfg["dist"]
    expected_n = cfg["n_sensors"]
    messages: List[str] = []

    h5_exists = h5_path.exists()
    adj_exists = adj_path.exists()
    dist_exists = dist_path.exists()
    shape = None
    n_timesteps = None
    n_sensors = None
    file_size_mb = round(h5_path.stat().st_size / (1024 * 1024), 2) if h5_exists else None
    adj_shape = None
    adjacency_shape = None
    sensor_count_match = None
    sensor_alignment_ok = None
    synthetic = True
    data_source = "missing"
    timestamp_min = None
    timestamp_max = None
    sampling_interval_minutes = None
    file_hash = _file_sha256(h5_path) if h5_exists else None

    if not h5_exists:
        messages.append(f"Missing h5: {h5_path}")
    else:
        shape_tuple = _peek_h5_shape(h5_path)
        if shape_tuple is None:
            messages.append(f"Cannot read shape from {h5_path}")
        else:
            n_timesteps, n_sensors = shape_tuple
            shape = [n_timesteps, n_sensors]
            if n_sensors != expected_n:
                messages.append(
                    f"Expected {expected_n} sensors, h5 has {n_sensors}"
                )
            if n_timesteps == SYNTHETIC_N_STEPS:
                messages.append(
                    f"Timesteps={n_timesteps} matches synthetic signature ({SYNTHETIC_N_STEPS})"
                )
                synthetic = True
                data_source = "synthetic_fallback"
            elif n_timesteps > SYNTHETIC_N_STEPS:
                synthetic = False
                data_source = "real_dcrnn"
            else:
                messages.append(f"Unexpected timestep count: {n_timesteps}")
                synthetic = True
                data_source = "unknown"

        npy_path = h5_path.with_name("speed.npy")
        if npy_path.exists() and shape is not None:
            npy_shape = tuple(int(x) for x in np.load(npy_path, mmap_mode="r").shape)
            if npy_shape != tuple(shape):
                messages.append(
                    f"Stale speed.npy shape {npy_shape} != h5 shape {tuple(shape)}; h5 will be used"
                )

        t_min, t_max, median_interval = _read_h5_time_bounds(h5_path)
        if t_min is not None and t_max is not None:
            timestamp_min = t_min.isoformat()
            timestamp_max = t_max.isoformat()
        else:
            messages.append("No datetime index in h5 (axis0 missing); timestamps unavailable")
        sampling_interval_minutes = _infer_sampling_interval_minutes(
            n_timesteps, t_min, t_max, median_interval
        )
        if sampling_interval_minutes != RESOLUTION_MINUTES:
            messages.append(
                f"Sampling interval {sampling_interval_minutes} min != expected {RESOLUTION_MINUTES} min"
            )

    adj_density = None
    adj_synthetic = None
    sensor_order_alignment_ok = None
    sensor_order_margin = None
    dist_meta = _inspect_distances_csv(dist_path, expected_n) if dist_exists else {
        "exists": False, "rows": 0, "synthetic": True, "source": "missing", "uses_sensor_ids": False
    }
    distances_source = str(dist_meta["source"])
    distances_synthetic = bool(dist_meta["synthetic"])
    distances_rows = int(dist_meta["rows"]) if dist_meta["exists"] else None

    if adj_exists:
        adj_arr = _load_adj_array(adj_path)
        if adj_arr is not None:
            adj_shape_tuple = adj_arr.shape
            adj_shape = [int(adj_shape_tuple[0]), int(adj_shape_tuple[1])]
            adjacency_shape = adj_shape
            nnz = int(np.count_nonzero(adj_arr))
            adj_density = round(nnz / adj_arr.size, 6)
            adj_synthetic = adj_density > 0.5
            if adj_synthetic:
                messages.append(
                    f"Adjacency density {adj_density:.3f} suggests synthetic fallback (expected sparse road graph)"
                )
            if n_sensors is not None:
                sensor_count_match = adj_shape_tuple[0] == n_sensors == adj_shape_tuple[1]
                if not sensor_count_match:
                    messages.append(
                        f"Adj shape {adj_shape} incompatible with h5 sensors ({n_sensors})"
                    )
            elif adj_shape_tuple[0] == expected_n:
                sensor_count_match = True
    else:
        messages.append(f"Missing adjacency: {adj_path}")

    if h5_exists and adj_exists and not synthetic:
        order_ok, edge_corr, rand_corr, margin = verify_sensor_order_alignment(h5_path, adj_path)
        sensor_order_alignment_ok = order_ok
        sensor_order_margin = round(margin, 4)
        if not order_ok:
            messages.append(
                f"Sensor order check failed: edge_corr={edge_corr:.3f} random_corr={rand_corr:.3f} margin={margin:.3f}"
            )

    sensor_alignment_ok = (
        sensor_count_match is True
        and (sensor_order_alignment_ok is not False)
    )

    if not dist_exists:
        messages.append(f"Missing distances CSV: {dist_path}")
    elif distances_synthetic:
        messages.append(
            f"Distances file looks synthetic ({distances_rows} rows, source={distances_source})"
        )
    elif dist_meta.get("uses_sensor_ids"):
        messages.append("Distances use DCRNN sensor IDs; index mapping via h5 axis0 is required")

    return DatasetInspection(
        dataset=name,
        data_dir=str(data_dir),
        h5_path=str(h5_path),
        h5_exists=h5_exists,
        adj_exists=adj_exists,
        dist_exists=dist_exists,
        shape=shape,
        n_timesteps=n_timesteps,
        n_sensors=n_sensors,
        resolution_minutes=RESOLUTION_MINUTES,
        sampling_interval_minutes=sampling_interval_minutes,
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
        file_size_mb=file_size_mb,
        file_hash=file_hash,
        synthetic=synthetic,
        data_source=data_source,
        adj_shape=adj_shape,
        adjacency_shape=adjacency_shape,
        adj_density=adj_density,
        adj_synthetic=adj_synthetic,
        sensor_count_match=sensor_count_match,
        sensor_order_alignment_ok=sensor_order_alignment_ok,
        sensor_order_margin=sensor_order_margin,
        sensor_alignment_ok=sensor_alignment_ok,
        distances_source=distances_source,
        distances_synthetic=distances_synthetic,
        distances_rows=distances_rows,
        messages=messages,
    )


def validate_dataset_for_full(data_dir: Path, name: str) -> DatasetInspection:
    report = inspect_dataset(data_dir, name)
    if not report.ok_for_full():
        lines = [
            f"Dataset '{name}' is not valid for FULL mode.",
            f"  data_source={report.data_source!r} synthetic={report.synthetic}",
            f"  shape={report.shape} adj_shape={report.adj_shape}",
        ]
        lines.extend(f"  - {m}" for m in report.messages)
        lines.append(
            "Place real DCRNN files (see scripts/download_dcrnn_data.py) "
            "and rename synthetic backups before re-running."
        )
        raise FileNotFoundError("\n".join(lines))
    return report


def ensure_dataset(
    data_dir: Path,
    name: str,
    *,
    allow_synthetic: bool = True,
) -> Tuple[Path, Path, Path]:
    cfg = DATASET_CONFIG[name]
    data_dir.mkdir(parents=True, exist_ok=True)
    h5_path = data_dir / cfg["h5"]
    adj_path = data_dir / cfg["adj"]
    dist_path = data_dir / cfg["dist"]

    if not h5_path.exists():
        if not allow_synthetic:
            raise FileNotFoundError(
                f"Real dataset missing: {h5_path}. "
                "Synthetic fallback is forbidden in FULL mode."
            )
        from .synthetic import generate_synthetic_dataset

        generate_synthetic_dataset(data_dir, dataset_name=name, n_days=30)

    if not adj_path.exists():
        if not allow_synthetic:
            raise FileNotFoundError(
                f"Real adjacency missing: {adj_path}. "
                "Synthetic fallback is forbidden in FULL mode."
            )
        if name == "pems-bay":
            from .synthetic import generate_synthetic_dataset

            generate_synthetic_dataset(data_dir, dataset_name=name, n_days=30)
        else:
            raise FileNotFoundError(f"Missing {adj_path}")

    if not allow_synthetic:
        validate_dataset_for_full(data_dir, name)

    return h5_path, adj_path, dist_path


def format_validation_report(reports: List[DatasetInspection]) -> str:
    lines = ["# Dataset validation report", ""]
    for r in reports:
        lines.append(f"## {r.dataset}")
        lines.append(f"- h5: `{r.h5_path}`")
        lines.append(f"- shape: {r.shape}")
        lines.append(f"- file_size_mb: {r.file_size_mb}")
        lines.append(f"- file_hash: `{r.file_hash}`")
        lines.append(f"- timestamp_min: {r.timestamp_min}")
        lines.append(f"- timestamp_max: {r.timestamp_max}")
        lines.append(f"- sampling_interval_minutes: {r.sampling_interval_minutes}")
        lines.append(f"- adjacency_shape: {r.adjacency_shape}")
        lines.append(f"- adj_density: {r.adj_density}")
        lines.append(f"- adj_synthetic: **{r.adj_synthetic}**")
        lines.append(f"- distances_rows: {r.distances_rows}")
        lines.append(f"- distances_source: **{r.distances_source}**")
        lines.append(f"- distances_synthetic: **{r.distances_synthetic}**")
        lines.append(f"- sensor_order_alignment_ok: **{r.sensor_order_alignment_ok}**")
        lines.append(f"- sensor_order_margin: {r.sensor_order_margin}")
        lines.append(f"- sensor_alignment_ok: **{r.sensor_alignment_ok}**")
        lines.append(f"- data_source: **{r.data_source}**")
        lines.append(f"- synthetic: **{r.synthetic}**")
        lines.append(f"- ok_for_full: **{r.ok_for_full()}**")
        if r.messages:
            lines.append("- notes:")
            for m in r.messages:
                lines.append(f"  - {m}")
        lines.append("")
    return "\n".join(lines)
