"""Operational risk field Y from speed-only observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class RiskFieldConfig:
    low_speed: float = 20.0
    mid_speed: float = 40.0
    delta_high: float = 15.0
    delta_mid: float = 8.0
    persistence_window: int = 3
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.4


def build_proxy_components(
    speed: np.ndarray,
    cfg: Optional[RiskFieldConfig] = None,
) -> Dict[str, np.ndarray]:
    cfg = cfg or RiskFieldConfig()
    delta = np.abs(np.diff(speed, axis=0))
    delta = np.vstack([delta[0:1], delta])

    rc = np.where(
        speed < cfg.low_speed,
        1.0,
        np.where(speed < cfg.mid_speed, 0.3, 0.0),
    ).astype(np.float32)
    rv = np.where(
        delta > cfg.delta_high,
        0.5,
        np.where(delta > cfg.delta_mid, 0.2, 0.0),
    ).astype(np.float32)

    severe = (speed < cfg.low_speed).astype(np.float32)
    kernel = np.ones(cfg.persistence_window, dtype=np.float32) / float(cfg.persistence_window)
    rp = np.stack(
        [np.convolve(severe[:, i], kernel, mode="same") for i in range(severe.shape[1])],
        axis=1,
    ).astype(np.float32)

    y = (cfg.alpha * rc + cfg.beta * rv + cfg.gamma * rp).astype(np.float32)
    return {"risk_congestion": rc, "risk_variation": rv, "persistence": rp, "y": y}


def aggregate_hourly(data: np.ndarray, steps_per_hour: int = 12) -> np.ndarray:
    t = data.shape[0] // steps_per_hour * steps_per_hour
    trimmed = data[:t]
    return trimmed.reshape(-1, steps_per_hour, *trimmed.shape[1:]).mean(axis=1).astype(np.float32)


def build_temporal_features(speed_h: np.ndarray, start: str = "2012-03-01") -> np.ndarray:
    import pandas as pd

    t, n = speed_h.shape
    ts = pd.date_range(start, periods=t, freq="1h")
    hour = ts.hour.values
    day = ts.dayofweek.values
    feats = [
        speed_h,
        np.tile(np.sin(2 * np.pi * hour / 24), (n, 1)).T,
        np.tile(np.cos(2 * np.pi * hour / 24), (n, 1)).T,
        np.tile(np.sin(2 * np.pi * day / 7), (n, 1)).T,
        np.tile(np.cos(2 * np.pi * day / 7), (n, 1)).T,
    ]
    return np.stack(feats, axis=-1).astype(np.float32)


def sliding_windows(x: np.ndarray, y: np.ndarray, seq_len: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(x) - seq_len):
        xs.append(x[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def risk_statistics(y: np.ndarray) -> Dict[str, float]:
    return {
        "zero_rate": float(np.mean(y == 0)),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "p95": float(np.percentile(y, 95)),
    }
