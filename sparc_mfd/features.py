"""Feature engineering for OTS pipeline."""

from __future__ import annotations

from typing import Tuple

import numpy as np


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


def _moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float32) / float(window)
    out = np.stack([np.convolve(x[:, i], kernel, mode="same") for i in range(x.shape[1])], axis=1)
    return out.astype(np.float32)


def build_dual_branch_features(speed_h: np.ndarray, window: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    trend = _moving_avg(speed_h, window)
    delta = np.abs(np.diff(speed_h, axis=0, prepend=speed_h[:1]))
    vol = _moving_avg(delta, window)
    vol_feat = np.stack([vol, delta], axis=-1).astype(np.float32)
    return trend, vol_feat


def sliding_windows(x: np.ndarray, y: np.ndarray, seq_len: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(x) - seq_len):
        xs.append(x[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def normalize_train_only(x: np.ndarray, train_ratio: float = 0.7):
    t_train = int(x.shape[0] * train_ratio)
    mean = x[:t_train].mean(axis=(0, 1), keepdims=True)
    std = x[:t_train].std(axis=(0, 1), keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)
