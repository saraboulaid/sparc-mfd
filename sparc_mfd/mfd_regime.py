"""Macroscopic Fundamental Diagram regime detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Tuple

import numpy as np


class MFDRegime(IntEnum):
    FREE_FLOW = 0
    CRITICAL = 1
    CONGESTED = 2


@dataclass
class MFDConfig:
    critical_speed: float = 45.0
    congested_speed: float = 30.0
    accumulation_scale: float = 1.0


@dataclass
class RegimeWeights:
    omega: float  # E[Y] weight
    kappa: float  # CVaR weight


DEFAULT_WEIGHTS = {
    MFDRegime.FREE_FLOW: RegimeWeights(omega=0.3, kappa=0.1),
    MFDRegime.CRITICAL: RegimeWeights(omega=0.8, kappa=0.5),
    MFDRegime.CONGESTED: RegimeWeights(omega=1.5, kappa=1.2),
}


def network_accumulation(speed: np.ndarray, dist_matrix: np.ndarray) -> float:
    """Proxy accumulation: inverse speed weighted by connectivity."""
    v = np.clip(speed, 5.0, None)
    degree = np.sum(dist_matrix < np.inf, axis=1) - 1
    degree = np.maximum(degree, 1).astype(np.float32)
    n = np.mean(1.0 / v * np.sqrt(degree))
    return float(n)


def network_avg_speed(speed: np.ndarray) -> float:
    return float(np.mean(np.clip(speed, 0.0, None)))


def classify_regime(avg_speed: float, cfg: MFDConfig) -> MFDRegime:
    if avg_speed >= cfg.critical_speed:
        return MFDRegime.FREE_FLOW
    if avg_speed >= cfg.congested_speed:
        return MFDRegime.CRITICAL
    return MFDRegime.CONGESTED


def fit_bathtub_curve(
    speeds: np.ndarray,
    accumulations: np.ndarray,
) -> Dict[str, float]:
    """Fit simple linear MFD segments for visualization."""
    order = np.argsort(accumulations)
    acc = accumulations[order]
    spd = speeds[order]
    n_bins = min(20, max(5, len(acc) // 50))
    if n_bins < 2:
        return {"v_free": float(np.max(spd)), "acc_crit": float(np.median(acc))}

    bins = np.linspace(acc.min(), acc.max(), n_bins + 1)
    v_bin, a_bin = [], []
    for i in range(n_bins):
        m = (acc >= bins[i]) & (acc < bins[i + 1])
        if m.sum() > 0:
            v_bin.append(float(np.mean(spd[m])))
            a_bin.append(float(np.mean(acc[m])))

    if len(v_bin) < 2:
        return {"v_free": float(np.max(spd)), "acc_crit": float(np.median(acc))}

    v_bin = np.array(v_bin)
    a_bin = np.array(a_bin)
    drop = np.diff(v_bin)
    crit_idx = int(np.argmin(drop)) + 1 if len(drop) > 0 else len(v_bin) // 2
    return {
        "v_free": float(v_bin[0]),
        "v_cong": float(v_bin[-1]),
        "acc_crit": float(a_bin[min(crit_idx, len(a_bin) - 1)]),
        "accumulation_mean": float(np.mean(accumulations)),
        "speed_mean": float(np.mean(speeds)),
    }


def regime_time_series(
    speed_ts: np.ndarray,
    dist_matrix: np.ndarray,
    cfg: MFDConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify regime per timestep. speed_ts: (T, N)."""
    t = speed_ts.shape[0]
    regimes = np.zeros(t, dtype=np.int32)
    acc = np.zeros(t, dtype=np.float32)
    v_avg = np.zeros(t, dtype=np.float32)
    for i in range(t):
        v_avg[i] = network_avg_speed(speed_ts[i])
        acc[i] = network_accumulation(speed_ts[i], dist_matrix)
        regimes[i] = int(classify_regime(v_avg[i], cfg))
    return regimes, acc, v_avg


def get_regime_weights(regime: MFDRegime) -> RegimeWeights:
    return DEFAULT_WEIGHTS[MFDRegime(regime)]
