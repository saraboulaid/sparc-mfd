"""Graph-based sparse sensor imputation with variogram kriging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ImputeConfig:
    variogram_model: str = "spherical"
    nugget: float = 0.01
    sill: float = 1.0
    range_m: float = 3000.0
    blend_gnn: float = 0.5


@dataclass(frozen=True)
class KrigingCacheKey:
    obs_idx: Tuple[int, ...]
    range_m: float
    nugget: float
    sill: float


@dataclass
class _FactorizationEntry:
    weights: np.ndarray
    var_unobs: np.ndarray
    obs_idx: np.ndarray
    unobs_idx: np.ndarray


class KrigingSolver:
    """Cached kriging system — solve(Gamma, y) instead of inv(Gamma) @ y."""

    def __init__(self, dist_matrix: np.ndarray, cfg: Optional[ImputeConfig] = None):
        self.dist = dist_matrix
        self.cfg = cfg or ImputeConfig()
        self._cache: Dict[KrigingCacheKey, _FactorizationEntry] = {}

    def _invert_gamma(self, gamma_mat: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.inv(gamma_mat)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(gamma_mat)

    def _solve_gamma(self, gamma_mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(gamma_mat, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(gamma_mat, rhs, rcond=None)[0]

    def _cache_key(self, obs_idx: np.ndarray, range_m: float) -> KrigingCacheKey:
        return KrigingCacheKey(
            tuple(int(i) for i in obs_idx),
            float(range_m),
            float(self.cfg.nugget),
            float(self.cfg.sill),
        )

    def _get_factorization(
        self, obs_idx: np.ndarray, unobs_idx: np.ndarray, range_m: float
    ) -> _FactorizationEntry:
        key = self._cache_key(obs_idx, range_m)
        if key in self._cache:
            return self._cache[key]

        gamma_mat = spherical_variogram(
            self.dist[np.ix_(obs_idx, obs_idx)], self.cfg.nugget, self.cfg.sill, range_m
        )
        gamma_mat = gamma_mat + np.eye(len(obs_idx), dtype=np.float64) * 1e-6

        if len(unobs_idx) == 0:
            entry = _FactorizationEntry(np.empty((0, len(obs_idx))), np.empty(0), obs_idx, unobs_idx)
            self._cache[key] = entry
            return entry

        gamma_cross = spherical_variogram(
            self.dist[np.ix_(unobs_idx, obs_idx)], self.cfg.nugget, self.cfg.sill, range_m
        )
        gamma_inv = self._invert_gamma(gamma_mat)
        weights = gamma_cross @ gamma_inv
        var_unobs = np.array(
            [max(self.cfg.sill - float(weights[j] @ gamma_cross[j]), 0.0) for j in range(len(unobs_idx))],
            dtype=np.float64,
        )
        entry = _FactorizationEntry(weights, var_unobs, obs_idx, unobs_idx)
        self._cache[key] = entry
        return entry

    def clear_cache(self) -> None:
        self._cache.clear()

    def solve(
        self,
        y_observed: np.ndarray,
        observed_mask: np.ndarray,
        gnn_prior: Optional[np.ndarray] = None,
        range_m: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y_observed, dtype=np.float64).copy()
        n = len(y)
        var = np.full(n, self.cfg.sill, dtype=np.float64)
        obs_idx = np.where(observed_mask)[0]
        unobs_idx = np.where(~observed_mask)[0]

        if len(obs_idx) == 0:
            fill = float(np.nanmean(y)) if np.isfinite(y).any() else 0.0
            return np.full(n, fill, dtype=np.float32), var.astype(np.float32)

        y_obs = y[obs_idx]
        if range_m is None:
            range_m = fit_range_from_distances(self.dist, y_obs)

        entry = self._get_factorization(obs_idx, unobs_idx, range_m)
        y[obs_idx] = y_observed[obs_idx].astype(np.float64)
        var[obs_idx] = 0.0

        if len(unobs_idx) > 0:
            krig_vals = entry.weights @ y_obs
            for j, idx in enumerate(unobs_idx):
                krig = float(krig_vals[j])
                var[idx] = entry.var_unobs[j]
                if gnn_prior is not None:
                    y[idx] = self.cfg.blend_gnn * float(gnn_prior[idx]) + (1.0 - self.cfg.blend_gnn) * krig
                else:
                    y[idx] = krig

        return y.astype(np.float32), var.astype(np.float32)

    def solve_batch(
        self,
        y_batch: np.ndarray,
        observed_mask: np.ndarray,
        gnn_prior_batch: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized kriging for T timesteps sharing the same observed_mask."""
        y_batch = np.asarray(y_batch, dtype=np.float64)
        if y_batch.ndim == 1:
            y_imp, v_imp = self.solve(y_batch, observed_mask, gnn_prior_batch)
            return y_imp[None, :], v_imp[None, :]

        t_steps, n = y_batch.shape
        out = y_batch.copy()
        var_out = np.full((t_steps, n), self.cfg.sill, dtype=np.float64)
        obs_idx = np.where(observed_mask)[0]
        unobs_idx = np.where(~observed_mask)[0]

        if len(obs_idx) == 0:
            fill = float(np.nanmean(y_batch)) if np.isfinite(y_batch).any() else 0.0
            return np.full_like(y_batch, fill, dtype=np.float32), var_out.astype(np.float32)

        out[:, obs_idx] = y_batch[:, obs_idx]
        var_out[:, obs_idx] = 0.0

        if len(unobs_idx) == 0:
            return out.astype(np.float32), var_out.astype(np.float32)

        y_obs_all = y_batch[:, obs_idx]
        range_groups: Dict[float, List[int]] = {}
        for t in range(t_steps):
            rm = fit_range_from_distances(self.dist, y_obs_all[t])
            range_groups.setdefault(rm, []).append(t)

        for range_m, ts in range_groups.items():
            entry = self._get_factorization(obs_idx, unobs_idx, range_m)
            y_sub = y_obs_all[ts].T
            krig = entry.weights @ y_sub
            for k, t in enumerate(ts):
                for j, idx in enumerate(unobs_idx):
                    val = float(krig[j, k])
                    if gnn_prior_batch is not None:
                        out[t, idx] = (
                            self.cfg.blend_gnn * float(gnn_prior_batch[t, idx])
                            + (1.0 - self.cfg.blend_gnn) * val
                        )
                    else:
                        out[t, idx] = val
                    var_out[t, idx] = entry.var_unobs[j]

        return out.astype(np.float32), var_out.astype(np.float32)


def spherical_variogram(h: np.ndarray, nugget: float, sill: float, range_m: float) -> np.ndarray:
    hr = np.clip(h / max(range_m, 1e-6), 0.0, 1.0)
    return nugget + (sill - nugget) * (1.5 * hr - 0.5 * hr**3)


def fit_range_from_distances(dist_matrix: np.ndarray, sample_y: np.ndarray) -> float:
    """Estimate variogram range from observed sensor semivariance."""
    idx = np.where(sample_y > 0)[0]
    if len(idx) < 5:
        return 3000.0
    sub = idx[: min(40, len(idx))]
    d = dist_matrix[np.ix_(sub, sub)].reshape(-1)
    y_sub = sample_y[sub]
    gamma = 0.5 * (y_sub[:, None] - y_sub[None, :]) ** 2
    gamma = gamma.reshape(-1)
    mask = d > 0
    if mask.sum() < 10:
        return 3000.0
    d, gamma = d[mask], gamma[mask]
    bins = np.linspace(0, np.percentile(d, 95), 12)
    bin_mid, bin_gamma = [], []
    for i in range(len(bins) - 1):
        m = (d >= bins[i]) & (d < bins[i + 1])
        if m.sum() > 0:
            bin_mid.append(0.5 * (bins[i] + bins[i + 1]))
            bin_gamma.append(float(np.mean(gamma[m])))
    if not bin_mid:
        return 3000.0
    target = 0.95 * float(np.var(sample_y[sub]))
    for bm, bg in zip(bin_mid, bin_gamma):
        if bg >= target:
            return float(max(bm, 500.0))
    return float(max(bin_mid[-1], 3000.0))


def _graph_kriging_impute_legacy(
    y_observed: np.ndarray,
    observed_mask: np.ndarray,
    dist_matrix: np.ndarray,
    cfg: Optional[ImputeConfig] = None,
    gnn_prior: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Original inv-based implementation — kept for regression tests only."""
    cfg = cfg or ImputeConfig()
    n = len(y_observed)
    y = np.array(y_observed, dtype=np.float64, copy=True)
    var = np.full(n, cfg.sill, dtype=np.float64)

    obs_idx = np.where(observed_mask)[0]
    unobs_idx = np.where(~observed_mask)[0]
    if len(obs_idx) == 0:
        fill = float(np.nanmean(y)) if np.isfinite(y).any() else 0.0
        return np.full(n, fill, dtype=np.float32), var.astype(np.float32)

    y_obs = y[obs_idx].astype(np.float64)
    range_m = fit_range_from_distances(dist_matrix, y_obs)

    gamma_mat = spherical_variogram(
        dist_matrix[np.ix_(obs_idx, obs_idx)], cfg.nugget, cfg.sill, range_m
    )
    gamma_mat = gamma_mat + np.eye(len(obs_idx)) * 1e-6
    try:
        gamma_inv = np.linalg.inv(gamma_mat)
    except np.linalg.LinAlgError:
        gamma_inv = np.linalg.pinv(gamma_mat)

    for i in obs_idx:
        y[i] = float(y_observed[i])
        var[i] = 0.0

    if len(unobs_idx) > 0:
        gamma_cross = spherical_variogram(
            dist_matrix[np.ix_(unobs_idx, obs_idx)], cfg.nugget, cfg.sill, range_m
        )
        weights = gamma_cross @ gamma_inv
        krig_vals = weights @ y_obs
        for j, i in enumerate(unobs_idx):
            krig = float(krig_vals[j])
            var[i] = float(max(cfg.sill - float(weights[j] @ gamma_cross[j]), 0.0))
            if gnn_prior is not None:
                y[i] = cfg.blend_gnn * float(gnn_prior[i]) + (1.0 - cfg.blend_gnn) * krig
            else:
                y[i] = krig

    return y.astype(np.float32), var.astype(np.float32)


def graph_kriging_impute(
    y_observed: np.ndarray,
    observed_mask: np.ndarray,
    dist_matrix: np.ndarray,
    cfg: Optional[ImputeConfig] = None,
    gnn_prior: Optional[np.ndarray] = None,
    solver: Optional[KrigingSolver] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Impute risk field on all sensors.

    y_observed: (N,) values at all nodes (NaN or 0 at unobserved)
    observed_mask: (N,) bool
    Returns: y_imputed (N,), variance (N,)
    """
    cfg = cfg or ImputeConfig()
    if solver is None:
        krig_solver = KrigingSolver(dist_matrix, cfg)
    else:
        krig_solver = solver
    return krig_solver.solve(y_observed, observed_mask, gnn_prior)


def mask_sensors(
    n_sensors: int,
    sparsity: float,
    seed: int,
    force_observed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return observed_mask: True = sensor has data."""
    rng = np.random.default_rng(seed)
    n_obs = max(2, int(round(n_sensors * (1.0 - sparsity))))
    mask = np.zeros(n_sensors, dtype=bool)
    chosen = rng.choice(n_sensors, size=n_obs, replace=False)
    mask[chosen] = True
    if force_observed is not None:
        mask[force_observed] = True
    return mask


def impute_timestep_batch(
    y_true: np.ndarray,
    observed_mask: np.ndarray,
    dist_matrix: np.ndarray,
    gnn_pred: Optional[np.ndarray] = None,
    cfg: Optional[ImputeConfig] = None,
    solver: Optional[KrigingSolver] = None,
) -> np.ndarray:
    """Impute one timestep (N,) or (T,N) along last axis."""
    cfg = cfg or ImputeConfig()
    krig_solver = solver or KrigingSolver(dist_matrix, cfg)

    if y_true.ndim == 1:
        y_obs = np.where(observed_mask, y_true, np.nan)
        y_obs = np.nan_to_num(y_obs, nan=0.0)
        y_imp, _ = krig_solver.solve(y_obs, observed_mask, gnn_pred)
        return y_imp

    y_obs = np.where(observed_mask, y_true, 0.0)
    y_imp, _ = krig_solver.solve_batch(y_obs, observed_mask, gnn_pred)
    return y_imp
