"""Experiment metrics and statistical tests."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def regret(values: np.ndarray, oracle: np.ndarray) -> float:
    """Mean relative regret vs oracle (lower is better)."""
    denom = np.maximum(np.abs(oracle), 1e-6)
    return float(np.mean((values - oracle) / denom))


#: Nominal two-sided coverage levels reported by :func:`calibration_report`.
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)

#: Standard normal quantiles z_{(1+p)/2} for the levels above.
_Z_SCORES = {0.50: 0.6745, 0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}


def picp(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, level: float = 0.90) -> float:
    """Prediction Interval Coverage Probability of the Gaussian interval at `level`."""
    z = _Z_SCORES[level]
    half = z * np.asarray(sigma, dtype=np.float64)
    inside = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) <= half
    return float(np.mean(inside))


def mpiw(sigma: np.ndarray, level: float = 0.90) -> float:
    """Mean Prediction Interval Width, in OTS units."""
    return float(np.mean(2.0 * _Z_SCORES[level] * np.asarray(sigma, dtype=np.float64)))


def interval_score(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, level: float = 0.90
) -> float:
    """Winkler interval score (lower is better): width plus asymmetric miss penalty."""
    alpha = 1.0 - level
    z = _Z_SCORES[level]
    y = np.asarray(y_true, dtype=np.float64)
    half = z * np.asarray(sigma, dtype=np.float64)
    lo, hi = np.asarray(mu, dtype=np.float64) - half, np.asarray(mu, dtype=np.float64) + half
    penalty = (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))
    return float(np.mean((hi - lo) + penalty))


def nll_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Mean Gaussian negative log-likelihood, the proper score behind the head."""
    s = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)
    resid = np.asarray(y_true, dtype=np.float64) - np.asarray(mu, dtype=np.float64)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * s**2) + 0.5 * (resid / s) ** 2))


def calibration_report(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> Dict:
    """Quantitative calibration of the heteroscedastic head.

    Reports empirical coverage against nominal coverage so the manuscript can
    state whether the intervals are calibrated instead of asserting it. The
    reliability gap is empirical minus nominal: positive means over-coverage
    (intervals too wide), negative means over-confidence.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    m = np.asarray(mu, dtype=np.float64).ravel()
    s = np.maximum(np.asarray(sigma, dtype=np.float64).ravel(), 1e-12)
    levels = []
    for level in COVERAGE_LEVELS:
        empirical = picp(y, m, s, level)
        levels.append(
            {
                "nominal": float(level),
                "picp": empirical,
                "gap": float(empirical - level),
                "mpiw": mpiw(s, level),
                "interval_score": interval_score(y, m, s, level),
            }
        )
    # Mean absolute reliability gap: a single number for the abstract.
    ace = float(np.mean([abs(entry["gap"]) for entry in levels]))
    return {
        "levels": levels,
        "ace": ace,
        "nll": nll_gaussian(y, m, s),
        "sigma_mean": float(np.mean(s)),
        "sigma_std": float(np.std(s)),
        "n": int(y.size),
    }


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate one-sided Wilcoxon signed-rank p-value (a > b)."""
    if len(a) < 5:
        return 1.0
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    diff = diff[np.abs(diff) > 1e-12]
    if len(diff) < 5:
        return 1.0
    ranks = _rankdata(np.abs(diff))
    w_plus = float(np.sum(ranks[diff > 0]))
    n = len(diff)
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma <= 0:
        return 1.0
    z = (w_plus - mu) / sigma
    # Normal approximation, one-sided P(W+ > observed)
    from math import erfc, sqrt

    return float(0.5 * erfc(z / sqrt(2.0)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


def summarize_routing_results(
    method_metrics: Dict[str, List[float]],
    oracle_key: str = "oracle",
) -> Dict[str, Dict[str, float]]:
    out = {}
    oracle_risk = np.array(method_metrics.get(oracle_key, method_metrics.get("sparc_mfd", [])))
    for method, vals in method_metrics.items():
        arr = np.array(vals)
        out[method] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
        }
        if method != oracle_key and len(oracle_risk) == len(arr) and len(arr) > 0:
            out[method]["regret_vs_oracle"] = regret(arr, oracle_risk)
    return out
