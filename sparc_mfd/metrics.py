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
