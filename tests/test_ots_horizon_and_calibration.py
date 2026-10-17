"""Tests for the OTS horizon and for uncertainty calibration metrics."""

from __future__ import annotations

import numpy as np
import pytest

from sparc_mfd.metrics import (
    COVERAGE_LEVELS,
    calibration_report,
    interval_score,
    mpiw,
    picp,
)
from sparc_mfd.ots_encoder import compute_ots_components, fit_ots_stats


def _speed(t: int = 300, n: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = 55.0 + 10.0 * np.sin(np.arange(t) / 8.0)[:, None]
    return (base + rng.normal(0, 4.0, size=(t, n))).astype(np.float32)


def test_horizon_one_makes_volatility_degenerate():
    """The published runs used h=1, which zeroes the volatility term."""
    comp = compute_ots_components(_speed(), horizon=1)
    assert np.all(comp["V"] == 0.0)
    # Persistence collapses to an indicator over a single future step.
    assert set(np.unique(comp["P"])) <= {0.0, 1.0}


def test_horizon_three_restores_volatility_and_persistence():
    comp = compute_ots_components(_speed(), horizon=3)
    assert comp["V"].std() > 0
    assert len(set(np.unique(comp["P"]))) > 2


def test_degenerate_component_std_hits_the_floor():
    """With h=1 the standardisation divides by the 1e-6 floor, not by zero."""
    comp = compute_ots_components(_speed(), horizon=1)
    stats = fit_ots_stats(comp, train_end=200)
    assert stats["V_std"] == pytest.approx(1e-6, rel=1e-5)
    assert stats["V_mean"] == 0.0


def test_picp_matches_nominal_for_well_specified_gaussian():
    rng = np.random.default_rng(3)
    n = 200_000
    mu = np.zeros(n)
    sigma = np.full(n, 0.7)
    y = mu + rng.normal(0, 0.7, size=n)
    for level in COVERAGE_LEVELS:
        assert abs(picp(y, mu, sigma, level) - level) < 0.01


def test_picp_detects_overconfidence():
    rng = np.random.default_rng(4)
    n = 50_000
    y = rng.normal(0, 1.0, size=n)
    mu = np.zeros(n)
    too_narrow = np.full(n, 0.25)
    assert picp(y, mu, too_narrow, 0.90) < 0.5


def test_mpiw_scales_with_sigma_and_level():
    sigma = np.full(100, 0.5)
    assert mpiw(sigma, 0.90) > mpiw(sigma, 0.50)
    assert mpiw(sigma * 2, 0.90) == 2 * mpiw(sigma, 0.90)


def test_interval_score_penalises_misses():
    y = np.array([0.0, 0.0])
    mu = np.array([0.0, 0.0])
    tight_and_right = interval_score(y, mu, np.array([1.0, 1.0]), 0.90)
    tight_and_wrong = interval_score(np.array([5.0, 5.0]), mu, np.array([0.1, 0.1]), 0.90)
    assert tight_and_wrong > tight_and_right


def test_calibration_report_structure():
    rng = np.random.default_rng(5)
    n = 20_000
    sigma = rng.uniform(0.3, 1.2, size=n)
    mu = rng.normal(0, 1, size=n)
    y = mu + rng.normal(0, 1, size=n) * sigma
    report = calibration_report(y, mu, sigma)
    assert report["n"] == n
    assert len(report["levels"]) == len(COVERAGE_LEVELS)
    assert report["ace"] < 0.02
    assert np.isfinite(report["nll"])
    for entry in report["levels"]:
        assert entry["picp"] >= 0.0
        assert entry["mpiw"] > 0.0
        assert abs(entry["gap"] - (entry["picp"] - entry["nominal"])) < 1e-12
