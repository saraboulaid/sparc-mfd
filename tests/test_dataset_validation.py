"""Tests for dataset inspection and FULL-mode guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from sparc_mfd.data_loader import (
    SYNTHETIC_N_STEPS,
    inspect_dataset,
    validate_dataset_for_full,
)


def test_metr_la_archive_is_real_dcrnn():
    report = inspect_dataset(Path("data/metr-la"), "metr-la")
    assert report.shape == [34272, 207]
    assert report.synthetic is False
    assert report.data_source == "real_dcrnn"
    assert report.ok_for_full() is True
    assert report.adjacency_shape == [207, 207]
    assert report.sensor_alignment_ok is True
    assert report.distances_synthetic is False
    assert report.file_hash is not None
    assert len(report.file_hash) == 64
    assert report.sampling_interval_minutes == 5.0
    assert report.n_timesteps > SYNTHETIC_N_STEPS


def test_pems_bay_archive_is_real_dcrnn():
    report = inspect_dataset(Path("data/pems-bay"), "pems-bay")
    assert report.shape == [52116, 325]
    assert report.synthetic is False
    assert report.data_source == "real_dcrnn"
    assert report.ok_for_full() is True


def test_full_mode_blocks_missing_archive(tmp_path):
    with pytest.raises(FileNotFoundError, match="not valid for FULL"):
        validate_dataset_for_full(tmp_path, "metr-la")


def test_full_mode_blocks_synthetic_signature(tmp_path, monkeypatch):
    """A 30-day archive carries the synthetic signature and must be refused."""
    from sparc_mfd.synthetic import generate_synthetic_dataset

    monkeypatch.chdir(tmp_path)
    generate_synthetic_dataset(tmp_path, dataset_name="metr-la", n_days=30)
    report = inspect_dataset(tmp_path, "metr-la")
    assert report.n_timesteps == SYNTHETIC_N_STEPS
    assert report.ok_for_full() is False
    with pytest.raises(FileNotFoundError, match="not valid for FULL"):
        validate_dataset_for_full(tmp_path, "metr-la")
