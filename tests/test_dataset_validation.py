"""Tests for dataset inspection and FULL-mode guards."""

from __future__ import annotations

import pytest

from sparc_mfd.data_loader import (
    SYNTHETIC_N_STEPS,
    inspect_dataset,
    validate_dataset_for_full,
)


def test_current_metr_la_detected_as_synthetic():
    report = inspect_dataset(__import__("pathlib").Path("data/metr-la"), "metr-la")
    assert report.shape == [SYNTHETIC_N_STEPS, 207]
    assert report.synthetic is True
    assert report.data_source == "synthetic_fallback"
    assert report.ok_for_full() is False
    assert report.adjacency_shape == [207, 207]
    assert report.sensor_alignment_ok is True
    assert report.file_hash is not None
    assert len(report.file_hash) == 64
    assert report.sampling_interval_minutes == 5.0


def test_current_pems_bay_detected_as_synthetic():
    report = inspect_dataset(__import__("pathlib").Path("data/pems-bay"), "pems-bay")
    assert report.shape == [SYNTHETIC_N_STEPS, 325]
    assert report.synthetic is True
    assert report.ok_for_full() is False


def test_full_mode_blocks_synthetic():
    with pytest.raises(FileNotFoundError, match="not valid for FULL"):
        validate_dataset_for_full(__import__("pathlib").Path("data/metr-la"), "metr-la")
