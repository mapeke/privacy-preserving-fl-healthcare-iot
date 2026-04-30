"""Tests for the classification metrics helpers."""

from __future__ import annotations

import numpy as np

from src.utils.metrics import (
    AAMI_CLASSES,
    aggregate_reports,
    compute_classification_report,
)


def test_perfect_classification_metrics():
    y = np.array([0, 1, 2, 3, 4, 0, 1])
    report = compute_classification_report(y, y)
    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    for cls in AAMI_CLASSES:
        assert report.per_class_f1[cls] in (0.0, 1.0)


def test_shape_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        compute_classification_report(np.array([0, 1]), np.array([0, 1, 2]))


def test_aggregate_reports_weights_by_samples():
    y_a = np.array([0, 0, 1, 1])
    p_a = np.array([0, 0, 1, 1])  # 100% accuracy, 4 samples
    y_b = np.array([2, 3])
    p_b = np.array([4, 4])         # 0% accuracy, 2 samples
    r_a = compute_classification_report(y_a, p_a)
    r_b = compute_classification_report(y_b, p_b)
    agg = aggregate_reports([r_a, r_b])
    # Weighted avg: (1.0 * 4 + 0.0 * 2) / 6 = 4/6 ≈ 0.667
    assert abs(agg.accuracy - 4 / 6) < 1e-9
    assert agg.num_samples == 6
