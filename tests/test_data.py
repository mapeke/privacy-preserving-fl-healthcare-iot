"""Tests for the data layer (synthetic generator + partitioner + preprocessing)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.mitbih_loader import generate_synthetic_ecg
from src.data.partitioner import partition
from src.data.preprocessing import (
    TorchEcgDataset,
    bandpass_filter,
    stratified_train_test_split,
)


@pytest.fixture
def synthetic_dataset():
    return generate_synthetic_ecg(n_samples_per_record=120, records=(100, 101, 102), window_size=180)


def test_synthetic_dataset_shapes_and_labels(synthetic_dataset):
    ds = synthetic_dataset
    assert ds.x.ndim == 3
    assert ds.x.shape[1] == 1
    assert ds.x.shape[2] == 180
    assert ds.x.shape[0] == ds.y.shape[0] == ds.record_ids.shape[0]
    assert ds.y.dtype == np.int64
    # All labels in 0..4
    assert set(np.unique(ds.y).tolist()).issubset({0, 1, 2, 3, 4})
    assert ds.is_synthetic is True


def test_synthetic_dataset_is_deterministic():
    a = generate_synthetic_ecg(n_samples_per_record=50, records=(100,), window_size=180, seed=7)
    b = generate_synthetic_ecg(n_samples_per_record=50, records=(100,), window_size=180, seed=7)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.y, b.y)


def test_bandpass_filter_preserves_shape(synthetic_dataset):
    flat = synthetic_dataset.x[:, 0, :]
    filtered = bandpass_filter(flat, sample_rate=180.0)
    assert filtered.shape == flat.shape
    assert filtered.dtype == flat.dtype


def test_stratified_split_preserves_class_presence(synthetic_dataset):
    split = stratified_train_test_split(synthetic_dataset, test_fraction=0.25, seed=0)
    train_classes = set(np.unique(split.train.y).tolist())
    test_classes = set(np.unique(split.test.y).tolist())
    assert train_classes == test_classes
    assert len(split.train) + len(split.test) == len(synthetic_dataset)


def test_iid_partition_equal_sizes(synthetic_dataset):
    parts = partition(synthetic_dataset, num_clients=4, scheme="iid")
    sizes = [len(p) for p in parts]
    assert sum(sizes) == len(synthetic_dataset)
    assert max(sizes) - min(sizes) <= 1  # within 1 sample of equal


def test_dirichlet_partition_total_preserved(synthetic_dataset):
    parts = partition(
        synthetic_dataset, num_clients=4, scheme="dirichlet",
        alpha=0.5, min_samples_per_client=20, seed=0,
    )
    assert sum(len(p) for p in parts) == len(synthetic_dataset)
    for p in parts:
        assert len(p) >= 20


def test_torch_dataset_indexing(synthetic_dataset):
    td = TorchEcgDataset(synthetic_dataset)
    assert len(td) == len(synthetic_dataset)
    x, y = td[0]
    assert x.shape == (1, 180)
    assert y.dtype.is_floating_point is False
