"""Preprocessing helpers: bandpass filter, train/test split, PyTorch dataset adapter.

The MIT-BIH loader already returns z-score-normalized beats. This module provides additional
filtering (50 Hz powerline rejection / baseline wander removal) and the PyTorch glue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset

from .mitbih_loader import EcgDataset


@dataclass
class TrainTestSplit:
    train: EcgDataset
    test: EcgDataset


def bandpass_filter(
    signal: np.ndarray,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    sample_rate: float = 360.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter along the last axis.

    0.5 Hz removes baseline wander; 40 Hz removes powerline + EMG noise while preserving QRS.
    """
    nyquist = 0.5 * sample_rate
    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    return filtfilt(b, a, signal, axis=-1).astype(signal.dtype, copy=False)


def stratified_train_test_split(
    dataset: EcgDataset,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> TrainTestSplit:
    """Stratified split keeping each class's prevalence stable across train/test."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in np.unique(dataset.y):
        idx = np.flatnonzero(dataset.y == cls)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_fraction)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    train_idx_arr = np.array(train_idx, dtype=np.int64)
    test_idx_arr = np.array(test_idx, dtype=np.int64)
    rng.shuffle(train_idx_arr)

    train = EcgDataset(
        x=dataset.x[train_idx_arr],
        y=dataset.y[train_idx_arr],
        record_ids=dataset.record_ids[train_idx_arr],
        is_synthetic=dataset.is_synthetic,
    )
    test = EcgDataset(
        x=dataset.x[test_idx_arr],
        y=dataset.y[test_idx_arr],
        record_ids=dataset.record_ids[test_idx_arr],
        is_synthetic=dataset.is_synthetic,
    )
    return TrainTestSplit(train=train, test=test)


class TorchEcgDataset(Dataset):
    """Thin :class:`torch.utils.data.Dataset` adapter over :class:`EcgDataset`."""

    def __init__(self, ds: EcgDataset):
        self._x = torch.from_numpy(ds.x)        # (N, 1, W) float32
        self._y = torch.from_numpy(ds.y)        # (N,) int64

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x[idx], self._y[idx]
