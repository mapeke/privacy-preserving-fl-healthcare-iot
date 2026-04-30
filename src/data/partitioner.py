"""Partition a flat ECG dataset across simulated IoT clients.

Two schemes are supported:

* ``iid``: random shuffle then equal split — every client sees the global class distribution.
* ``dirichlet``: per-class Dirichlet sampling with concentration parameter ``alpha`` controlling
  the heterogeneity. Lower ``alpha`` = more skewed = more "non-IID" — closer to the realistic
  per-patient setting where each device only ever sees its owner's arrhythmias.

A third "natural" scheme (one client per MIT-BIH record) is also exposed via :func:`per_record`.
"""

from __future__ import annotations

import logging

import numpy as np

from .mitbih_loader import EcgDataset

logger = logging.getLogger(__name__)


def iid(dataset: EcgDataset, num_clients: int, seed: int = 0) -> list[EcgDataset]:
    """Random equal-size partition. Every client sees the global class mix."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [_select(dataset, split) for split in splits]


def dirichlet(
    dataset: EcgDataset,
    num_clients: int,
    alpha: float = 0.5,
    *,
    min_samples_per_client: int = 10,
    seed: int = 0,
    max_attempts: int = 20,
) -> list[EcgDataset]:
    """Dirichlet non-IID partition.

    For each class, sample a probability vector ``p ~ Dir(alpha · 1_K)`` over the ``num_clients``
    clients and split that class's indices accordingly. Lower ``alpha`` => more skewed assignment.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    rng = np.random.default_rng(seed)
    classes = np.unique(dataset.y)

    for attempt in range(max_attempts):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_idx = np.flatnonzero(dataset.y == cls)
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet([alpha] * num_clients)
            cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            for client_id, chunk in enumerate(np.split(cls_idx, cuts)):
                client_indices[client_id].extend(chunk.tolist())

        sizes = [len(idx) for idx in client_indices]
        if min(sizes) >= min_samples_per_client:
            logger.info(
                "Dirichlet partition (alpha=%.2f): client sizes = %s", alpha, sizes
            )
            return [_select(dataset, np.array(idx, dtype=np.int64)) for idx in client_indices]

        logger.debug(
            "Dirichlet attempt %d produced min size %d < %d; retrying.",
            attempt + 1, min(sizes), min_samples_per_client,
        )

    raise RuntimeError(
        f"Could not satisfy min_samples_per_client={min_samples_per_client} "
        f"after {max_attempts} attempts. Try lowering it, raising alpha, "
        "or reducing num_clients."
    )


def per_record(dataset: EcgDataset) -> list[EcgDataset]:
    """One client per MIT-BIH record — the most realistic per-patient simulation."""
    clients: list[EcgDataset] = []
    for rec in np.unique(dataset.record_ids):
        idx = np.flatnonzero(dataset.record_ids == rec)
        clients.append(_select(dataset, idx))
    return clients


def partition(
    dataset: EcgDataset,
    num_clients: int,
    scheme: str = "dirichlet",
    *,
    alpha: float = 0.5,
    min_samples_per_client: int = 10,
    seed: int = 0,
) -> list[EcgDataset]:
    """Dispatch entry point used by experiment scripts."""
    scheme = scheme.lower()
    if scheme == "iid":
        return iid(dataset, num_clients, seed=seed)
    if scheme == "dirichlet":
        return dirichlet(
            dataset, num_clients,
            alpha=alpha,
            min_samples_per_client=min_samples_per_client,
            seed=seed,
        )
    if scheme == "per_record":
        return per_record(dataset)
    raise ValueError(f"Unknown partition scheme: {scheme!r}")


def _select(dataset: EcgDataset, indices: np.ndarray) -> EcgDataset:
    return EcgDataset(
        x=dataset.x[indices],
        y=dataset.y[indices],
        record_ids=dataset.record_ids[indices],
        is_synthetic=dataset.is_synthetic,
    )
