"""Server-side helpers for the pairwise-mask secure aggregation protocol.

The cryptographic primitives live in :mod:`src.privacy.secure_agg`. This module is the server's
viewpoint: it sums masked deltas across clients (where the masks cancel by construction) and
returns the unmasked aggregate.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def aggregate_deltas(deltas_per_client: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Element-wise sum of per-client per-parameter deltas.

    Args:
        deltas_per_client: ``deltas_per_client[i]`` is the list of parameter deltas (one ndarray
            per parameter tensor, in canonical model order) submitted by client ``i``. With the
            pairwise-mask protocol, each delta is ``Δw_i + Σ ±m_ij``; summing across clients
            cancels every mask.

    Returns:
        The aggregated deltas in the same canonical order, suitable for adding to the previous
        global parameters to obtain the new global parameters.
    """
    if not deltas_per_client:
        raise ValueError("aggregate_deltas: empty input")
    n_params = len(deltas_per_client[0])
    if any(len(d) != n_params for d in deltas_per_client):
        raise ValueError("aggregate_deltas: clients submitted different numbers of parameters")

    aggregated: list[np.ndarray] = []
    for p in range(n_params):
        stack = np.stack([np.asarray(d[p]) for d in deltas_per_client], axis=0)
        aggregated.append(stack.sum(axis=0).astype(np.float32))
    return aggregated


def weighted_average(
    aggregated_deltas: list[np.ndarray],
    num_examples_per_client: list[int],
    *,
    use_weighted: bool = True,
) -> list[np.ndarray]:
    """Convert the SecAgg sum into a (weighted) average so it composes with FedAvg's update rule.

    With ``use_weighted=True`` the aggregator divides by ``Σ n_i`` and multiplies in the
    per-client weight ahead of time. This requires the *clients* to scale their deltas by
    ``n_i`` before masking, which the server cannot enforce; so by default we just divide by
    the number of clients (FedAvg with equal weights). Pass ``use_weighted=False`` to keep that
    behavior explicit.
    """
    if not aggregated_deltas:
        raise ValueError("weighted_average: empty input")
    n = max(1, len(num_examples_per_client))
    if use_weighted:
        total = max(1, sum(num_examples_per_client))
        # SecAgg cancels masks but cannot recover individual weights — fall back to equal split.
        # The framework's PrivacyAwareFedAvg therefore documents that SecAgg + non-uniform
        # weighting requires a (here-unimplemented) "input-weighted" SecAgg variant.
        logger.debug(
            "SecAgg unmasked aggregate divided by n=%d; weighted-by-examples (sum=%d) not "
            "applicable without input-weighted SecAgg.",
            n, total,
        )
    return [d / n for d in aggregated_deltas]
