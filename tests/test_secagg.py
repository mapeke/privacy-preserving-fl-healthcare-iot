"""Tests for pairwise-mask secure aggregation.

The headline correctness property: summing every client's masked update equals summing the
original (unmasked) updates exactly. Equivalently, masks cancel under summation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.privacy.secure_agg import (
    SecAggSession,
    aggregate_masked_updates,
    derive_mask,
    mask_update,
    setup_session,
)


def test_setup_session_pair_count():
    sess = setup_session(num_clients=4, rng_seed=42)
    # C(4, 2) = 6 unordered pairs
    assert len(sess.pair_seeds) == 6


def test_setup_session_minimum_clients():
    with pytest.raises(ValueError):
        setup_session(num_clients=1)


def test_seed_lookup_symmetric():
    sess = setup_session(num_clients=3, rng_seed=42)
    assert sess.seed_for(0, 1) == sess.seed_for(1, 0)
    assert sess.seed_for(0, 1) != sess.seed_for(1, 2)


def test_derive_mask_deterministic_and_shape():
    seed = b"\x00" * 32
    a = derive_mask(seed, shape=(7,))
    b = derive_mask(seed, shape=(7,))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (7,)
    assert a.dtype == np.float32


def test_masks_cancel_under_aggregation():
    """The cryptographic correctness property in code form."""
    rng = np.random.default_rng(0)
    num_clients = 4
    update_shape = (10,)
    sess = setup_session(num_clients=num_clients, rng_seed=42)

    raw_updates = [rng.standard_normal(update_shape).astype(np.float32) for _ in range(num_clients)]
    masked = [
        mask_update(u, client_id=cid, session=sess)
        for cid, u in enumerate(raw_updates)
    ]

    # Each individual masked update is NOT equal to its raw update.
    for raw, m in zip(raw_updates, masked):
        assert not np.allclose(raw, m, atol=1e-6)

    # But their sums are equal (within float32 noise).
    agg = aggregate_masked_updates(masked)
    expected = np.sum(np.stack(raw_updates, axis=0), axis=0)
    np.testing.assert_allclose(agg, expected, atol=1e-4, rtol=1e-4)


def test_single_client_mask_is_nonzero():
    """Sanity check: a lone client's mask vector should be non-trivial."""
    sess = setup_session(num_clients=3, rng_seed=42)
    update = np.zeros(50, dtype=np.float32)
    masked = mask_update(update, client_id=0, session=sess)
    assert np.linalg.norm(masked) > 0.0


def test_invalid_client_id_raises():
    sess = setup_session(num_clients=3, rng_seed=42)
    with pytest.raises(ValueError):
        mask_update(np.zeros(4, dtype=np.float32), client_id=5, session=sess)
