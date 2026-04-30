"""Pairwise-mask secure aggregation primitives.

Implements the textbook "additive masking" scheme (Bonawitz et al., 2017, simplified): every
unordered pair of clients ``{i, j}`` shares a seed; client ``i`` adds ``+m_ij`` and client ``j``
adds ``-m_ij`` to its update; the server's sum cancels every mask exactly.

Limitations of this prototype (intentional, documented in docs/privacy-analysis.md):

* No Diffie-Hellman key exchange — seeds are pre-shared by the simulation orchestrator.
* No Shamir secret sharing — the protocol does not tolerate dropouts.
* Float32 arithmetic — production deployments use modular integer arithmetic.

These are appropriate trade-offs for a diploma-scale simulation that wants to demonstrate the
*principle* and measure its bandwidth / accuracy overhead end-to-end.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SecAggSession:
    """Pre-shared seeds for one round of secure aggregation.

    ``pair_seeds[(i, j)]`` is the seed shared by clients ``i`` and ``j`` (with ``i < j``).
    """

    num_clients: int
    pair_seeds: dict[tuple[int, int], bytes]

    def seed_for(self, i: int, j: int) -> bytes:
        if i == j:
            raise ValueError("seed_for requires distinct client indices")
        key = (i, j) if i < j else (j, i)
        return self.pair_seeds[key]


def setup_session(num_clients: int, *, seed_size_bytes: int = 32, rng_seed: int | None = None) -> SecAggSession:
    """Generate one fresh secret seed per unordered pair of clients.

    Args:
        num_clients: Number of participating clients.
        seed_size_bytes: Length of each PRG seed.
        rng_seed: If given, makes the (otherwise cryptographic) seed generation deterministic
            for tests. **Never set this in production** — it makes seeds predictable.

    Returns:
        :class:`SecAggSession` ready to be distributed to clients.
    """
    if num_clients < 2:
        raise ValueError("Secure aggregation requires at least 2 clients")
    if seed_size_bytes < 16:
        raise ValueError("seed_size_bytes must be >= 16 for any meaningful security")

    if rng_seed is None:
        gen = lambda: secrets.token_bytes(seed_size_bytes)  # noqa: E731
    else:
        det_rng = np.random.default_rng(rng_seed)
        gen = lambda: bytes(det_rng.integers(0, 256, size=seed_size_bytes, dtype=np.uint8))  # noqa: E731

    pair_seeds: dict[tuple[int, int], bytes] = {}
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            pair_seeds[(i, j)] = gen()
    return SecAggSession(num_clients=num_clients, pair_seeds=pair_seeds)


def derive_mask(seed: bytes, shape: tuple[int, ...] | int) -> np.ndarray:
    """Expand a pair seed into a mask vector with the given shape using NumPy's PCG64.

    The mask is centered Gaussian with unit variance; the choice of distribution is irrelevant for
    correctness (since masks cancel exactly) but Gaussian is convenient for testing.
    """
    if isinstance(shape, int):
        shape = (shape,)
    seed_int = int.from_bytes(seed, byteorder="big", signed=False)
    rng = np.random.default_rng(seed_int)
    return rng.standard_normal(size=shape).astype(np.float32)


def mask_update(
    update: np.ndarray,
    client_id: int,
    session: SecAggSession,
) -> np.ndarray:
    """Add the pairwise masks for ``client_id`` to ``update``.

    For every other client ``j``: if ``client_id < j``, add ``+m_{ij}``; otherwise add ``-m_{ji}``.
    """
    if not (0 <= client_id < session.num_clients):
        raise ValueError(f"client_id {client_id} out of range [0, {session.num_clients})")

    masked = update.astype(np.float32, copy=True)
    for j in range(session.num_clients):
        if j == client_id:
            continue
        seed = session.seed_for(client_id, j)
        mask = derive_mask(seed, update.shape)
        if client_id < j:
            masked += mask
        else:
            masked -= mask
    return masked


def aggregate_masked_updates(masked_updates: list[np.ndarray]) -> np.ndarray:
    """Element-wise sum of all masked updates.

    Because every pair-mask appears once with ``+`` and once with ``-``, the sum equals the sum of
    the original (unmasked) updates exactly.
    """
    if not masked_updates:
        raise ValueError("empty update list")
    stacked = np.stack(masked_updates, axis=0)
    return stacked.sum(axis=0)
