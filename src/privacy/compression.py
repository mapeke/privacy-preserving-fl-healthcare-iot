"""Bandwidth-saving update compression: top-k sparsification + 8-bit quantization.

These reduce the size of each client-to-server message — important for IoT links — and they are
post-processing-immune to differential privacy. They do not interfere with secure aggregation as
long as both endpoints agree on the sparsification mask, which we transmit alongside the values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompressedUpdate:
    """Wire format for a compressed update.

    Attributes:
        indices: 1-D int32 indices into the flattened original update (or empty if not sparsified).
        values: corresponding values, dtype int8 or float32 depending on quantization.
        original_shape: shape of the dense original update.
        scale: dequantization scale (only meaningful when quantized; else 1.0).
        zero_point: dequantization zero point (only meaningful when quantized; else 0.0).
        is_quantized: whether ``values`` is int8 (True) or float32 (False).
        is_sparse: whether ``indices`` is meaningful.
    """

    indices: np.ndarray
    values: np.ndarray
    original_shape: tuple[int, ...]
    scale: float = 1.0
    zero_point: float = 0.0
    is_quantized: bool = False
    is_sparse: bool = False

    def num_bytes(self) -> int:
        return int(self.indices.nbytes + self.values.nbytes)


def top_k_sparsify(update: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the top-``ratio`` entries by absolute value.

    Args:
        update: dense numpy array (any shape — flattened internally).
        ratio: fraction in (0, 1] of entries to keep.

    Returns:
        ``(indices, values)`` where ``indices`` are positions in the flattened array.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    flat = update.ravel().astype(np.float32, copy=False)
    k = max(1, int(round(flat.size * ratio)))
    if k >= flat.size:
        return np.arange(flat.size, dtype=np.int32), flat.copy()
    # argpartition is O(n) and gives us the top-k unsorted, which is enough.
    idx = np.argpartition(np.abs(flat), -k)[-k:]
    idx = np.sort(idx).astype(np.int32)
    return idx, flat[idx].copy()


def quantize_int8(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Min-max quantize a float vector into int8.

    Returns ``(int8_values, scale, zero_point)`` so that
    ``values ≈ scale * (int8_values.astype(float32) - zero_point)``.
    """
    if values.size == 0:
        return values.astype(np.int8), 1.0, 0.0
    v_min = float(values.min())
    v_max = float(values.max())
    if v_max == v_min:
        return np.zeros_like(values, dtype=np.int8), 1.0, 0.0
    qmin, qmax = -128, 127
    scale = (v_max - v_min) / (qmax - qmin)
    zero_point = qmin - v_min / scale
    quant = np.round(values / scale + zero_point)
    quant = np.clip(quant, qmin, qmax).astype(np.int8)
    return quant, float(scale), float(zero_point)


def dequantize_int8(quant: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
    return (quant.astype(np.float32) - zero_point) * scale


def compress(
    update: np.ndarray,
    *,
    method: str = "top_k_quantize",
    top_k_ratio: float = 0.1,
    quant_bits: int = 8,
) -> CompressedUpdate:
    """Apply the configured compression pipeline.

    Supported methods: ``top_k_quantize`` (default), ``top_k_only``, ``quantize_only``, ``none``.
    """
    if quant_bits not in (8,):
        raise NotImplementedError("Only 8-bit quantization is implemented in this prototype.")
    method = method.lower()
    shape = update.shape

    if method == "none":
        return CompressedUpdate(
            indices=np.arange(update.size, dtype=np.int32),
            values=update.ravel().astype(np.float32),
            original_shape=shape,
        )

    if method in ("top_k_only", "top_k_quantize"):
        idx, vals = top_k_sparsify(update, top_k_ratio)
        is_sparse = True
    elif method == "quantize_only":
        idx = np.arange(update.size, dtype=np.int32)
        vals = update.ravel().astype(np.float32)
        is_sparse = False
    else:
        raise ValueError(f"Unknown compression method: {method!r}")

    if method in ("top_k_quantize", "quantize_only"):
        q, scale, zp = quantize_int8(vals)
        return CompressedUpdate(
            indices=idx,
            values=q,
            original_shape=shape,
            scale=scale,
            zero_point=zp,
            is_quantized=True,
            is_sparse=is_sparse,
        )

    return CompressedUpdate(
        indices=idx,
        values=vals,
        original_shape=shape,
        is_sparse=is_sparse,
    )


def decompress(payload: CompressedUpdate) -> np.ndarray:
    """Reconstruct a dense update array (zeros where sparsified)."""
    n = int(np.prod(payload.original_shape))
    out = np.zeros(n, dtype=np.float32)

    values = (
        dequantize_int8(payload.values, payload.scale, payload.zero_point)
        if payload.is_quantized
        else payload.values.astype(np.float32, copy=False)
    )
    out[payload.indices] = values
    return out.reshape(payload.original_shape)
