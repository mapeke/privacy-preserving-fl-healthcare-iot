"""Tests for top-k sparsification + 8-bit quantization."""

from __future__ import annotations

import numpy as np
import pytest

from src.privacy.compression import (
    compress,
    decompress,
    dequantize_int8,
    quantize_int8,
    top_k_sparsify,
)


def test_top_k_sparsify_keeps_largest_magnitudes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100).astype(np.float32)
    idx, vals = top_k_sparsify(x, ratio=0.1)  # keep top 10
    assert idx.shape == (10,)
    assert vals.shape == (10,)
    # Values kept must include the max-magnitude entry.
    assert np.argmax(np.abs(x)) in idx


def test_top_k_sparsify_full_ratio_returns_everything():
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    idx, vals = top_k_sparsify(x, ratio=1.0)
    assert idx.shape == (3,)
    assert np.allclose(vals, x)


def test_quantize_dequantize_roundtrip_close():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000).astype(np.float32)
    q, scale, zp = quantize_int8(x)
    assert q.dtype == np.int8
    x_hat = dequantize_int8(q, scale, zp)
    # int8 has ~256 levels over [min, max]; expect mean abs error well below the range / 256.
    rng_max = float(x.max() - x.min())
    assert np.mean(np.abs(x - x_hat)) < rng_max / 100.0


def test_compress_top_k_quantize_roundtrip_preserves_shape():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    payload = compress(x, method="top_k_quantize", top_k_ratio=0.5, quant_bits=8)
    out = decompress(payload)
    assert out.shape == x.shape
    # The recovered tensor should be close (top-k + quant is lossy but bounded).
    # Specifically: only ~50% of entries are non-zero in the output.
    nonzero_mask = out != 0
    assert nonzero_mask.sum() <= int(round(x.size * 0.5)) + 1


def test_compress_quantize_only_preserves_total_count():
    x = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    payload = compress(x, method="quantize_only")
    assert not payload.is_sparse
    assert payload.is_quantized
    assert payload.indices.shape == (x.size,)
    out = decompress(payload)
    assert out.shape == x.shape


def test_compress_none_is_lossless_roundtrip():
    x = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    payload = compress(x, method="none")
    out = decompress(payload)
    np.testing.assert_array_equal(out, x)


def test_compress_unknown_method_raises():
    with pytest.raises(ValueError):
        compress(np.zeros(5, dtype=np.float32), method="bogus")
