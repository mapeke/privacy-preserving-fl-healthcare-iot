"""Tests for the differential-privacy wrapper.

Heavy Opacus end-to-end tests would require a working PyTorch + Opacus install at test time,
which complicates CI. We instead test:

1. ``DPConfig`` validation rejects nonsensical values.
2. ``attach_dp`` only imports Opacus lazily (the import lives inside the function body).

A full integration test that actually runs DP-SGD lives implicitly in
``experiments/run_federated_dp.py`` and can be exercised manually.
"""

from __future__ import annotations

import inspect

import pytest

from src.privacy.differential_privacy import DPConfig, attach_dp


def test_dp_config_defaults_are_valid():
    cfg = DPConfig()
    cfg.validate()
    assert cfg.target_epsilon > 0
    assert 0 < cfg.target_delta < 1
    assert cfg.max_grad_norm > 0
    assert cfg.accountant in {"rdp", "gdp", "prv"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_epsilon": 0.0},
        {"target_epsilon": -1.0},
        {"target_delta": 0.0},
        {"target_delta": 1.0},
        {"max_grad_norm": 0.0},
        {"noise_multiplier": -0.1},
    ],
)
def test_dp_config_validation_rejects_invalid(kwargs):
    base = {"target_epsilon": 5.0, "target_delta": 1e-5, "max_grad_norm": 1.0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        DPConfig(**base).validate()


def test_attach_dp_imports_opacus_lazily():
    """If Opacus isn't installed, attach_dp should fail at call-time, not import-time."""
    src = inspect.getsource(attach_dp)
    # Sanity: the import sits inside the function body, not at module top.
    assert "from opacus import PrivacyEngine" in src
