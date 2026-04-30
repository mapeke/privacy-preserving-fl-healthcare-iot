"""Smoke tests for the ECG models."""

from __future__ import annotations

import torch

from src.models import build_model
from src.models.ecg_cnn import EcgCNN
from src.models.lightweight import LightweightEcgCNN


def test_ecg_cnn_forward_shape():
    model = EcgCNN(num_classes=5)
    x = torch.randn(8, 1, 360)
    y = model(x)
    assert y.shape == (8, 5)


def test_lightweight_forward_shape():
    model = LightweightEcgCNN(num_classes=5)
    x = torch.randn(4, 1, 360)
    y = model(x)
    assert y.shape == (4, 5)


def test_lightweight_has_fewer_params_than_full():
    full = EcgCNN(num_classes=5)
    light = LightweightEcgCNN(num_classes=5)
    assert light.num_parameters() < full.num_parameters()


def test_build_model_factory():
    assert isinstance(build_model("ecg_cnn"), EcgCNN)
    assert isinstance(build_model("lightweight"), LightweightEcgCNN)


def test_build_model_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        build_model("nope")
