"""Neural-network architectures for ECG arrhythmia classification."""

from .ecg_cnn import EcgCNN
from .lightweight import LightweightEcgCNN

__all__ = ["EcgCNN", "LightweightEcgCNN", "build_model"]


def build_model(name: str, num_classes: int = 5, dropout: float = 0.3):
    """Factory used by experiment scripts and the federated client."""
    name = name.lower()
    if name == "ecg_cnn":
        return EcgCNN(num_classes=num_classes, dropout=dropout)
    if name == "lightweight":
        return LightweightEcgCNN(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model name: {name!r}")
