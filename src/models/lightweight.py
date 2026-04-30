"""Lightweight 1D CNN for resource-constrained IoT edge devices.

Uses depthwise-separable convolutions (à la MobileNet) to drop parameter count by ~10x relative
to :class:`src.models.ecg_cnn.EcgCNN`, which matters when the model is shipped over a low-
bandwidth link to a wearable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DepthwiseSeparable1d(nn.Module):
    """Depthwise conv -> pointwise conv. Same receptive field as a regular conv at lower cost."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size, padding=padding, groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class LightweightEcgCNN(nn.Module):
    """Tiny 1D CNN suitable for on-device IoT inference.

    Input shape: ``(batch, 1, window_size)``.
    Output: ``(batch, num_classes)`` raw logits.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.GroupNorm(2, 8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.body = nn.Sequential(
            _DepthwiseSeparable1d(8, 16),
            nn.MaxPool1d(2),
            _DepthwiseSeparable1d(16, 16),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.body(self.stem(x)))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
