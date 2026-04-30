"""Full 1D CNN for ECG arrhythmia classification.

Architecture inspired by Kachuee et al. ("ECG Heartbeat Classification: A Deep Transferable
Representation", 2018) but trimmed to be Opacus-compatible (BatchNorm replaced with GroupNorm
because Opacus does not support BN under DP-SGD).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Conv -> GroupNorm -> ReLU -> Conv -> GroupNorm -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, *, groups: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EcgCNN(nn.Module):
    """1D CNN for windowed ECG beat classification.

    Input shape: ``(batch, 1, window_size)`` — typically ``(batch, 1, 360)``.
    Output: ``(batch, num_classes)`` raw logits.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(1, 16),
            _ConvBlock(16, 32),
            _ConvBlock(32, 64),
        )
        # After 3 max-pools of stride 2: 360 -> 180 -> 90 -> 45
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
