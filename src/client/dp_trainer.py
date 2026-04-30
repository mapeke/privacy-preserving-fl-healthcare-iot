"""Local training loop, with optional Opacus DP-SGD.

The trainer exposes a single :func:`train_one_round` entry point used by both the federated
client and the centralized baseline. When the supplied :class:`DPConfig` is ``None``, training
proceeds as plain SGD; otherwise Opacus is attached and per-step gradients are clipped + noised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.privacy.differential_privacy import DPConfig, attach_dp

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """What :func:`train_one_round` returns to the calling federated client."""

    final_loss: float
    final_accuracy: float
    num_examples: int
    epsilon: float | None  # None when DP is disabled


def train_one_round(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    epochs: int = 1,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    dp_config: DPConfig | None = None,
    device: str | torch.device = "cpu",
) -> TrainResult:
    """Run ``epochs`` of local SGD (optionally DP-SGD) and return summary metrics."""
    device = torch.device(device)
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    handles = None
    active_loader = data_loader
    active_optimizer = optimizer
    if dp_config is not None:
        handles = attach_dp(model, optimizer, data_loader, dp_config, epochs=epochs)
        model = handles.model
        active_optimizer = handles.optimizer
        active_loader = handles.data_loader

    final_loss = 0.0
    correct = 0
    seen = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_seen = 0
        for x, y in active_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            active_optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            active_optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                epoch_correct += int((preds == y).sum().item())
                epoch_seen += int(y.numel())
                epoch_loss += float(loss.item()) * int(y.numel())

        final_loss = epoch_loss / max(1, epoch_seen)
        correct = epoch_correct
        seen = epoch_seen
        logger.debug(
            "epoch %d/%d  loss=%.4f  acc=%.4f  n=%d",
            epoch + 1, epochs, final_loss, correct / max(1, seen), seen,
        )

    epsilon = handles.get_epsilon() if handles is not None else None
    return TrainResult(
        final_loss=final_loss,
        final_accuracy=(correct / max(1, seen)),
        num_examples=seen,
        epsilon=epsilon,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: str | torch.device = "cpu",
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute (loss, accuracy, y_true, y_pred) over ``data_loader``."""
    device = torch.device(device)
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    seen = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    for x, y in data_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += float(criterion(logits, y).item())
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        seen += int(y.numel())
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.empty(0, dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.empty(0, dtype=np.int64)
    return total_loss / max(1, seen), correct / max(1, seen), y_true, y_pred
