"""Differential privacy via Opacus.

Wraps :class:`opacus.PrivacyEngine` so the federated client can attach DP-SGD to a model and
optimizer with a single call, and read the spent ε at any point.

The framework follows the **per-round per-client** privacy budget convention: each client gets a
fresh accountant per round so its budget is the sum of within-round budgets across rounds, not
a global tightly composed budget. This matches how DP-FedAvg is typically reported in the
literature (McMahan et al. 2018) and avoids needing cross-round shared state in the simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time hints only
    import torch
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class DPConfig:
    """User-facing DP configuration."""

    target_epsilon: float = 5.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float | None = None  # None => auto-calibrate to target_epsilon
    accountant: str = "rdp"                # "rdp" | "gdp" | "prv"

    def validate(self) -> None:
        if self.target_epsilon <= 0:
            raise ValueError("target_epsilon must be positive")
        if not (0 < self.target_delta < 1):
            raise ValueError("target_delta must be in (0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.noise_multiplier is not None and self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be non-negative if provided")


@dataclass
class DPHandles:
    """Bundle of Opacus handles returned from :func:`attach_dp`."""

    engine: object  # opacus.PrivacyEngine
    model: "torch.nn.Module"
    optimizer: "torch.optim.Optimizer"
    data_loader: "DataLoader"
    epochs: int

    def get_epsilon(self) -> float:
        """Return spent ε under the configured ``target_delta``."""
        return float(self.engine.get_epsilon(delta=self._target_delta))

    @property
    def _target_delta(self) -> float:
        # Stored on engine.accountant by Opacus; we mirror it here for callers.
        return getattr(self, "target_delta", 1e-5)


def attach_dp(
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    data_loader: "DataLoader",
    config: DPConfig,
    *,
    epochs: int = 1,
) -> DPHandles:
    """Attach an Opacus :class:`PrivacyEngine` to (model, optimizer, dataloader).

    If ``config.noise_multiplier`` is ``None``, Opacus auto-calibrates it to hit
    ``config.target_epsilon`` over the requested ``epochs``.

    Returns wrapped versions of the inputs which the caller MUST use for training.
    """
    config.validate()
    from opacus import PrivacyEngine  # type: ignore[import-untyped]

    engine = PrivacyEngine(accountant=config.accountant)

    if config.noise_multiplier is None:
        wrapped_model, wrapped_opt, wrapped_loader = engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=config.target_epsilon,
            target_delta=config.target_delta,
            max_grad_norm=config.max_grad_norm,
            epochs=epochs,
        )
        logger.info(
            "Opacus auto-calibrated noise multiplier sigma=%.4f for "
            "target (epsilon=%.2f, delta=%.1e) over %d epoch(s).",
            wrapped_opt.noise_multiplier, config.target_epsilon, config.target_delta, epochs,
        )
    else:
        wrapped_model, wrapped_opt, wrapped_loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=config.noise_multiplier,
            max_grad_norm=config.max_grad_norm,
        )
        logger.info(
            "Attached Opacus with fixed sigma=%.4f, max_grad_norm=%.2f.",
            config.noise_multiplier, config.max_grad_norm,
        )

    handles = DPHandles(
        engine=engine,
        model=wrapped_model,
        optimizer=wrapped_opt,
        data_loader=wrapped_loader,
        epochs=epochs,
    )
    handles.target_delta = config.target_delta  # type: ignore[attr-defined]
    return handles
