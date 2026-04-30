"""Flower NumPyClient implementation that ties together the privacy stack.

Each round, the client:
  1. Receives the global parameters from the server.
  2. Trains locally for a few epochs (optionally with DP-SGD via Opacus).
  3. Computes the parameter delta ``Δw_i = w_i_new - w_i_old``.
  4. Optionally compresses the delta (top-k + 8-bit quant).
  5. Optionally adds the pairwise SecAgg masks.
  6. Returns the (compressed, masked) parameters back to the server.

Notes on the protocol surface:
- We always return *parameters* (not deltas) to stay compatible with vanilla Flower strategies
  when no privacy stack is active. The custom :class:`PrivacyAwareFedAvg` strategy can ask for
  deltas instead via the ``config`` channel.
- Returned compressed/masked tensors are flattened to a single 1-D float32 vector per parameter
  so Flower's transport stays uniform.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.client.dp_trainer import evaluate, train_one_round
from src.client.iot_device import IoTDeviceProfile
from src.privacy import compression as comp
from src.privacy import secure_agg
from src.privacy.differential_privacy import DPConfig

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Per-client runtime configuration assembled by the experiment script."""

    client_id: int
    local_epochs: int
    learning_rate: float
    momentum: float
    weight_decay: float
    device: str = "cpu"
    dp_config: DPConfig | None = None
    secagg_session: secure_agg.SecAggSession | None = None
    compression_method: str | None = None      # None | "none" | "top_k_only" | "quantize_only" | "top_k_quantize"
    compression_top_k_ratio: float = 0.1
    compression_bits: int = 8
    iot_profile: IoTDeviceProfile | None = None


class FlowerHealthcareClient(fl.client.NumPyClient):
    """Flower client wrapping a PyTorch model + ECG data loaders."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: ClientConfig,
    ):
        self._model = model
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._config = config
        self._param_keys = list(model.state_dict().keys())

    # ---- Flower interface -------------------------------------------------

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:  # noqa: D401, ARG002
        return [v.detach().cpu().numpy() for v in self._model.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        state = {k: torch.from_numpy(np.asarray(v)) for k, v in zip(self._param_keys, parameters)}
        self._model.load_state_dict(state, strict=True)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """Local training step. Returns (updated_parameters, num_examples, metrics)."""
        self.set_parameters(parameters)

        result = train_one_round(
            self._model,
            self._train_loader,
            epochs=self._config.local_epochs,
            learning_rate=self._config.learning_rate,
            momentum=self._config.momentum,
            weight_decay=self._config.weight_decay,
            dp_config=self._config.dp_config,
            device=self._config.device,
        )

        new_params = self.get_parameters({})
        out_params = self._maybe_apply_privacy_stack(parameters, new_params)

        metrics: dict[str, Any] = {
            "train_loss": float(result.final_loss),
            "train_accuracy": float(result.final_accuracy),
            "client_id": int(self._config.client_id),
        }
        if result.epsilon is not None:
            metrics["epsilon"] = float(result.epsilon)
        if self._config.iot_profile is not None:
            metrics["device_label"] = self._config.iot_profile.label
            metrics["bandwidth_kbps"] = self._config.iot_profile.bandwidth_kbps

        return out_params, int(result.num_examples), metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],  # noqa: ARG002
    ) -> tuple[float, int, dict[str, Any]]:
        self.set_parameters(parameters)
        loss, acc, _, _ = evaluate(self._model, self._test_loader, device=self._config.device)
        n = int(sum(yb.numel() for _, yb in self._test_loader))
        return float(loss), n, {"accuracy": float(acc), "client_id": int(self._config.client_id)}

    # ---- Privacy stack ----------------------------------------------------

    def _maybe_apply_privacy_stack(
        self,
        old_params: list[np.ndarray],
        new_params: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Optionally compress and SecAgg-mask the parameter delta.

        Returns reconstructed *parameters* (old + processed_delta) so vanilla strategies still
        work. The custom :class:`PrivacyAwareFedAvg` server strategy can subtract its broadcast
        parameters to recover the delta server-side.
        """
        cfg = self._config

        deltas: list[np.ndarray] = [
            (np.asarray(n_p) - np.asarray(o_p)).astype(np.float32)
            for o_p, n_p in zip(old_params, new_params)
        ]

        if cfg.compression_method and cfg.compression_method != "none":
            deltas = [
                comp.decompress(comp.compress(
                    d,
                    method=cfg.compression_method,
                    top_k_ratio=cfg.compression_top_k_ratio,
                    quant_bits=cfg.compression_bits,
                ))
                for d in deltas
            ]

        if cfg.secagg_session is not None:
            deltas = [
                secure_agg.mask_update(d, client_id=cfg.client_id, session=cfg.secagg_session)
                for d in deltas
            ]

        return [
            (np.asarray(o_p) + d).astype(np.float32)
            for o_p, d in zip(old_params, deltas)
        ]
