"""Custom Flower aggregation strategy with hooks for the privacy stack.

:class:`PrivacyAwareFedAvg` extends :class:`flwr.server.strategy.FedAvg` so it can:

* Treat client returns as **masked parameters** (= old + masked delta) when SecAgg is enabled,
  recover deltas by subtracting the broadcast parameters, sum them so the masks cancel, and
  apply the aggregate as ``new_global = old_global + agg_delta / n``.
* Fall back to vanilla weighted FedAvg averaging when SecAgg is disabled.
* Aggregate evaluation metrics with sample-weighted averaging by default.

This is the only place in the framework where the protocol's correctness assumption — masks
cancel because every pair appears once with each sign — is exercised at the system level.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.server import secure_aggregation as sec_srv

logger = logging.getLogger(__name__)


class PrivacyAwareFedAvg(FedAvg):
    """FedAvg variant that understands SecAgg-masked client returns."""

    def __init__(self, *args, secagg_enabled: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._secagg_enabled = bool(secagg_enabled)
        self._previous_parameters: Parameters | None = None

    # Capture broadcast parameters so we can recover deltas server-side.
    def configure_fit(self, server_round, parameters, client_manager):
        self._previous_parameters = parameters
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[BaseException | tuple[ClientProxy, FitRes]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self._secagg_enabled:
            # Standard weighted FedAvg.
            return super().aggregate_fit(server_round, results, failures)

        if self._previous_parameters is None:
            raise RuntimeError(
                "PrivacyAwareFedAvg: no broadcast parameters cached for SecAgg unmasking."
            )

        prev = parameters_to_ndarrays(self._previous_parameters)

        deltas_per_client: list[list[np.ndarray]] = []
        examples_per_client: list[int] = []
        for _, fit_res in results:
            new_params = parameters_to_ndarrays(fit_res.parameters)
            if len(new_params) != len(prev):
                raise RuntimeError(
                    f"Parameter count mismatch: client returned {len(new_params)}, expected {len(prev)}."
                )
            deltas = [
                (np.asarray(n_p) - np.asarray(p_p)).astype(np.float32)
                for p_p, n_p in zip(prev, new_params)
            ]
            deltas_per_client.append(deltas)
            examples_per_client.append(int(fit_res.num_examples))

        # Sum deltas — pairwise masks cancel here.
        summed = sec_srv.aggregate_deltas(deltas_per_client)
        averaged = sec_srv.weighted_average(summed, examples_per_client, use_weighted=True)

        new_global = [p + d for p, d in zip(prev, averaged)]

        # Aggregate scalar metrics by sample-weighted average where possible.
        metrics_aggregated = self._aggregate_scalar_metrics(results)

        logger.info(
            "Round %d SecAgg aggregation: %d clients, %d total examples.",
            server_round, len(results), sum(examples_per_client),
        )
        return ndarrays_to_parameters(new_global), metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[BaseException | tuple[ClientProxy, EvaluateRes]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        if not results:
            return None, {}
        loss_avg = (
            sum(er.loss * er.num_examples for _, er in results)
            / max(1, sum(er.num_examples for _, er in results))
        )
        agg_metrics: dict[str, Scalar] = {}
        keys = {k for _, er in results for k in er.metrics.keys()}
        total = sum(er.num_examples for _, er in results)
        for key in keys:
            try:
                agg_metrics[key] = (
                    sum(float(er.metrics.get(key, 0.0)) * er.num_examples for _, er in results)
                    / max(1, total)
                )
            except (TypeError, ValueError):
                agg_metrics[key] = next(er.metrics[key] for _, er in results if key in er.metrics)
        return float(loss_avg), agg_metrics

    @staticmethod
    def _aggregate_scalar_metrics(
        results: list[tuple[ClientProxy, FitRes]],
    ) -> dict[str, Scalar]:
        keys = {k for _, fr in results for k in fr.metrics.keys()}
        total = sum(fr.num_examples for _, fr in results) or 1
        agg: dict[str, Scalar] = {}
        for key in keys:
            try:
                agg[key] = sum(
                    float(fr.metrics.get(key, 0.0)) * fr.num_examples for _, fr in results
                ) / total
            except (TypeError, ValueError):
                # Non-numeric metric (e.g., device label). Keep first occurrence.
                agg[key] = next(fr.metrics[key] for _, fr in results if key in fr.metrics)
        return agg
