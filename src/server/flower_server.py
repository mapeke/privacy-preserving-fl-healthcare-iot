"""Flower simulation orchestrator.

The framework runs Flower's in-process simulation (``flwr.simulation.start_simulation``) so we
can spawn N clients, each with its own data partition and IoT profile, in a single Python
process. This keeps the demo reproducible (no cluster setup) while still exercising the full
client/server message flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import flwr as fl
import numpy as np
import torch
from flwr.common import Parameters, ndarrays_to_parameters

from src.server.strategies import PrivacyAwareFedAvg

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Inputs to :func:`run_simulation` assembled by experiment scripts."""

    num_clients: int
    num_rounds: int
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    secagg_enabled: bool = False
    initial_parameters: Parameters | None = None


ClientFactory = Callable[[str], fl.client.NumPyClient]


def make_initial_parameters(model: torch.nn.Module) -> Parameters:
    """Snapshot a freshly initialised model into Flower's wire format."""
    arrays = [v.detach().cpu().numpy() for v in model.state_dict().values()]
    return ndarrays_to_parameters(arrays)


def run_simulation(
    client_factory: ClientFactory,
    sim: SimulationConfig,
    *,
    on_fit_metrics_aggregation_fn=None,
    on_evaluate_metrics_aggregation_fn=None,
) -> fl.server.History:
    """Spawn ``sim.num_clients`` Flower clients via ``client_factory(cid)`` and run training.

    Args:
        client_factory: function ``cid -> NumPyClient`` Flower will call to materialise each
            simulated client. ``cid`` is the string client id Flower assigns ("0", "1", ...).
        sim: simulation configuration.
        on_fit_metrics_aggregation_fn: optional Flower hook for aggregating per-round fit
            metrics (e.g., to compute average epsilon across clients).
        on_evaluate_metrics_aggregation_fn: optional hook for evaluation metrics.

    Returns:
        Flower's ``History`` object with per-round metrics.
    """
    strategy = PrivacyAwareFedAvg(
        fraction_fit=sim.fraction_fit,
        fraction_evaluate=sim.fraction_evaluate,
        min_fit_clients=sim.num_clients,
        min_evaluate_clients=sim.num_clients,
        min_available_clients=sim.num_clients,
        initial_parameters=sim.initial_parameters,
        secagg_enabled=sim.secagg_enabled,
        fit_metrics_aggregation_fn=on_fit_metrics_aggregation_fn or _mean_metrics_aggregation,
        evaluate_metrics_aggregation_fn=on_evaluate_metrics_aggregation_fn or _mean_metrics_aggregation,
    )

    history = fl.simulation.start_simulation(
        client_fn=lambda cid: client_factory(cid).to_client(),
        num_clients=sim.num_clients,
        config=fl.server.ServerConfig(num_rounds=sim.num_rounds),
        strategy=strategy,
    )
    return history


def _mean_metrics_aggregation(metrics):
    """Default Flower-compatible metric aggregator: sample-weighted mean for numerics."""
    if not metrics:
        return {}
    keys = {k for _, m in metrics for k in m.keys()}
    total = sum(n for n, _ in metrics) or 1
    out: dict[str, float] = {}
    for k in keys:
        values = [(n, m.get(k)) for n, m in metrics if k in m]
        try:
            out[k] = sum(float(v) * n for n, v in values) / total
        except (TypeError, ValueError):
            # Non-numeric — drop it from the aggregate.
            continue
    return out


def history_to_dict(history: fl.server.History) -> dict:
    """Convert a Flower ``History`` into a JSON-serialisable nested dict."""
    return {
        "losses_distributed": _serialize_pairs(history.losses_distributed),
        "losses_centralized": _serialize_pairs(history.losses_centralized),
        "metrics_distributed_fit": _serialize_metric_history(history.metrics_distributed_fit),
        "metrics_distributed": _serialize_metric_history(history.metrics_distributed),
        "metrics_centralized": _serialize_metric_history(history.metrics_centralized),
    }


def _serialize_pairs(pairs):
    return [[int(r), float(v)] for r, v in (pairs or [])]


def _serialize_metric_history(d):
    out: dict[str, list[list[float]]] = {}
    for k, pairs in (d or {}).items():
        out[str(k)] = [[int(r), _to_jsonable(v)] for r, v in pairs]
    return out


def _to_jsonable(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, np.generic):
        return v.item()
    return str(v)
