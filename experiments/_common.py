"""Shared plumbing for experiment entry points.

Each ``experiments/run_*.py`` script is a thin CLI wrapper. The bulk of the work — loading data,
partitioning across clients, building the model, wiring the privacy stack — lives here so the
scripts stay short and the diploma writeup can cite a single integration point.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.client.flower_client import ClientConfig, FlowerHealthcareClient
from src.client.iot_device import default_profiles
from src.data.mitbih_loader import generate_synthetic_ecg, load_mitbih
from src.data.partitioner import partition
from src.data.preprocessing import TorchEcgDataset, stratified_train_test_split
from src.models import build_model
from src.privacy import secure_agg
from src.privacy.differential_privacy import DPConfig
from src.server.flower_server import (
    SimulationConfig,
    history_to_dict,
    make_initial_parameters,
    run_simulation,
)
from src.utils.config import load_config, save_config
from src.utils.logging_utils import setup_logging
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class ExperimentArtifacts:
    """Bag of objects an experiment script needs after building from a config."""

    cfg: dict[str, Any]
    output_dir: Path
    train_loaders: list[DataLoader]
    test_loader_global: DataLoader
    model_factory: Any
    iot_profiles: list
    dp_config: DPConfig | None
    secagg_session: secure_agg.SecAggSession | None
    compression_method: str | None
    compression_top_k_ratio: float
    compression_bits: int


def build_experiment(config_path: str, overrides: dict[str, Any] | None = None) -> ExperimentArtifacts:
    """Materialise everything an experiment needs from a YAML config."""
    cfg = load_config(config_path, overrides=overrides)

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=cfg["logging"]["level"], log_file=output_dir / "run.log")
    set_global_seed(int(cfg["experiment"]["seed"]))

    save_config(cfg, output_dir / "resolved_config.yaml")

    # ---- Data --------------------------------------------------------------
    if cfg["data"]["source"] == "mitbih":
        dataset = load_mitbih(
            records=tuple(cfg["data"]["records"]),
            cache_dir=cfg["data"]["cache_dir"],
            window_size=int(cfg["data"]["window_size"]),
        )
    else:
        dataset = generate_synthetic_ecg(
            records=tuple(cfg["data"]["records"]),
            window_size=int(cfg["data"]["window_size"]),
        )

    if dataset.is_synthetic:
        logger.warning("Using synthetic ECG data — results are illustrative, not clinical.")

    split = stratified_train_test_split(
        dataset,
        test_fraction=float(cfg["data"]["test_fraction"]),
        seed=int(cfg["experiment"]["seed"]),
    )

    client_partitions = partition(
        split.train,
        num_clients=int(cfg["federation"]["num_clients"]),
        scheme=str(cfg["partitioning"]["scheme"]),
        alpha=float(cfg["partitioning"].get("dirichlet_alpha", 0.5)),
        min_samples_per_client=int(cfg["partitioning"].get("min_samples_per_client", 10)),
        seed=int(cfg["experiment"]["seed"]),
    )

    batch_size = int(cfg["federation"]["local_batch_size"])
    train_loaders = [
        DataLoader(TorchEcgDataset(p), batch_size=batch_size, shuffle=True, drop_last=False)
        for p in client_partitions
    ]
    test_loader_global = DataLoader(
        TorchEcgDataset(split.test), batch_size=batch_size, shuffle=False
    )

    # ---- Privacy stack -----------------------------------------------------
    dp_block = cfg["privacy"]["differential_privacy"]
    dp_config: DPConfig | None = None
    if dp_block.get("enabled", False):
        dp_config = DPConfig(
            target_epsilon=float(dp_block["target_epsilon"]),
            target_delta=float(dp_block["target_delta"]),
            max_grad_norm=float(dp_block["max_grad_norm"]),
            noise_multiplier=dp_block.get("noise_multiplier"),
            accountant=str(dp_block.get("accountant", "rdp")),
        )

    secagg_session: secure_agg.SecAggSession | None = None
    if cfg["privacy"]["secure_aggregation"].get("enabled", False):
        secagg_session = secure_agg.setup_session(
            num_clients=int(cfg["federation"]["num_clients"]),
            seed_size_bytes=int(cfg["privacy"]["secure_aggregation"].get("seed_size_bytes", 32)),
            rng_seed=int(cfg["experiment"]["seed"]),  # deterministic for reproducible experiments
        )

    comp_block = cfg["privacy"]["compression"]
    if comp_block.get("enabled", False):
        compression_method = str(comp_block.get("method", "top_k_quantize"))
        compression_top_k_ratio = float(comp_block.get("top_k_ratio", 0.1))
        compression_bits = int(comp_block.get("quant_bits", 8))
    else:
        compression_method = None
        compression_top_k_ratio = 0.1
        compression_bits = 8

    # ---- Model factory + IoT profiles -------------------------------------
    model_name = str(cfg["model"]["name"])
    dropout = float(cfg["model"].get("dropout", 0.3))
    num_classes = int(cfg["data"]["num_classes"])

    def model_factory():
        return build_model(model_name, num_classes=num_classes, dropout=dropout)

    iot_profiles = default_profiles(int(cfg["federation"]["num_clients"]))

    return ExperimentArtifacts(
        cfg=cfg,
        output_dir=output_dir,
        train_loaders=train_loaders,
        test_loader_global=test_loader_global,
        model_factory=model_factory,
        iot_profiles=iot_profiles,
        dp_config=dp_config,
        secagg_session=secagg_session,
        compression_method=compression_method,
        compression_top_k_ratio=compression_top_k_ratio,
        compression_bits=compression_bits,
    )


def make_client_factory(art: ExperimentArtifacts):
    """Return a Flower-compatible ``cid -> NumPyClient`` factory."""
    cfg = art.cfg
    fed = cfg["federation"]
    opt = cfg["optimization"]

    def factory(cid: str) -> FlowerHealthcareClient:
        client_id = int(cid)
        model = art.model_factory()
        client_cfg = ClientConfig(
            client_id=client_id,
            local_epochs=int(fed["local_epochs"]),
            learning_rate=float(opt["learning_rate"]),
            momentum=float(opt.get("momentum", 0.9)),
            weight_decay=float(opt.get("weight_decay", 0.0)),
            device="cuda" if torch.cuda.is_available() else "cpu",
            dp_config=art.dp_config,
            secagg_session=art.secagg_session,
            compression_method=art.compression_method,
            compression_top_k_ratio=art.compression_top_k_ratio,
            compression_bits=art.compression_bits,
            iot_profile=art.iot_profiles[client_id],
        )
        return FlowerHealthcareClient(
            model=model,
            train_loader=art.train_loaders[client_id],
            test_loader=art.test_loader_global,
            config=client_cfg,
        )

    return factory


def run_federated_experiment(art: ExperimentArtifacts) -> dict[str, Any]:
    """Run the Flower simulation defined by ``art`` and persist a JSON results file."""
    initial_model = art.model_factory()
    sim = SimulationConfig(
        num_clients=int(art.cfg["federation"]["num_clients"]),
        num_rounds=int(art.cfg["federation"]["num_rounds"]),
        fraction_fit=(
            int(art.cfg["federation"]["clients_per_round"])
            / int(art.cfg["federation"]["num_clients"])
        ),
        secagg_enabled=art.secagg_session is not None,
        initial_parameters=make_initial_parameters(initial_model),
    )

    history = run_simulation(make_client_factory(art), sim)
    results = history_to_dict(history)

    results_path = art.output_dir / "history.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_fallback)
    logger.info("Wrote results -> %s", results_path)
    return results


def _json_fallback(o: Any):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
