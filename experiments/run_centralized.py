"""Centralized baseline: pool every client's data and train a single model.

This is the upper bound the federated runs are compared against. It is **not** privacy-preserving
and exists only as a reference number for the diploma's results table.

Usage::

    python experiments/run_centralized.py
    python experiments/run_centralized.py --config configs/default.yaml --epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from experiments._common import build_experiment
from src.client.dp_trainer import evaluate, train_one_round
from src.utils.metrics import compute_classification_report

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", default="results/centralized")
    args = parser.parse_args()

    art = build_experiment(args.config, overrides={"experiment.output_dir": args.output_dir})

    pooled = ConcatDataset([loader.dataset for loader in art.train_loaders])
    pooled_loader = DataLoader(
        pooled,
        batch_size=int(art.cfg["federation"]["local_batch_size"]),
        shuffle=True,
    )

    model = art.model_factory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Centralized baseline: epochs=%d, device=%s", args.epochs, device)

    train_result = train_one_round(
        model,
        pooled_loader,
        epochs=args.epochs,
        learning_rate=float(art.cfg["optimization"]["learning_rate"]),
        momentum=float(art.cfg["optimization"].get("momentum", 0.9)),
        weight_decay=float(art.cfg["optimization"].get("weight_decay", 0.0)),
        dp_config=None,
        device=device,
    )
    test_loss, test_acc, y_true, y_pred = evaluate(model, art.test_loader_global, device=device)
    report = compute_classification_report(np.asarray(y_true), np.asarray(y_pred))

    out: dict[str, object] = {
        "train_loss": train_result.final_loss,
        "train_accuracy": train_result.final_accuracy,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "report": report.to_dict(),
        "epochs": args.epochs,
    }

    out_path: Path = art.output_dir / "centralized_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote -> %s", out_path)


if __name__ == "__main__":
    main()
