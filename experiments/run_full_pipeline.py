"""End-to-end privacy pipeline: DP-SGD + Secure Aggregation + Compression.

This is the headline experiment for the diploma. It exercises every component of the framework
and writes a JSON results file (``results/federated_full_pipeline/history.json``) that can be
plotted directly into the thesis.

Usage::

    python experiments/run_full_pipeline.py
    python experiments/run_full_pipeline.py --num-clients 4 --rounds 30 --target-epsilon 8.0
"""

from __future__ import annotations

import argparse

from experiments._common import build_experiment, run_federated_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_full.yaml")
    parser.add_argument("--num-clients", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--target-epsilon", type=float)
    parser.add_argument("--top-k-ratio", type=float)
    args = parser.parse_args()

    overrides: dict[str, object] = {}
    if args.num_clients is not None:
        overrides["federation.num_clients"] = args.num_clients
        overrides["federation.clients_per_round"] = args.num_clients
    if args.rounds is not None:
        overrides["federation.num_rounds"] = args.rounds
    if args.target_epsilon is not None:
        overrides["privacy.differential_privacy.target_epsilon"] = args.target_epsilon
    if args.top_k_ratio is not None:
        overrides["privacy.compression.top_k_ratio"] = args.top_k_ratio

    art = build_experiment(args.config, overrides=overrides)
    run_federated_experiment(art)


if __name__ == "__main__":
    main()
