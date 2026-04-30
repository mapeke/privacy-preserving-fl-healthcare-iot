"""Federated learning with differential privacy (DP-SGD via Opacus)."""

from __future__ import annotations

import argparse

from experiments._common import build_experiment, run_federated_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_dp.yaml")
    parser.add_argument("--num-clients", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--target-epsilon", type=float)
    args = parser.parse_args()

    overrides: dict[str, object] = {}
    if args.num_clients is not None:
        overrides["federation.num_clients"] = args.num_clients
        overrides["federation.clients_per_round"] = args.num_clients
    if args.rounds is not None:
        overrides["federation.num_rounds"] = args.rounds
    if args.target_epsilon is not None:
        overrides["privacy.differential_privacy.target_epsilon"] = args.target_epsilon

    art = build_experiment(args.config, overrides=overrides)
    run_federated_experiment(art)


if __name__ == "__main__":
    main()
