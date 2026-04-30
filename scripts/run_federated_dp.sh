#!/usr/bin/env bash
# Run FedAvg + differential privacy (DP-SGD).
# Usage: ./scripts/run_federated_dp.sh [--target-epsilon E] [--num-clients N] [--rounds R]

set -euo pipefail
cd "$(dirname "$0")/.."

python experiments/run_federated_dp.py --config configs/experiment_dp.yaml "$@"
