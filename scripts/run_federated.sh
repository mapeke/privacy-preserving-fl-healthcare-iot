#!/usr/bin/env bash
# Run vanilla FedAvg (no privacy defenses).
# Usage: ./scripts/run_federated.sh [--num-clients N] [--rounds R]

set -euo pipefail
cd "$(dirname "$0")/.."

python experiments/run_federated.py --config configs/default.yaml "$@"
