#!/usr/bin/env bash
# Run every experiment back-to-back. Useful for refreshing the diploma's
# results table in one shot before a thesis revision.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "===> Centralized baseline"
./scripts/run_centralized.sh

echo "===> Vanilla federated"
./scripts/run_federated.sh

echo "===> Federated + DP"
./scripts/run_federated_dp.sh

echo "===> Federated + SecAgg"
python experiments/run_federated_secagg.py --config configs/experiment_secagg.yaml

echo "===> Full privacy pipeline (DP + SecAgg + Compression)"
./scripts/run_full_pipeline.sh

echo
echo "All experiments finished. Results under results/"
