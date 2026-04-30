#!/usr/bin/env bash
# Run the headline diploma experiment: DP + SecAgg + compression.
# Usage: ./scripts/run_full_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

python experiments/run_full_pipeline.py --config configs/experiment_full.yaml "$@"
