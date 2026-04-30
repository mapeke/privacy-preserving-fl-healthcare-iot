#!/usr/bin/env bash
# Run the centralized (no-privacy) baseline.
# Usage: ./scripts/run_centralized.sh [extra args passed to the python script]

set -euo pipefail
cd "$(dirname "$0")/.."

python experiments/run_centralized.py --config configs/default.yaml --epochs 5 "$@"
