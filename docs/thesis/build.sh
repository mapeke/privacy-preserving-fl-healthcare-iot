#!/usr/bin/env bash
# Build docs/thesis/main.pdf with two pdflatex passes (resolves TOC + cross-refs).
# Requires a TeX distribution (TinyTeX, MiKTeX, TeX Live) on PATH.

set -euo pipefail
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null

echo "Built $(pwd)/main.pdf"
