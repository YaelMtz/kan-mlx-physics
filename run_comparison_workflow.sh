#!/usr/bin/env bash
# Full toy-comparison workflow: run the 4-way benchmark, render the figure,
# print the report, and open the figure. Meant to be launched in a Ghostty
# window so you watch it run end-to-end.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
echo "===== KAN toy comparison: fit sin(pi x) ====="
echo "[1/2] running 4-way benchmark (MLX spline / spline-compiled / Fourier / pykan)..."
uv run python3 toy_comparison.py
echo
echo "[2/2] rendering comparison figure..."
uv run python3 toy_comparison_plot.py
echo
echo "Opening figure..."
open toy_comparison.png || true
echo "Report: $DIR/TOY_COMPARISON.md"
