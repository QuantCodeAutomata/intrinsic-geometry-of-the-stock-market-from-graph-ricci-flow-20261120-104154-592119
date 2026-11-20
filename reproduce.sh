#!/usr/bin/env bash
# Runs the unit test suite and both experiments from the repository root.
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp1
python -m exp.exp2

echo "Done. See results/RESULTS.md for the comparison against the paper."
