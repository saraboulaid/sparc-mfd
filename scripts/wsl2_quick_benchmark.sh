#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/sparc-mfd"
source .venv/bin/activate

echo "[bench] METR-LA quick with scipy + 4 workers ..."
time python -u run_experiment.py --dataset metr-la --quick \
  --routing-backend scipy \
  --od-workers 4 \
  --no-resume 2>&1 | tail -30

echo "[bench] Done."
