#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/sparc-mfd"
source .venv/bin/activate

echo "[run] METR-LA FULL (scipy, 6 workers, resume E3) ..."
python -u run_experiment.py --dataset metr-la \
  --routing-backend scipy \
  --od-workers 6

echo "[run] PEMS-BAY FULL ..."
python -u run_experiment.py --dataset pems-bay --no-resume \
  --routing-backend scipy \
  --od-workers 6

echo "[run] Done."
