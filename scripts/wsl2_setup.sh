#!/usr/bin/env bash
set -euo pipefail

SRC="/mnt/c/projets_doctorat/sparc-mfd"
DEST="$HOME/sparc-mfd"

if [ ! -d "$DEST" ] || [ ! -f "$DEST/run_experiment.py" ]; then
  echo "[setup] Copying project to $DEST ..."
  mkdir -p "$DEST"
  rsync -a --exclude '.venv' --exclude '__pycache__' "$SRC/" "$DEST/"
else
  echo "[setup] Project already at $DEST, syncing outputs/checkpoints ..."
  mkdir -p "$DEST/outputs"
  rsync -a "$SRC/outputs/metr-la/" "$DEST/outputs/metr-la/" 2>/dev/null || true
  rsync -a "$SRC/outputs/routing_regression_report.json" "$DEST/outputs/" 2>/dev/null || true
fi

cd "$DEST"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# CPU-only torch first (avoid multi-GB CUDA wheels on WSL)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas h5py networkx matplotlib scipy pytest

python -c "import scipy; print('scipy', scipy.__version__)"
python -c "import torch; print('torch', torch.__version__)"
python scripts/validate_datasets.py --require-full

echo "[setup] OK -> $DEST"
