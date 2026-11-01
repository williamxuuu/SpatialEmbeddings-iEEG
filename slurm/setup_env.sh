#!/bin/bash
# One-time setup, run on a LOGIN node (needs internet):
#   cp slurm/cluster.env.example slurm/cluster.env   # then edit it
#   bash slurm/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f slurm/cluster.env ] && { set -a; source slurm/cluster.env; set +a; } \
  || echo "NOTE: slurm/cluster.env not found, using defaults"
if type module >/dev/null 2>&1 && [ -n "${LOCEMB_PYTHON_MODULE:-}" ]; then
  module load "$LOCEMB_PYTHON_MODULE"
fi
ENV_DIR="${LOCEMB_ENV_DIR:-$PWD/.venv}"
echo ">> creating env at $ENV_DIR with $(python3 --version)"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
# On Linux the default torch wheel already includes CUDA support.
pip install -e ".[dev,neuroprobe]"
echo ">> caching brain atlases (compute nodes may have no internet)"
LOCEMB_ATLAS_DIR="${LOCEMB_ATLAS_DIR:-$PWD/nilearn_data}" python scripts/fetch_atlases.py
echo ">> running CPU tests"
python -m pytest -q
echo ">> done. Next: bash slurm/submit_sweep.sh configs/sweeps/smoke.yaml"
