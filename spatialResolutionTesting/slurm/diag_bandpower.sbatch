#!/bin/bash
#SBATCH --job-name=diag_bandpower
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Diagnostic: per-window band power on all 12 Lite sessions (scripts/diag_bandpower.py).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/activate_env.sh
_local="${TMPDIR:-/tmp}/locemb_cache_${SLURM_JOB_ID}"
mkdir -p "$_local"; trap 'rm -rf "$_local"' EXIT
rsync -a "$LOCEMB_CACHE_DIR/" "$_local/" && export LOCEMB_CACHE_DIR="$_local"
echo "[diag] cache at $LOCEMB_CACHE_DIR $(date)"
python scripts/diag_bandpower.py --out "${1:-reports/diag/bandpower}" --workers 7
