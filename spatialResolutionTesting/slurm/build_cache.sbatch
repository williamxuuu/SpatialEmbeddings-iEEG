#!/bin/bash
# Filter the 12 Lite recordings once (CPU only, one array task per recording).
#   sbatch --account=... --qos=... slurm/build_cache.sbatch
# Then submit sweeps with --dependency=afterok:<this job id>, or just wait.
# Measured on a laptop: ~1.4 s of filtering per electrode per 2.5 h of recording,
# plus reading the file: expect ~5-15 min per recording, ~17 GB written in total.
#SBATCH --job-name=btb-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/activate_env.sh
python scripts/build_session_cache.py --config configs/neuroprobe.yaml --index "${SLURM_ARRAY_TASK_ID}"
