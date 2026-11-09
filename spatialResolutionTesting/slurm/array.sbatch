#!/bin/bash
# One sweep = one Slurm "job array". Each array task runs ONE line of the sweep
# file. Don't sbatch this directly; use slurm/submit_sweep.sh, which fills in
# the array size and your cluster settings.
#SBATCH --job-name=locemb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --requeue
set -euo pipefail
SWEEP_FILE="$1"
cd "${SLURM_SUBMIT_DIR}"
source slurm/activate_env.sh
echo "[job] array=${SLURM_ARRAY_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-?} host=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[job] no GPU visible"
python -m locemb.train --sweep-file "$SWEEP_FILE" --index "${SLURM_ARRAY_TASK_ID}" \
  "train.num_workers=$(( ${SLURM_CPUS_PER_TASK:-2} - 1 ))"
echo "[job] end=$(date)"
