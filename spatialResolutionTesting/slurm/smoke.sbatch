#!/bin/bash
# GPU smoke test (see scripts/smoke_test.py). Use slurm/submit_smoke.sh, which
# adds your cluster.env settings. The log must end with "ALL STAGES PASSED".
#SBATCH --job-name=locemb-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/activate_env.sh
echo "[job] id=${SLURM_JOB_ID} host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} start=$(date)"
nvidia-smi || echo "[job] nvidia-smi failed"
python scripts/smoke_test.py "$@"
rc=$?
echo "[job] end=$(date) exit=$rc"
exit $rc
