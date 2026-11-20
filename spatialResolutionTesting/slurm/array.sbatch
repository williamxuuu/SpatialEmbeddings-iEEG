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
# Arguments: <sweep.jsonl> [runs per task K, default 1] [total runs N].
# With K > 1, array task i runs sweep lines i*K .. i*K+K-1 one after another
# (fewer, longer jobs: less queueing, and the recordings stay in the node's
# file cache). A run that fails doesn't stop the others; the task exits non-zero
# at the end if any failed, and re-submitting redoes only runs without metrics.json.
set -uo pipefail
SWEEP_FILE="$1"; PER_TASK="${2:-1}"; N_RUNS="${3:-}"
cd "${SLURM_SUBMIT_DIR}"
source slurm/activate_env.sh
[ -z "$N_RUNS" ] && N_RUNS=$(grep -c . "$SWEEP_FILE")
echo "[job] array=${SLURM_ARRAY_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-?} runs/task=$PER_TASK host=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[job] no GPU visible"
rc=0
first=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
for (( i = first; i < first + PER_TASK && i < N_RUNS; i++ )); do
  echo "[job] run index $i start=$(date)"
  python -m locemb.train --sweep-file "$SWEEP_FILE" --index "$i" \
    "train.num_workers=$(( ${SLURM_CPUS_PER_TASK:-2} - 1 ))" || { echo "[job] run index $i FAILED"; rc=1; }
done
echo "[job] end=$(date) exit=$rc"
exit $rc
