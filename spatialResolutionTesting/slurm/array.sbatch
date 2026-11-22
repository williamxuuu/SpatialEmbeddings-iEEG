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
# Stage the session cache onto node-local disk. Runs memory-map data.npy, and
# when it lives on scratch (Lustre) the DataLoader workers intermittently hang
# in cl_sync_io_wait once the page cache drops pages: on 2026-11-22 single
# epochs took 20 min to 3.3 h (normally ~16 s) with the GPU at 0%, and one task
# timed out. The cache is ~15 GB; one sequential copy takes ~1 min. Set
# LOCEMB_LOCAL_CACHE=0 to read from scratch directly.
if [ "${LOCEMB_LOCAL_CACHE:-1}" = 1 ] && [ -n "${LOCEMB_CACHE_DIR:-}" ] && [ -d "$LOCEMB_CACHE_DIR" ]; then
  _local="${TMPDIR:-/tmp}/locemb_cache_${SLURM_JOB_ID:-$$}"
  _need_kb=$(du -sk "$LOCEMB_CACHE_DIR" | cut -f1)
  _free_kb=$(df -Pk "${TMPDIR:-/tmp}" | awk 'NR==2{print $4}')
  if (( _free_kb > _need_kb * 12 / 10 )); then
    _t0=$(date +%s)
    mkdir -p "$_local"
    trap 'rm -rf "$_local"' EXIT
    if timeout 1800 rsync -a "$LOCEMB_CACHE_DIR/" "$_local/"; then
      echo "[job] staged cache ($(( _need_kb / 1048576 )) GB) to $_local in $(( $(date +%s) - _t0 ))s"
      export LOCEMB_CACHE_DIR="$_local"
    else
      echo "[job] WARN: staging cache failed; reading from $LOCEMB_CACHE_DIR"
      rm -rf "$_local"
    fi
  else
    echo "[job] WARN: not enough local disk for the cache; reading from $LOCEMB_CACHE_DIR"
  fi
fi
rc=0
first=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
for (( i = first; i < first + PER_TASK && i < N_RUNS; i++ )); do
  echo "[job] run index $i start=$(date)"
  python -m locemb.train --sweep-file "$SWEEP_FILE" --index "$i" \
    "train.num_workers=$(( ${SLURM_CPUS_PER_TASK:-2} - 1 ))" || { echo "[job] run index $i FAILED"; rc=1; }
done
echo "[job] end=$(date) exit=$rc"
exit $rc
