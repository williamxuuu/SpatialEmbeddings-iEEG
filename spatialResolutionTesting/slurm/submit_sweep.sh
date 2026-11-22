#!/bin/bash
# Submit a whole sweep as one job array.
#   bash slurm/submit_sweep.sh configs/sweeps/exp1_resolution.yaml            # all runs
#   bash slurm/submit_sweep.sh configs/sweeps/exp1_resolution.yaml 8          # at most 8 at once
#   bash slurm/submit_sweep.sh configs/sweeps/exp1_resolution.yaml 8 --time=02:00:00   # extra sbatch flags
#   LOCEMB_PER_TASK=4 bash slurm/submit_sweep.sh ...   # 4 runs one after another per array task
#     (set --time to cover all of them: ~4x one run's time plus margin)
# Finished runs are skipped automatically, so re-running this after failures
# only redoes what's missing.
set -euo pipefail
SWEEP="${1:?usage: submit_sweep.sh <sweep.yaml> [max_parallel] [extra sbatch args]}"; shift
SWEEP="$(cd "$(dirname "$SWEEP")" && pwd)/$(basename "$SWEEP")"   # works from any folder
cd "$(dirname "$0")/.."
MAXPAR=16
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then MAXPAR="$1"; shift; fi
source slurm/activate_env.sh
read -r N FILE < <(python -m locemb.sweep "$SWEEP")
# Queue only runs without metrics.json (LOCEMB_ONLY_PENDING=0 queues all; finished
# ones are still skipped at run time, but each would hold a GPU slot briefly).
if [ "${LOCEMB_ONLY_PENDING:-1}" = 1 ]; then
  read -r N FILE < <(python scripts/pending_runs.py "$FILE")
  if [ "$N" -eq 0 ]; then echo ">> nothing to do: every run has metrics.json"; exit 0; fi
fi
mkdir -p logs
PER_TASK="${LOCEMB_PER_TASK:-1}"
NTASK=$(( (N + PER_TASK - 1) / PER_TASK ))
ARGS=(--array="0-$((NTASK-1))%${MAXPAR}" --job-name="$(basename "$FILE" .jsonl)")
[ -n "${LOCEMB_ACCOUNT:-}" ]    && ARGS+=(--account="$LOCEMB_ACCOUNT")
[ -n "${LOCEMB_QOS:-}" ]        && ARGS+=(--qos="$LOCEMB_QOS")
[ -n "${LOCEMB_PARTITION:-}" ]  && ARGS+=(--partition="$LOCEMB_PARTITION")
[ -n "${LOCEMB_GRES:-}" ]       && ARGS+=(--gres="$LOCEMB_GRES")
[ -n "${LOCEMB_CONSTRAINT:-}" ] && ARGS+=(--constraint="$LOCEMB_CONSTRAINT")
[ -n "${LOCEMB_TIME:-}" ]       && ARGS+=(--time="$LOCEMB_TIME")
[ -n "${LOCEMB_CPUS:-}" ]       && ARGS+=(--cpus-per-task="$LOCEMB_CPUS")
[ -n "${LOCEMB_MEM:-}" ]        && ARGS+=(--mem="$LOCEMB_MEM")
echo ">> $N runs from $FILE in $NTASK array tasks ($PER_TASK per task)"
echo ">> sbatch ${ARGS[*]} $* slurm/array.sbatch $FILE $PER_TASK $N"
sbatch "${ARGS[@]}" "$@" slurm/array.sbatch "$FILE" "$PER_TASK" "$N"
