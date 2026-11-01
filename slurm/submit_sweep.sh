#!/bin/bash
# Submit a whole sweep as one job array.
#   bash slurm/submit_sweep.sh configs/sweeps/exp1_resolution.yaml            # all runs
#   bash slurm/submit_sweep.sh configs/sweeps/exp1_resolution.yaml 8          # at most 8 at once
#   bash slurm/submit_sweep.sh configs/sweeps/exp1_resolution.yaml 8 --time=02:00:00   # extra sbatch flags
# Finished runs are skipped automatically, so re-running this after failures
# only redoes what's missing.
set -euo pipefail
cd "$(dirname "$0")/.."
SWEEP="${1:?usage: submit_sweep.sh <sweep.yaml> [max_parallel] [extra sbatch args]}"; shift
MAXPAR=16
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then MAXPAR="$1"; shift; fi
source slurm/activate_env.sh
read -r N FILE < <(python -m locemb.sweep "$SWEEP")
mkdir -p logs
ARGS=(--array="0-$((N-1))%${MAXPAR}" --job-name="$(basename "$FILE" .jsonl)")
[ -n "${LOCEMB_ACCOUNT:-}" ]    && ARGS+=(--account="$LOCEMB_ACCOUNT")
[ -n "${LOCEMB_QOS:-}" ]        && ARGS+=(--qos="$LOCEMB_QOS")
[ -n "${LOCEMB_PARTITION:-}" ]  && ARGS+=(--partition="$LOCEMB_PARTITION")
[ -n "${LOCEMB_GRES:-}" ]       && ARGS+=(--gres="$LOCEMB_GRES")
[ -n "${LOCEMB_CONSTRAINT:-}" ] && ARGS+=(--constraint="$LOCEMB_CONSTRAINT")
[ -n "${LOCEMB_TIME:-}" ]       && ARGS+=(--time="$LOCEMB_TIME")
[ -n "${LOCEMB_CPUS:-}" ]       && ARGS+=(--cpus-per-task="$LOCEMB_CPUS")
[ -n "${LOCEMB_MEM:-}" ]        && ARGS+=(--mem="$LOCEMB_MEM")
echo ">> $N runs from $FILE"
echo ">> sbatch ${ARGS[*]} $* slurm/array.sbatch $FILE"
sbatch "${ARGS[@]}" "$@" slurm/array.sbatch "$FILE"
