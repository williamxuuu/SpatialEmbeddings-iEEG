#!/bin/bash
# Submit the GPU smoke test with your cluster.env settings.
#   bash slurm/submit_smoke.sh          # synthetic data, ~5 min
#   bash slurm/submit_smoke.sh real     # real Brain Treebank (cache must be built first)
#   bash slurm/submit_smoke.sh real --time=00:45:00     # extra sbatch flags after the mode
# Then: tail -f logs/locemb-smoke_<jobid>.out   (must end "ALL STAGES PASSED")
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=synthetic; PYARGS=(); TIME="00:20:00"
if [ "${1:-}" = "real" ]; then MODE=real; PYARGS=(--real); TIME="00:45:00"; shift; fi
[ -f slurm/cluster.env ] && { set -a; source slurm/cluster.env; set +a; }
mkdir -p logs
ARGS=(--job-name="locemb-smoke-$MODE" --output="logs/locemb-smoke-${MODE}_%j.out" --time="$TIME")
[ -n "${LOCEMB_ACCOUNT:-}" ]    && ARGS+=(--account="$LOCEMB_ACCOUNT")
[ -n "${LOCEMB_QOS:-}" ]        && ARGS+=(--qos="$LOCEMB_QOS")
[ -n "${LOCEMB_PARTITION:-}" ]  && ARGS+=(--partition="$LOCEMB_PARTITION")
[ -n "${LOCEMB_GRES:-}" ]       && ARGS+=(--gres="$LOCEMB_GRES")
[ -n "${LOCEMB_CONSTRAINT:-}" ] && ARGS+=(--constraint="$LOCEMB_CONSTRAINT")
[ -n "${LOCEMB_CPUS:-}" ]       && ARGS+=(--cpus-per-task="$LOCEMB_CPUS")
[ -n "${LOCEMB_MEM:-}" ]        && ARGS+=(--mem="$LOCEMB_MEM")
echo ">> sbatch ${ARGS[*]} $* slurm/smoke.sbatch ${PYARGS[*]:-}"
JOB=$(sbatch --parsable "${ARGS[@]}" "$@" slurm/smoke.sbatch "${PYARGS[@]}")
echo ">> submitted job $JOB; log: logs/locemb-smoke-${MODE}_${JOB}.out"
