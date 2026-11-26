#!/bin/bash
# Main experiment chain (2026-11-26), from spatialResolutionTesting/:
#   bash slurm/submit_main.sh
# 1 smoke_main: 10 real runs at once (epoch times under concurrency, new knobs)
# 2 check_smoke_main (afterok 1): FAILS unless scripts/check_smoke_main.py passes
# 3 main_xsess (192 runs, 4 per task) and main_xsubj (144 runs, 6 per task), afterok 2
# 4 main_report xsess / xsubj, afterany on their sweep
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/activate_env.sh
ACC=(--account="$LOCEMB_ACCOUNT" --qos="$LOCEMB_QOS")
jid() { grep -oE '[0-9]+$' | tail -1; }
S=$(LOCEMB_PER_TASK=1 bash slurm/submit_sweep.sh configs/sweeps/smoke_main.yaml 10 --time=01:00:00 | jid)
C=$(sbatch --parsable "${ACC[@]}" --dependency=afterok:"$S" slurm/check_smoke_main.sbatch)
X=$(LOCEMB_PER_TASK=4 bash slurm/submit_sweep.sh configs/sweeps/main_xsess.yaml 12 --time=03:30:00 --dependency=afterok:"$C" | jid)
Y=$(LOCEMB_PER_TASK=6 bash slurm/submit_sweep.sh configs/sweeps/main_xsubj.yaml 12 --time=02:30:00 --dependency=afterok:"$C" | jid)
RX=$(sbatch --parsable "${ACC[@]}" --dependency=afterany:"$X" slurm/main_report.sbatch xsess)
RY=$(sbatch --parsable "${ACC[@]}" --dependency=afterany:"$Y" slurm/main_report.sbatch xsubj)
echo "smoke=$S check=$C main_xsess=$X main_xsubj=$Y report_xsess=$RX report_xsubj=$RY"
