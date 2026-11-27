#!/bin/bash
# Held-out shafts + extended new-person chain (2026-11-27), from spatialResolutionTesting/:
#   bash slurm/submit_ext.sh
# 1 smoke_ext: 10 real runs at once
# 2 check_smoke_ext (afterok 1): FAILS unless scripts/check_smoke_ext.py passes
# 3 heldout_shafts (288 runs, 4 per task) and main_xsubj_ext (432 runs, 6 per task), afterok 2
# 4 reports xheld / xsubj2, afterany on their sweep
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/activate_env.sh
ACC=(--account="$LOCEMB_ACCOUNT" --qos="$LOCEMB_QOS")
jid() { grep -oE '[0-9]+$' | tail -1; }
S=$(LOCEMB_PER_TASK=1 bash slurm/submit_sweep.sh configs/sweeps/smoke_ext.yaml 10 --time=01:00:00 | jid)
C=$(sbatch --parsable "${ACC[@]}" --dependency=afterok:"$S" slurm/check_smoke_ext.sbatch)
H=$(LOCEMB_PER_TASK=4 bash slurm/submit_sweep.sh configs/sweeps/heldout_shafts.yaml 12 --time=03:30:00 --dependency=afterok:"$C" | jid)
Y=$(LOCEMB_PER_TASK=6 bash slurm/submit_sweep.sh configs/sweeps/main_xsubj_ext.yaml 12 --time=02:30:00 --dependency=afterok:"$C" | jid)
RH=$(sbatch --parsable "${ACC[@]}" --dependency=afterany:"$H" slurm/main_report.sbatch xheld)
RY=$(sbatch --parsable "${ACC[@]}" --dependency=afterany:"$Y" slurm/main_report.sbatch xsubj2)
echo "smoke=$S check=$C heldout_shafts=$H main_xsubj_ext=$Y report_held=$RH report_xsubj2=$RY"
